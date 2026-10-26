"""Fine-tune one enhancement member (MossFormerGAN, 104M BSRNN or TF-Locoformer) for Track 2.

Loss: SI-SDR (MossFormerGAN adds its compressed-spectrum L1) + the scoring ASR's encoder distance (--enc-loss) + TDT
transcript NLL of the clean speech's teacher tokens (--nll-loss) + optional deletion-weighted sampled MWER (--mwer) on
windows of 8 s or more. With --long-beds, every step adds one micro-batch of 12-18 s single-speaker windows trained with
the ASR losses only. Held-out protocol (default): the 209 held-out validation pairs are excluded from training and scored
every --eval-every steps; --full trains on all 695 pairs. Weights are tracked with an EMA, which is what is saved.

Configurations for every stage of the final members are in configs/; scripts/train_all.sh runs them in order.

  python train/train_member.py --config configs/loco_LS.yaml
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import track2.losses as L  # noqa: E402
from data import BINS, W_P2, LongStream, Stream, coverage, load_meta  # noqa: E402
from track2.audio import si_sdr  # noqa: E402
from track2.members import MEMBERS  # noqa: E402
from track2.post import post  # noqa: E402

CKPT_UNITS = {"mossgan": ("MossFormer",), "bsrnn_xl": ("LSTM",), "loco": ("TFLocoformerBlock",)}


def memory_saving(net, units):
    """Attention through the fused SDPA kernel (bf16) and activation checkpointing per block."""
    _mha = torch.nn.MultiheadAttention.forward

    def mha(self, *args, **kw):
        kw["need_weights"] = False
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, _ = _mha(self, *args, **kw)
        return out.float(), None
    torch.nn.MultiheadAttention.forward = mha
    n = 0
    for m in net.modules():
        if type(m).__name__ in units:
            f = m.forward

            def wrapped(*args, _f=f, **kw):
                if torch.is_grad_enabled():
                    return torch.utils.checkpoint.checkpoint(_f, *args, use_reentrant=False, **kw)
                return _f(*args, **kw)
            m.forward = wrapped
            n += 1
    return n


@torch.no_grad()
def evaluate(mem, ev, lab=None):
    """Held-out SI-SDR (raw and post-processed, mean and Phase 2 coverage-reweighted), encoder distance, transcript NLL."""
    mem.net.eval()
    raw, pst, enc, nll = [], [], [], []
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x, np.float32))[None].to(mem.dev)
    for i, (cl, mx, _) in enumerate(ev):
        e = mem.enhance(mx)
        e = e[:len(mx)] if len(e) >= len(mx) else np.concatenate([e, mx[len(e):]])
        raw.append(si_sdr(cl, e))
        pst.append(si_sdr(cl, post(mx, e)))
        if L.ENC_W > 0:
            enc.append(float(L.enc_dist(t(e), t(cl))))
        if lab is not None:
            tk = lab[i].long()[None] if len(lab[i]) else torch.zeros(1, 1, dtype=torch.long)
            nll.append((float(L.tdt_nll(L.asr_encode(t(e)), tk, torch.tensor([0]), torch.tensor([len(lab[i])]),
                                        torch.tensor([[0, 0]]), torch.tensor([[1 << 30, 1 << 30]]))[0]), len(lab[i])))
    mem.net.train()
    b = np.digitize(np.array([c for _, _, c in ev]), BINS) - 1
    rw = lambda v: float(sum(W_P2[k] * np.array(v)[b == k].mean() for k in range(5) if (b == k).any())
                         / sum(W_P2[k] for k in range(5) if (b == k).any()))
    return {"raw": float(np.mean(raw)), "raw_p2": rw(raw), "post": float(np.mean(pst)), "post_p2": rw(pst),
            **({"enc_dist": float(np.mean(enc))} if enc else {}),
            **({"nll": sum(v for v, _ in nll) / max(1, sum(n for _, n in nll))} if nll else {})}


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="", help="yaml with any of the options below (command-line flags override it)")
    ap.add_argument("--model", choices=sorted(MEMBERS))
    ap.add_argument("--out")
    ap.add_argument("--init", default="", help="state dict to start from (default: the public pretrained checkpoint)")
    ap.add_argument("--pretrained", default="", help="TF-Locoformer only: MERL's DNS checkpoint used when --init is empty")
    ap.add_argument("--val-root", default="data/validation", help="organiser validation set (validationMetadata.json, audio folders)")
    ap.add_argument("--noise-bank", default="data/noisebank", help="noise events extracted from the validation pairs (prepare_noisebank.py)")
    ap.add_argument("--teacher", default="data/tdt_teacher", help="SraVaani greedy tokens of every clean bed (make_teacher.py)")
    ap.add_argument("--long-beds", default="", help="manifest.jsonl of 12-18 s single-speaker recordings (prepare_long_beds.py)")
    ap.add_argument("--long-s", default="12,18")
    ap.add_argument("--long-bs", type=int, default=2)
    ap.add_argument("--long-gain-db", type=float, default=0.0, help="shift of the long-stream event gain median, dB")
    ap.add_argument("--full", action="store_true", help="train on all 695 pairs (no honest held-out score then)")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-min", type=float, default=1e-6)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--seg", type=int, default=32000, help="crop length in samples")
    ap.add_argument("--p-real", type=float, default=0.5, help="share of crops taken from the real pairs (rest re-mixed)")
    ap.add_argument("--enc-loss", type=float, default=0.0)
    ap.add_argument("--nll-loss", type=float, default=0.0)
    ap.add_argument("--mwer", type=float, default=0.0)
    ap.add_argument("--mwer-n", type=int, default=4)
    ap.add_argument("--mwer-del", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a0, _ = ap.parse_known_args()
    if a0.config:
        cfg = yaml.safe_load(open(a0.config))
        ap.set_defaults(**{k.replace("-", "_"): v for k, v in cfg.items()})
    a = ap.parse_args()
    if not (a.model and a.out):
        ap.error("--model and --out are required (on the command line or in --config)")
    if a.long_beds and not (a.enc_loss > 0 or a.nll_loss > 0):
        ap.error("--long-beds carries only the ASR losses: set --enc-loss or --nll-loss")
    if a.mwer > 0 and a.nll_loss <= 0:
        ap.error("--mwer needs --nll-loss (it uses the teacher labels)")
    return a


def main():
    a = parse()
    L.ENC_W, L.NLL_W, L.MWER_W, L.MWER_N, L.MWER_DEL = a.enc_loss, a.nll_loss, a.mwer, a.mwer_n, a.mwer_del
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    rows = json.load(open(os.path.join(a.noise_bank, "noisebank.json")))
    held = {r["src_clip"] for r in rows if r["split"] == "heldout"}
    meta = {r["clip_id"]: r for r in load_meta(a.val_root) if r["syntheticData"]}
    pairs = {}
    for c, r in meta.items():
        cl, mx = sf.read(r["clean_path"], dtype="float32")[0], sf.read(r["noisy_path"], dtype="float32")[0]
        k = min(len(cl), len(mx))
        pairs[c] = (mx[:k].copy(), cl[:k].copy())
    train_ids = sorted(c for c in pairs if a.full or c not in held)
    test_ids = sorted(c for c in pairs if c in held)
    noise = [sf.read(os.path.join(a.noise_bank, "noise", r["file"]), dtype="float32")[0] for r in rows]
    snr = []
    for c in train_ids:
        mx, cl = pairs[c]
        ec, en = float((cl ** 2).sum()), float(((mx - cl) ** 2).sum())
        if ec > 0 and en > 0:
            snr.append(10 * np.log10(ec / en))
    long_files, long_beds = [], []
    if a.long_beds:
        root = os.path.dirname(os.path.abspath(a.long_beds))
        long_files = [os.path.join(root, r["path"]) for r in map(json.loads, open(a.long_beds))]
        long_beds = [sf.read(f, dtype="float32")[0] for f in long_files]
    ev = [(pairs[c][1], pairs[c][0], coverage(meta[c])) for c in test_ids]
    beds_lab, ev_lab = None, None
    if L.NLL_W > 0:
        teacher = L.load_teacher(a.teacher)
        tr = [(*teacher[f"val/{c}"], len(pairs[c][1])) for c in train_ids]
        beds_lab = {0: tr, 1: tr, LongStream.POOL: [(*teacher[os.path.abspath(f)], len(x)) for f, x in zip(long_files, long_beds)]}
        ev_lab = [teacher[f"val/{c}"][0] for c in test_ids]
    ds = Stream([pairs[c] for c in train_ids], [pairs[c][1] for c in train_ids], noise, snr, a.steps * a.bs * a.accum, a.seg, a.seed,
                p_real=a.p_real, with_src=L.NLL_W > 0)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=a.workers, drop_last=True, shuffle=False)
    long_items = None
    if long_beds:
        lo, hi = map(float, str(a.long_s).split(","))
        rms = [10 * np.log10(float(np.mean(pairs[c][1].astype(np.float64) ** 2)) + 1e-12) for c in train_ids]
        long_items = iter(torch.utils.data.DataLoader(LongStream(long_beds, noise, rms, a.steps, a.long_bs, lo, hi, a.seed, a.long_gain_db),
                                                      batch_size=None, num_workers=4))

    mem = MEMBERS[a.model](dev, a.init, **({"pretrained": a.pretrained} if a.model == "loco" and not a.init else {}))
    net = mem.net
    print(f"activation checkpointing on {memory_saving(net, CKPT_UNITS[a.model])} blocks", flush=True)
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()}
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = (s - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr_min + 0.5 * (a.lr - a.lr_min) * (1 + np.cos(np.pi * p))

    def batch_loss(batch):
        mx, cl, *src = batch
        lab = L.crop_labels(src[0], beds_lab, a.seg) if src else None
        return mem.loss(mx.to(dev), cl.to(dev), lab)

    def long_loss(batch):
        mx, cl, src = batch
        lab = L.crop_labels(src, beds_lab, mx.shape[-1]) if L.NLL_W > 0 else None
        return mem.asr_loss(mx.to(dev), cl.to(dev), lab)

    log = open(os.path.join(a.out, "log.jsonl"), "a")
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    print(f"train pairs {len(train_ids)} | held-out pairs {len(test_ids)} | noise events {len(noise)} | long beds {len(long_beds)}", flush=True)
    if ev:
        r0 = evaluate(mem, ev, ev_lab)
        print("step 0 " + " ".join(f"{k} {v:.3f}" for k, v in r0.items()), flush=True)
    net.train()
    ls, long_ls, t0 = [], [], time.time()
    batches = iter(dl)
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step - 1)
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            loss = batch_loss(next(batches)) / a.accum
            loss.backward()
            ls.append(float(loss.detach()) * a.accum)
        if long_items is not None:
            loss = long_loss(next(long_items))
            loss.backward()
            long_ls.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        with torch.no_grad():
            for k, v in net.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(a.ema).add_(v.detach(), alpha=1 - a.ema)
                else:
                    ema[k].copy_(v)
        if step % a.eval_every == 0 or step == a.steps:
            cur = {k: v.detach().clone() for k, v in net.state_dict().items()}
            net.load_state_dict(ema)
            res = evaluate(mem, ev, ev_lab) if ev else {}
            torch.save(ema, os.path.join(a.out, f"ema_{step:06d}.pt"))
            net.load_state_dict(cur)
            row = {"step": step, "loss": float(np.mean(ls)), "lr": lr_at(step - 1), **{f"ema_{k}": v for k, v in res.items()},
                   **({"long_loss": float(np.mean(long_ls))} if long_ls else {}), "sec": time.time() - t0}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(" ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
            ls, long_ls = [], []


if __name__ == "__main__":
    main()
