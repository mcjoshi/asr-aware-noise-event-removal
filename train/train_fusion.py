"""Train the per-cell fusion of the three members (track2.fusion.FusionNet) on out-of-sample member outputs.

Every training set is a directory with noisy/, clean/, meta.json and one sub-directory of member outputs per member
(the members must not have trained on that set's speech: held-out-protocol members for the held-out pairs and the
simulated sets, see configs/fusion.yaml). Each step draws one batch of 8 s crops from one set (in-domain sets with
probability --p-in), passes the fused estimate through the shipped post-processing (masks held fixed for the gradient)
and minimises -SI-SDR + the scoring ASR's encoder distance + transcript NLL (fresh mixtures only, which carry teacher
labels) + optional MWER. --folds/--fold hold back a speaker-disjoint fold of the in-domain sets for out-of-fold scoring.

  python train/train_fusion.py --config configs/fusion.yaml --out runs/fusion
"""
import argparse
import json
import math
import os
import random
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import track2.losses as L  # noqa: E402
from track2.audio import SR, si_sdr, si_sdr_loss  # noqa: E402
from track2.fusion import FusionNet, run_fusion  # noqa: E402
from track2.post import post_weights  # noqa: E402


def ls_scale(e, y):
    return e * (np.dot(e, y) / max(np.dot(e, e), 1e-12))


def read_fit(path, n):
    x = sf.read(path, dtype="float64")[0][:n]
    return np.pad(x, (0, n - len(x)))


def load_clip(job):
    """(id, noisy, clean, LS-scaled members [M, n]) with every signal cut to the noisy length."""
    c, root, dirs = job
    y = sf.read(os.path.join(root, "noisy", f"{c}.wav"), dtype="float64")[0]
    s = sf.read(os.path.join(root, "clean", f"{c}.wav"), dtype="float64")[0]
    n = min(len(y), len(s))
    y, s = y[:n], s[:n].astype(np.float32)
    X = np.stack([ls_scale(read_fit(os.path.join(root, d, f"{c}.wav"), n), y) for d in dirs]).astype(np.float32)
    return c, y.astype(np.float32), s, X


def speakers(st, c):
    """Speakers in a clip, for speaker-disjoint folds (meta.json 'speaker' or the Vaani speaker token of each bed)."""
    if "_spk" not in st:
        meta = json.load(open(os.path.join(st["root"], "meta.json"))) if os.path.exists(os.path.join(st["root"], "meta.json")) else []
        st["_spk"] = {r["clip_id"]: {r["speaker"]} if "speaker" in r else {b.split("_")[5] for b in r["beds"]} if "beds" in r
                      else {r["clip_id"].split("_")[5]} for r in meta}
    return st["_spk"].get(c) or {c.split("_")[5]}


def hash_split(spk, frac, fold, folds):
    h = zlib.crc32(spk.encode()) / 2 ** 32
    return int(h * folds) == fold if folds else h < frac


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="yaml: sets (name, root, members, fresh, limit) and hyper-parameters")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--folds", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg["fold"] = a.fold if a.fold is not None else cfg.get("fold", 0)
    cfg["folds"] = a.folds if a.folds is not None else cfg.get("folds", 0)
    return a.out, cfg


def main():
    out, c = parse()
    torch.manual_seed(c.get("seed", 0))
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(out, exist_ok=True)
    json.dump(c, open(os.path.join(out, "config.json"), "w"), indent=1)
    log = open(os.path.join(out, "log.jsonl"), "a")

    tr, dv, grp = [], [], []
    for st in c["sets"]:
        st["root"] = root = os.path.abspath(st["root"])
        fresh = bool(st.get("fresh"))
        ids = sorted(f[:-4] for f in os.listdir(os.path.join(root, "noisy")) if f.endswith(".wav"))[:st.get("limit") or None]
        frac, fold, folds = (c.get("dev_frac", 0.06), 0, 0) if fresh else (0.0, c["fold"], c["folds"])
        hold = {i for i in ids if any(hash_split(k, frac, fold, folds) for k in speakers(st, i))}
        with ProcessPoolExecutor(32) as ex:
            clips = list(ex.map(load_clip, [(i, root, st["members"]) for i in ids], chunksize=8))
        keep = [x for x in clips if x[0] not in hold and len(x[1]) >= c.get("min_s", 2.0) * SR]
        tr += keep
        grp += [root if fresh else None] * len(keep)
        if fresh:
            dv += [x for x in clips if x[0] in hold]
        print(f"{st['name']}: train {len(keep)} held back {len(hold)}", flush=True)
    fresh_i = [i for i, f in enumerate(grp) if f]
    indom_i = [i for i, f in enumerate(grp) if not f]
    dv = dv[:c.get("dev_n", 200)]

    w0 = tuple(c["w0"])
    net = FusionNet(ch=c.get("ch", 64), w0=w0, gmax=c.get("gmax", 0.0), rmax=c.get("rmax", 0.2), anchor=c.get("anchor", True)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=c.get("lr", 5e-4), weight_decay=c.get("wd", 1e-4))
    L.ENC_W, L.NLL_W, L.MWER_W = c.get("enc_loss", 30.0), c.get("nll_loss", 2.5), c.get("mwer", 0.0)
    beds_lab = None
    if L.NLL_W > 0:     # SraVaani greedy tokens of each fresh clean clip (make_teacher.py --set <fresh root>)
        lab = torch.load(c["teacher"])
        beds_lab = {0: [(*lab[os.path.join(r, "clean", f"{cid}.wav")], len(y)) if r else None for (cid, y, _, _), r in zip(tr, grp)]}
    rng = random.Random(c.get("seed", 0))
    seg, bs, steps, warmup, lr0 = int(c.get("seg", 8.0) * SR), c.get("bs", 4), c.get("steps", 3000), c.get("warmup", 200), c.get("lr", 5e-4)

    def batch():
        pool = indom_i if indom_i and (not fresh_i or rng.random() < c.get("p_in", 0.4)) else fresh_i
        idx = [pool[rng.randrange(len(pool))] for _ in range(bs)]
        n = min(seg, min(len(tr[i][1]) for i in idx))
        ys, ss, xs, src = [], [], [], []
        for i in idx:
            _, y, s, X = tr[i]
            j = rng.randrange(0, len(y) - n + 1)
            ys.append(y[j:j + n]); ss.append(s[j:j + n]); xs.append(X[:, j:j + n]); src.append((0, i, j))
        t = lambda v: torch.from_numpy(np.stack(v)).to(dev)
        lab = L.crop_labels(torch.tensor(src), beds_lab, n) if beds_lab is not None and pool is fresh_i else None
        return t(xs), t(ys), t(ss), lab

    def through_post(y, e):
        """The shipped post-processing with masks from the current output, held fixed for the gradient."""
        yn, en = y.detach().double().cpu().numpy(), e.detach().double().cpu().numpy()
        W, G = zip(*[post_weights(yn[i], en[i]) for i in range(len(yn))])
        W, G = torch.from_numpy(np.stack(W)).to(e.device), torch.from_numpy(np.stack(G)).to(e.device)
        return (W * e + (1 - W) * y) * G

    def evaluate():
        net.eval()
        d = []
        for _, y, s, X in dv:
            e = run_fusion(net, dev, y, X)
            b = (np.array(w0)[:, None] * X).sum(0)
            d.append(si_sdr(s, e) - si_sdr(s, b))
        net.train()
        return {"dev_dsi": float(np.mean(d))}

    ls, t0 = [], time.time()
    net.train()
    for step in range(1, steps + 1):
        lr = lr0 * min(1.0, step / warmup) * (0.5 * (1 + math.cos(math.pi * min(1.0, step / steps))) * 0.98 + 0.02)
        for g in opt.param_groups:
            g["lr"] = lr
        xt, yt, st_, lab = batch()
        o = through_post(yt, net.wave(xt, yt))
        loss = si_sdr_loss(o, st_) + L.enc_term(o, st_, lab)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        ls.append(loss.item())
        if step % c.get("eval_every", 500) == 0 or step == steps:
            r = {"step": step, "loss": float(np.mean(ls)), "lr": lr, **(evaluate() if dv else {}), "sec": time.time() - t0}
            log.write(json.dumps(r) + "\n")
            log.flush()
            print(r, flush=True)
            ls = []
            torch.save({"cfg": net.cfg, "state": net.state_dict(), "step": step}, os.path.join(out, f"step_{step:06d}.pt"))


if __name__ == "__main__":
    main()
