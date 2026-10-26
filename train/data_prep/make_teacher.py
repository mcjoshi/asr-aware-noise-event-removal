"""Teacher labels for the TDT sequence-NLL loss: SraVaani greedy TDT tokens and their encoder frames on every
clean bed the trainer can sample, decoded once at full context exactly as the scoring path runs the ASR
(fp32, one clip per encoder pass). Greedy decoding is batched over clips; --verify checks it
against the model's own one-clip decoder.

  python train/data_prep/make_teacher.py --set val
  python train/data_prep/make_teacher.py --set $TRACK2_DATA/beds_large/manifest.jsonl --drop-source vaani_tp_tagless [--shard 0/4]

Output: $TRACK2_DATA/tdt_teacher/<name>.pt, {key: (tokens int16, frames int32)} as tensors; keys are "val/<clip_id>" for the
validation clean references (truncated to the noisy length, as the trainer holds them) and absolute wav paths
for manifest beds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = os.environ.get("TRACK2_DATA", "data")   # data root, see README "Data layout"
TEACHER = os.path.join(DATA, "tdt_teacher")


def load_asr(dev):
    from transformers import AutoModel
    from track2.losses import asr_path
    ASR = asr_path()
    m = AutoModel.from_pretrained(ASR, trust_remote_code=True).to(dev).eval()
    m._ensure_loaded()
    return m


@torch.no_grad()
def encode(m, wavs):
    """Encoder states of each clip on its own (padding a batch shifts SraVaani's states by up to ~4 at the
    clip ends, which flips tokens), zero-padded into one batch for the greedy decoder."""
    encs = []
    for w in wavs:
        f, sl = m.extract_features(torch.from_numpy(w)[None], torch.tensor([len(w)]))
        e, el = m.encoder(f.to(m._io_dtype), sl.to(f.device))
        encs.append(e[0, :, :int(el[0])])
    enc = torch.zeros(len(encs), encs[0].shape[0], max(e.shape[1] for e in encs), device=encs[0].device, dtype=encs[0].dtype)
    for i, e in enumerate(encs):
        enc[i, :, :e.shape[1]] = e
    return enc, torch.tensor([e.shape[1] for e in encs], device=enc.device)


@torch.no_grad()
def greedy_batch(m, enc, enc_len):
    """SraVaaniTDTModel._greedy_one over a batch: same symbol cap and the same extra frame skip after it."""
    cfg, dev = m.config, enc.device
    nd, blank, cap = cfg.num_durations, cfg.blank_id, cfg.max_symbols
    durs = torch.tensor(cfg.durations, device=dev)
    B = enc.shape[0]
    h = torch.zeros(cfg.pred_rnn_layers, B, cfg.pred_hidden, device=dev, dtype=m._io_dtype)
    c = torch.zeros_like(h)
    last = torch.full((B,), blank, dtype=torch.long, device=dev)
    t = torch.zeros(B, dtype=torch.long, device=dev)
    added = torch.zeros(B, dtype=torch.long, device=dev)
    out_b, out_k, out_t = [], [], []
    while True:
        idx = (t < enc_len).nonzero()[:, 0]
        if not len(idx):
            break
        f = enc[idx, :, t[idx]][..., None]
        lg, _, h2, c2 = m.decoder_joint(f, last[idx, None].int(), torch.ones(len(idx), dtype=torch.int32, device=dev),
                                        h[:, idx], c[:, idx])
        lg = lg[:, 0, 0]
        k, skip = lg[:, :-nd].argmax(-1), durs[lg[:, -nd:].argmax(-1)]
        em = k != blank
        out_b.append(idx[em]); out_k.append(k[em]); out_t.append(t[idx[em]])
        h[:, idx[em]], c[:, idx[em]], last[idx[em]] = h2[:, em], c2[:, em], k[em]
        added[idx] += 1
        t[idx] += skip
        capped = added[idx] == cap
        t[idx[capped]] += 1
        added[idx[(skip > 0) | capped]] = 0
    b, k, fr = (torch.cat(v).cpu() for v in (out_b, out_k, out_t))
    return [(k[b == i].short(), fr[b == i].int()) for i in range(B)]


def load_set(a):
    if a.set == "val":
        from data import load_meta
        keys, wavs = [], []
        for r in load_meta():
            if r["syntheticData"]:
                cl = sf.read(r["clean_path"], dtype="float32")[0]
                n = min(len(cl), sf.info(r["noisy_path"]).frames)
                keys.append(f"val/{r['clip_id']}"); wavs.append(np.ascontiguousarray(cl[:n]))
        return "val695", keys, wavs
    root, drop = os.path.dirname(os.path.abspath(a.set)), set(filter(None, a.drop_source.split(",")))
    keys = [os.path.join(root, r["path"]) for r in map(json.loads, open(a.set)) if r.get("source") not in drop]
    i, n = map(int, a.shard.split("/"))
    keys = keys[i::n]
    with ThreadPoolExecutor(32) as ex:
        wavs = list(ex.map(lambda f: sf.read(f, dtype="float32")[0], keys))
    return os.path.basename(root) + (f".{i}of{n}" if n > 1 else ""), keys, wavs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", required=True, help="'val' or a bed manifest.jsonl (paths relative to it)")
    ap.add_argument("--drop-source", default="")
    ap.add_argument("--shard", default="0/1", help="i/n: decode every n-th manifest bed from the i-th (parallel processes)")
    ap.add_argument("--name", default="", help="output name (default: val695 or the manifest's directory)")
    ap.add_argument("--batch-sec", type=float, default=600.0, help="audio seconds per encoder batch")
    ap.add_argument("--verify", type=int, default=0, help="also decode this many clips one by one and compare")
    a = ap.parse_args()
    dev = "cuda"
    m = load_asr(dev)
    name, keys, wavs = load_set(a)
    name = a.name or name
    print(f"{name}: {len(wavs)} beds, {sum(map(len, wavs)) / 16000 / 3600:.1f} h", flush=True)
    order = np.argsort([len(w) for w in wavs])
    lab, i, nb, t0 = {}, 0, 0, time.time()
    while i < len(order):
        j = i + 1   # ascending lengths: the padded batch is the last clip's length times the count
        while j < len(order) and len(wavs[order[j]]) * (j - i + 1) <= a.batch_sec * 16000:
            j += 1
        idx = order[i:j]
        enc, enc_len = encode(m, [wavs[k] for k in idx])
        for k, r in zip(idx, greedy_batch(m, enc, enc_len)):
            lab[keys[k]] = r
        i, nb = j, nb + 1
        if nb % 20 == 0:
            print(f"  {len(lab)}/{len(order)} beds, {time.time() - t0:.0f}s", flush=True)
    os.makedirs(TEACHER, exist_ok=True)
    torch.save(lab, os.path.join(TEACHER, f"{name}.pt"))
    nt = sum(len(v[0]) for v in lab.values())
    print(f"saved {TEACHER}/{name}.pt: {len(lab)} beds, {nt} tokens, {time.time() - t0:.0f}s", flush=True)
    if a.verify:
        rng = np.random.default_rng(0)
        same = tot_t = diff_t = 0
        for k in rng.choice(len(keys), min(a.verify, len(keys)), replace=False):
            enc, enc_len = encode(m, [wavs[k]])
            tk, fr = m._greedy_one(enc, int(enc_len[0]))
            ok = list(tk) == lab[keys[k]][0].tolist() and list(fr) == lab[keys[k]][1].tolist()
            same += ok; tot_t += len(tk)
            diff_t += 0 if ok else max(len(tk), len(lab[keys[k]][0]))
        print(f"verify: {same}/{min(a.verify, len(keys))} beds identical to the one-clip decoder "
              f"({diff_t} of {tot_t} tokens in differing beds)", flush=True)


if __name__ == "__main__":
    main()
