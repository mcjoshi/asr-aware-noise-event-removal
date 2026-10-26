"""Build a Phase-2-shaped synthetic validation set with clean references and transcripts.

Beds are the clean references of held-out validation speakers, concatenated within a language to
the Phase 2 clip length. Noise is the organisers' validation event pool, pasted at the density and
gain spread measured on the Phase 2 synthetic clips (about 0.5 events per second, gain relative
to the validation amplitude log-normal around 0.93), with the bed level set so that passthrough
SI-SDR matches the board's passthrough entry (-4.83 dB). Models under test must not have trained on
the held-out speakers' speech; the noise events are meant to be seen, as they are in the test set.

  python train/data_prep/make_valset.py --out data/valset_p2f --n 1200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import random

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = os.environ.get("TRACK2_DATA", "data")   # data root, see README "Data layout"
from data import load_meta

SR, BANK = 16000, os.path.join(DATA, "noisebank")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--rate", type=float, default=0.5, help="events per second")
    ap.add_argument("--bed-db", type=float, default=-3.5, help="bed level against the validation references")
    ap.add_argument("--dur-median", type=float, default=9.5)
    ap.add_argument("--gain-median", type=float, default=0.93)
    ap.add_argument("--gain-sigma", type=float, default=0.80, help="std of ln(gain)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--floor", type=int, default=1, help="add each bed's pre-gate residual to the noisy side")
    ap.add_argument("--floor-db", type=float, default=0.0, help="extra gain on that residual, to match the test pauses")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = json.load(open(f"{BANK}/noisebank.json"))
    held = {r["src_clip"] for r in rows if r["split"] == "heldout"}
    meta = {r["clip_id"]: r for r in load_meta() if r["syntheticData"]}
    by_lang: dict[str, list[str]] = {}
    for c in sorted(held):
        if (meta[c].get("transcript") or "").strip():
            by_lang.setdefault(meta[c]["language"], []).append(c)
    langs = [l for l, v in by_lang.items() if len(v) >= 2]
    weights = [len(by_lang[l]) for l in langs]
    clean = {c: sf.read(meta[c]["clean_path"], dtype="float32")[0] for c in held}
    # The organisers' noisy side carries the bed before its silence gate: outside the events, noisy - clean
    # is the low-level content the gate removed from the reference. Each bed brings its own.
    floor = {}
    for c in held:
        y = sf.read(meta[c]["noisy_path"], dtype="float32")[0][:len(clean[c])]
        d = np.zeros(len(clean[c]), np.float32)
        d[:len(y)] = y - clean[c][:len(y)]
        for e in meta[c]["NoiseSubCategoryTimeStamp"] or []:
            d[max(0, int(float(e["start"]) * SR) - SR // 50):int(float(e["end"]) * SR) + SR // 50] = 0.0
        # inside the original events the pre-gate residual is hidden by the noise; where the reference is
        # gated to zero there, fill it with noise at the level the visible residual has on gated samples
        z = clean[c] == 0
        seen = z & (d != 0)
        lvl = float(np.sqrt(np.mean(d[seen] ** 2))) if seen.sum() > 100 else 3e-5
        hole = z & (d == 0)
        d[hole] = np.random.default_rng(len(floor)).normal(0, lvl, int(hole.sum())).astype(np.float32)
        floor[c] = d if a.floor else np.zeros_like(d)
    noise = [sf.read(f"{BANK}/noise/{r['file']}", dtype="float32")[0] for r in rows]
    for d in ("noisy", "clean"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    out = []
    for i in range(a.n):
        lang = rng.choices(langs, weights)[0]
        target = min(18.0, max(6.0, rng.lognormvariate(np.log(a.dur_median), 0.27)))
        parts, fparts, ids, dur = [], [], [], 0.0
        pool = by_lang[lang][:]
        rng.shuffle(pool)
        while dur < target and pool:
            c = pool.pop()
            parts += [clean[c], np.zeros(SR // 20, np.float32)]
            fparts += [floor[c], np.random.default_rng(i).normal(0, 3e-5, SR // 20).astype(np.float32)]
            ids.append(c)
            dur += len(clean[c]) / SR + 0.05
        bed = np.concatenate(parts[:-1]) * 10 ** (a.bed_db / 20)
        n = len(bed)
        mix, spans = bed + np.concatenate(fparts[:-1]) * 10 ** ((a.bed_db + a.floor_db) / 20), []
        k = max(1, int(np.random.default_rng(a.seed * 100003 + i).poisson(a.rate * n / SR)))
        for _ in range(k):
            ev = rng.choice(noise)
            ev = ev[:n] * float(a.gain_median * np.exp(rng.gauss(0, a.gain_sigma)))
            p = rng.randrange(0, n - len(ev) + 1)
            mix[p:p + len(ev)] += ev
            spans.append((p / SR, (p + len(ev)) / SR))
        pk = float(np.abs(mix).max())
        scale = 0.99 / pk if pk > 0.99 else 1.0
        m = np.zeros(n, bool)
        for s0, s1 in spans:
            m[int(s0 * SR):int(s1 * SR)] = True
        cid = f"p2val_{i:05d}"
        sf.write(os.path.join(a.out, "noisy", f"{cid}.wav"), mix * scale, SR, subtype="PCM_16")
        sf.write(os.path.join(a.out, "clean", f"{cid}.wav"), bed * scale, SR, subtype="PCM_16")
        out.append({"clip_id": cid, "language": lang, "beds": ids, "duration": n / SR, "coverage": float(m.mean()),
                    "spans": spans, "transcript": " ".join(meta[c]["transcript"] for c in ids)})
    json.dump(out, open(os.path.join(a.out, "meta.json"), "w"), ensure_ascii=False)
    cov = np.array([r["coverage"] for r in out])
    dur = np.array([r["duration"] for r in out])
    print(f"{len(out)} clips | {dur.sum()/3600:.2f} h | duration p10/p50/p90 {np.percentile(dur, [10, 50, 90]).round(1)} | "
          f"coverage bands {np.round(np.histogram(cov, [0, .2, .4, .6, .8, 1.01])[0] / len(cov), 3)} | "
          f"languages {len(langs)} | held-out beds {sum(len(by_lang[l]) for l in langs)}")


if __name__ == "__main__":
    main()
