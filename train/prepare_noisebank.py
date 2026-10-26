"""Extract the organisers' noise events from the synthetic validation pairs, as training augmentation material.

The organisers' synthetic mixing is additive (speech gain 1.0), so noise = noisy - clean is the exact interferer.
Each annotated event span (NoiseSubCategoryTimeStamp, > 50 ms) is saved as one noise segment. Validation speakers are
split once by a fixed hash (70% train, 30% held out); clips of held-out speakers form the 209-pair held-out set that
training never sees in the held-out protocol. The bank is used ONLY to synthesise training mixtures; nothing in the
inference pipeline reads it.

  python train/prepare_noisebank.py --val-root data/validation --out data/noisebank
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_meta  # noqa: E402


def speaker_of(stem, district):
    p = stem.split("_")
    try:
        return p[p.index(district) + 1]        # the speaker token follows the district; its shape varies
    except ValueError:
        return None


def in_train(spk, frac=0.70):
    return (int(hashlib.sha1(spk.encode()).hexdigest()[:8], 16) % 10000) / 10000.0 < frac


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-root", default="data/validation")
    ap.add_argument("--out", default="data/noisebank")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "noise"), exist_ok=True)
    rows, skipped = [], 0
    for r in (r for r in load_meta(a.val_root) if r["syntheticData"]):
        c = r["clip_id"]
        cl = sf.read(r["clean_path"], dtype="float32")[0]
        mx = sf.read(r["noisy_path"], dtype="float32")[0]
        n = min(len(cl), len(mx))
        cl, mx = cl[:n], mx[:n]
        noise = (mx - cl).astype(np.float32)
        spk = speaker_of(c, r["district"])
        if float((noise ** 2).sum()) <= 0 or float((cl ** 2).sum()) <= 0 or spk is None:
            skipped += 1
            continue
        split = "train" if in_train(spk) else "heldout"
        segs = []
        for ev in (r["NoiseSubCategoryTimeStamp"] or []):
            try:
                i, j = max(0, int(float(ev["start"]) * 16000)), min(n, int(float(ev["end"]) * 16000))
            except (TypeError, ValueError):
                continue
            if j - i > 800:
                segs.append((i, j, ev))
        for k, (i, j, ev) in enumerate(segs):
            seg = noise[i:j]
            if float(np.abs(seg).max()) < 1e-4:
                continue
            fn = f"{c}__{k}.wav"
            sf.write(os.path.join(a.out, "noise", fn), seg, 16000, subtype="FLOAT")
            rows.append({"file": fn, "src_clip": c, "split": split, "speaker": spk, "category": ev.get("category", "?"),
                         "tag": ev.get("tag", "?"), "dur": (j - i) / 16000, "rms": float(np.sqrt((seg ** 2).mean()))})
    json.dump(rows, open(os.path.join(a.out, "noisebank.json"), "w"))
    ho = {r["src_clip"] for r in rows if r["split"] == "heldout"}
    print(f"{len(rows)} noise segments ({skipped} clips skipped); held-out clips {len(ho)}; "
          f"{sum(r['dur'] for r in rows) / 60:.1f} min of noise")


if __name__ == "__main__":
    main()
