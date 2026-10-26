"""The 209 held-out validation pairs as a set directory (noisy/, clean/, meta.json): the speakers prepare_noisebank.py put
in the held-out split. Held-out-protocol members never train on them; the fusion trains on their member outputs.

  python train/data_prep/make_heldout.py --out data/heldout
"""
import argparse
import json
import os
import sys

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import load_meta  # noqa: E402

DATA = os.environ.get("TRACK2_DATA", "data")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(DATA, "heldout"))
    ap.add_argument("--noise-bank", default=os.path.join(DATA, "noisebank"))
    a = ap.parse_args()
    held = {r["src_clip"] for r in json.load(open(os.path.join(a.noise_bank, "noisebank.json"))) if r["split"] == "heldout"}
    for d in ("noisy", "clean"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    meta = []
    for r in load_meta():
        if not r["syntheticData"] or r["clip_id"] not in held:
            continue
        y = sf.read(r["noisy_path"], dtype="float32")[0]
        s = sf.read(r["clean_path"], dtype="float32")[0]
        n = min(len(y), len(s))
        sf.write(os.path.join(a.out, "noisy", f"{r['clip_id']}.wav"), y[:n], 16000, subtype="PCM_16")
        sf.write(os.path.join(a.out, "clean", f"{r['clip_id']}.wav"), s[:n], 16000, subtype="PCM_16")
        meta.append({"clip_id": r["clip_id"], "transcript": r["transcript"], "duration": n / 16000})
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), ensure_ascii=False)
    print(f"{len(meta)} held-out pairs -> {a.out}")


if __name__ == "__main__":
    main()
