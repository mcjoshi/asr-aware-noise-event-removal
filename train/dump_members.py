"""Run one trained member over a directory of wavs and write its float32 outputs (resumable): the fusion's training
inputs on the held-out pairs, the simulated sets and the fusion mixtures.

  python train/dump_members.py --model bsrnn_xl --ckpt runs/bsrnnxl_LB/ema_006000.pt --in-dir data/heldout/noisy \
      --out data/heldout/bsrnnxl_LB_tta --shifts 0,41,83,124
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from track2.members import MEMBERS, enhance_tta  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MEMBERS), required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shifts", default="0", help="comma-separated sample delays averaged as test-time augmentation")
    ap.add_argument("--limit", type=int, default=0, help="first N clips by name (the fusion's set limit)")
    a = ap.parse_args()
    shifts = tuple(int(k) for k in a.shifts.split(","))
    torch.backends.cuda.matmul.allow_tf32 = True
    mem = MEMBERS[a.model]("cuda", a.ckpt)
    mem.net.eval()
    os.makedirs(a.out, exist_ok=True)
    ids = sorted(f[:-4] for f in os.listdir(a.in_dir) if f.endswith(".wav"))[:a.limit or None]
    with torch.no_grad():
        for c in ids:
            if os.path.exists(os.path.join(a.out, f"{c}.wav")):
                continue
            y = sf.read(os.path.join(a.in_dir, f"{c}.wav"), dtype="float32")[0]
            sf.write(os.path.join(a.out, f"{c}.wav"), enhance_tta(mem, y, shifts), 16000, subtype="FLOAT")
    print(f"{a.model} {a.ckpt}: {len(ids)} clips -> {a.out}")


if __name__ == "__main__":
    main()
