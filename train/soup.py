"""Uniform weight average ("model soup") of fine-tunes that share one initialisation: one model, no extra inference cost.
The final TF-Locoformer averages the three 4k-step continuations of loco_LS (MWER, plain, louder long-stream events).

  python train/soup.py --out checkpoints/tflocoformer_soup.pt runs/loco_MWm/ema_004000.pt runs/loco_MWc/ema_004000.pt \
      runs/loco_MWch6/ema_004000.pt
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("ckpts", nargs="+")
    a = ap.parse_args()
    sds = [torch.load(p, map_location="cpu") for p in a.ckpts]
    soup = {k: (sum(sd[k].float() for sd in sds) / len(sds)).to(sds[0][k].dtype) if sds[0][k].dtype.is_floating_point else sds[0][k]
            for k in sds[0]}
    torch.save(soup, a.out)
    print(f"soup of {len(sds)} checkpoints -> {a.out}")


if __name__ == "__main__":
    main()
