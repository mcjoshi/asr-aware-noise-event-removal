"""Training data for the members.

- Stream: crops of the organisers' synthetic validation pairs (p_real) or of their clean references re-mixed with 1-3
  events from the validation noise bank at an SNR drawn from the real pairs' SNRs.
- LongStream: 12-18 s windows of single-speaker long recordings (Vaani transcription release) with validation-pool events
  pasted as the organisers' generator does (Poisson count, log-normal gain); used for the ASR losses only.
Every sample draws from an RNG seeded by (seed, sample index), so DataLoader workers never replay each other.
"""
import json
import os
import random

import numpy as np
import torch

SR = 16000
BINS = [0, .2, .4, .6, .8, 1.01]                      # event-coverage bands
W_P2 = np.array([.231, .446, .250, .069, .004])     # Phase 2 test share per band (for reweighted held-out SI-SDR)


def load_meta(val_root=None):
    """Organiser validation metadata (validation/validationMetadata.json) with clip ids and audio paths."""
    val_root = val_root or os.path.join("data", "validation")
    recs = json.load(open(os.path.join(val_root, "validationMetadata.json")))
    for r in recs:
        r["clip_id"] = os.path.splitext(r["segmentFileName"])[0]
        sub = "syntheticNoiseAudio" if r["syntheticData"] else "naturalNoisyAudio"
        r["noisy_path"] = os.path.join(val_root, sub, r["segmentFileName"])
        r["clean_path"] = os.path.join(val_root, "syntheticCleanRefAudio", r["segmentFileName"]) if r["syntheticData"] else None
    return recs


def coverage(r):
    dur = float(r.get("duration") or 0) or 1e-9
    return min(1.0, sum(max(0.0, float(e["end"]) - float(e["start"])) for e in (r.get("NoiseSubCategoryTimeStamp") or [])) / dur)


class Stream(torch.utils.data.Dataset):
    def __init__(self, real, beds, noise, snr_pool, n, seg, seed, p_real=0.5, extra_beds=None, p_extra=0.0, with_src=False):
        self.real, self.beds, self.noise, self.snr = real, beds, noise, snr_pool
        self.extra, self.p_extra = extra_beds or [], p_extra
        self.n, self.seg, self.seed, self.p_real = n, seg, seed, p_real
        self.with_src = with_src    # also return (pool, index, crop start): pool 0 real pairs, 1 beds, 2 extra beds

    def __len__(self):
        return self.n

    def _crop(self, rng, *xs):
        k, seg = len(xs[0]), self.seg
        if k >= seg:
            j = rng.randrange(0, k - seg + 1)
            return j, [x[j:j + seg] for x in xs]
        return 0, [np.pad(x, (0, seg - k)) for x in xs]

    def __getitem__(self, i):
        rng = random.Random(self.seed * 1_000_003 + i)
        if rng.random() < self.p_real:
            src, bi = 0, rng.randrange(len(self.real))    # randrange(n) draws exactly as choice() does
            off, (mx, cl) = self._crop(rng, *self.real[bi])
        else:
            src = 2 if (self.extra and rng.random() < self.p_extra) else 1
            pool = self.extra if src == 2 else self.beds
            bi = rng.randrange(len(pool))
            off, (cl,) = self._crop(rng, pool[bi])
            noise = np.zeros(self.seg, np.float32)
            for _ in range(rng.randint(1, 3)):
                ns = rng.choice(self.noise)
                if len(ns) > self.seg:
                    k = rng.randrange(0, len(ns) - self.seg + 1)
                    ns = ns[k:k + self.seg]
                p = rng.randrange(0, self.seg - len(ns) + 1)
                noise[p:p + len(ns)] += ns
            ec, en = float((cl ** 2).sum()), float((noise ** 2).sum())
            if en > 1e-10 and ec > 1e-10:
                noise = noise * float(np.sqrt(ec / (en * 10 ** (rng.choice(self.snr) / 10))))
            mx = cl + noise
        pk = float(np.abs(mx).max())
        if pk > 1.0:
            mx, cl = mx / pk, cl / pk
        out = torch.from_numpy(np.ascontiguousarray(mx, np.float32)), torch.from_numpy(np.ascontiguousarray(cl, np.float32))
        return out + (torch.tensor([src, bi, off]),) if self.with_src else out


class LongStream(torch.utils.data.Dataset):
    """Micro-batches of whole-recording windows (length drawn in [lo, hi] s, cut to the shortest bed) with validation-pool
    events pasted as train/data_prep/make_valset.py does: Poisson event count at RATE per second, absolute gain log-normal around
    GAIN_MED, bed level a training reference's RMS + BED_DB. Item i is one micro-batch: mx, cl [B, L], src [B, 3]."""

    RATE, GAIN_MED, GAIN_SIG, BED_DB = 0.5, 0.93, 0.80, -3.5
    POOL = 3    # src pool id of long beds in crop_labels

    def __init__(self, beds, noise, rms_pool, n, bs, lo, hi, seed, gain_db=0.0):
        self.beds, self.noise, self.rms, self.n, self.bs = beds, noise, rms_pool, n, bs
        self.GAIN_MED = self.GAIN_MED * 10 ** (gain_db / 20)
        self.lo, self.hi, self.seed = int(lo * SR), int(hi * SR), seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = random.Random((self.seed + 7919) * 1_000_003 + i)
        idx = [rng.randrange(len(self.beds)) for _ in range(self.bs)]
        L = min(rng.randrange(self.lo, self.hi + 1), *(len(self.beds[k]) for k in idx))
        mxs, cls, src = [], [], []
        for k in idx:
            off = rng.randrange(0, len(self.beds[k]) - L + 1)
            cl = self.beds[k][off:off + L].astype(np.float64)
            cl *= 10 ** ((rng.choice(self.rms) + self.BED_DB) / 20) / max(np.sqrt(np.mean(cl ** 2)), 1e-9)
            mx = cl.copy()
            for _ in range(max(1, int(np.random.default_rng(rng.getrandbits(32)).poisson(self.RATE * L / SR)))):
                ev = rng.choice(self.noise)[:L] * float(self.GAIN_MED * np.exp(rng.gauss(0, self.GAIN_SIG)))
                p = rng.randrange(0, L - len(ev) + 1)
                mx[p:p + len(ev)] += ev
            s = min(1.0, 0.99 / max(float(np.abs(mx).max()), 1e-9))
            mxs.append(mx * s); cls.append(cl * s); src.append((self.POOL, k, off))
        f = lambda v: torch.from_numpy(np.stack(v).astype(np.float32))
        return f(mxs), f(cls), torch.tensor(src)
