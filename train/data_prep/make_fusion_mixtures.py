"""Fusion training mixtures: test-shaped single-speaker clips (8-18 s) on clean beds that no held-out-protocol member
trained on (Vaani transcription release tagless beds and IndicVoices-R, low pause floor), with validation-pool noise
events pasted as the organisers' generator does. The fresh2 set used for the final fusion:

  python train/data_prep/make_fusion_mixtures.py --out data/fresh2 --n 5000 --seed 1 --rate 0.65 --dur-median 13 \
      --min-s 8 --max-s 18 --ivr-scan 60000
"""
import argparse
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf

SR = 16000
DATA = "data"


def _floor_db(path):
    """10th percentile of the 20 ms energy envelope, dB below its peak: how quiet the bed's pauses are."""
    from scipy.ndimage import uniform_filter1d
    x = sf.read(path, dtype="float64")[0]
    e = uniform_filter1d(x ** 2, 320)
    return float(np.percentile(10 * np.log10(e / max(e.max(), 1e-20) + 1e-15), 10))


def mix(a):
    """Phase-2-shaped mixtures on clean beds unseen by every held-out-protocol member. Organiser beds are near-silent
    in pauses (median floor -63 dB), so only low-floor beds are kept; levels follow the held-out references."""
    rng = random.Random(a.seed)
    root = os.path.join(DATA, "beds_large")
    # held-out validation speakers stay out, so the held-out pairs remain speaker-disjoint from fusion training
    held = {f.split("_")[5] for f in os.listdir(os.path.join(DATA, "heldout", "noisy"))}
    rows = [r for r in map(json.loads, open(os.path.join(root, "manifest.jsonl"))) if r["speaker_id"] not in held]
    rng.shuffle(rows)
    pick = {"vaani_tp_tagless": [], "indicvoices_r": []}
    lim = {"vaani_tp_tagless": a.floor_vaani, "indicvoices_r": a.floor_ivr}
    cand = [r for r in rows if r["source"] == "vaani_tp_tagless"] + [r for r in rows if r["source"] == "indicvoices_r"][:a.ivr_scan]
    with ProcessPoolExecutor(32) as ex:
        fl = list(ex.map(_floor_db, [os.path.join(root, r["path"]) for r in cand], chunksize=32))
    for r, f in zip(cand, fl):
        if f < lim[r["source"]]:
            pick[r["source"]].append(r)
    print({k: (len(v), round(sum(r["dur"] for r in v) / 3600, 1)) for k, v in pick.items()}, flush=True)
    by_spk = {}
    for src, v in pick.items():
        for r in v:
            by_spk.setdefault((src, r["speaker_id"]), []).append(r)
    # single recordings like the test clips: only speakers with enough speech for a clip of at least min_s
    spk = {src: [k for k in by_spk if k[0] == src and sum(r["dur"] for r in by_spk[k]) >= a.min_s] for src in pick}
    print({k: len(v) for k, v in spk.items()}, "speakers with >= min_s of beds", flush=True)
    ho = os.path.join(DATA, "heldout", "clean")
    rms_pool = [10 * np.log10(np.mean(sf.read(os.path.join(ho, f), dtype="float64")[0] ** 2) + 1e-12)
                for f in sorted(os.listdir(ho)) if f.endswith(".wav")]
    bank = os.path.join(DATA, "noisebank")
    noise = [sf.read(os.path.join(bank, "noise", r["file"]), dtype="float32")[0] for r in json.load(open(os.path.join(bank, "noisebank.json")))]
    for d in ("noisy", "clean"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    out = []
    for i in range(a.n):
        src = "vaani_tp_tagless" if rng.random() < a.p_vaani else "indicvoices_r"
        key = rng.choice(spk[src])
        target = min(a.max_s, max(a.min_s, rng.lognormvariate(math.log(a.dur_median), 0.35)))
        beds = by_spk[key][:]
        rng.shuffle(beds)
        parts, used, dur = [], [], 0.0
        while beds and dur < target:
            r = beds.pop()
            x = sf.read(os.path.join(root, r["path"]), dtype="float32")[0]
            if dur + len(x) / SR > a.max_s:
                x = x[:int((a.max_s - dur) * SR)]
            parts += [x, np.zeros(SR // 20, np.float32)]
            used.append(r["path"])
            dur += len(x) / SR + 0.05
        bed = np.concatenate(parts[:-1]).astype(np.float64)
        if len(bed) < a.min_s * SR * 0.5:
            continue
        lvl = rng.choice(rms_pool) + a.bed_db
        bed *= 10 ** (lvl / 20) / max(np.sqrt(np.mean(bed ** 2)), 1e-9)
        n, mx, spans = len(bed), bed.copy(), []
        k = max(1, int(np.random.default_rng(a.seed * 100003 + i).poisson(a.rate * n / SR)))
        for _ in range(k):
            ev = rng.choice(noise)[:n] * float(a.gain_median * np.exp(rng.gauss(0, a.gain_sigma)))
            p = rng.randrange(0, n - len(ev) + 1)
            mx[p:p + len(ev)] += ev
            spans.append((p / SR, (p + len(ev)) / SR))
        pk = float(np.abs(mx).max())
        sc = 0.99 / pk if pk > 0.99 else 1.0
        m = np.zeros(n, bool)
        for s0, s1 in spans:
            m[int(s0 * SR):int(s1 * SR)] = True
        cid = f"fz_{i:05d}"
        sf.write(os.path.join(a.out, "noisy", f"{cid}.wav"), mx * sc, SR, subtype="PCM_16")
        sf.write(os.path.join(a.out, "clean", f"{cid}.wav"), bed * sc, SR, subtype="PCM_16")
        out.append({"clip_id": cid, "source": src, "speaker": key[1], "language": by_spk[key][0]["language"], "beds": used,
                    "duration": n / SR, "coverage": float(m.mean()), "spans": spans})
    json.dump(out, open(os.path.join(a.out, "meta.json"), "w"), ensure_ascii=False)
    with open(os.path.join(a.out, "clean_manifest.jsonl"), "w") as f:   # for make_teacher.py --set
        f.writelines(json.dumps({"path": f"clean/{r['clip_id']}.wav"}) + "\n" for r in out)
    dur = np.array([r["duration"] for r in out])
    cov = np.array([r["coverage"] for r in out])
    print(f"{len(out)} clips | {dur.sum() / 3600:.2f} h | duration p10/p50/p90 {np.percentile(dur, [10, 50, 90]).round(1)} | coverage bands "
          f"{np.round(np.histogram(cov, [0, .2, .4, .6, .8, 1.01])[0] / len(cov), 3)} | speakers {len({r['speaker'] for r in out})}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p = ap
    p.add_argument("--out", default=os.path.join(DATA, "fresh2"))
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p-vaani", type=float, default=0.35, help="share of clips on Vaani tagless beds (rest IndicVoices-R)")
    p.add_argument("--floor-vaani", type=float, default=-42.0)
    p.add_argument("--floor-ivr", type=float, default=-50.0)
    p.add_argument("--ivr-scan", type=int, default=20000, help="IndicVoices-R beds screened for a quiet floor")
    p.add_argument("--dur-median", type=float, default=8.0)
    p.add_argument("--min-s", type=float, default=3.0)
    p.add_argument("--max-s", type=float, default=16.0)
    p.add_argument("--bed-db", type=float, default=-3.5, help="bed level against the held-out references, as make_valset.py")
    p.add_argument("--rate", type=float, default=0.5)
    p.add_argument("--gain-median", type=float, default=0.93)
    p.add_argument("--gain-sigma", type=float, default=0.80)
    mix(ap.parse_args())


if __name__ == "__main__":
    main()
