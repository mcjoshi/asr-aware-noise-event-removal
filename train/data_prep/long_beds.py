"""Single-recording long clean beds and a single-recording long validation set from Vaani-transcription-part.

Test clips are one speaker for 9-18 s; organiser pairs are 4.5 s and every long clip we used before was glued.
Each Vaani-transcription-part segment is one recording, so a 10-18 s segment is a test-shaped bed.
Kept only if the transcript carries benign tags alone, the pauses are quiet (`_floor_db`, as the fusion mixer),
there is no clipping or long digital silence, and the recording is outside exclude.json (re-checked here).
Speakers are split by a fixed hash into an evaluation pool and a training pool before anything is selected;
evaluation speakers also avoid every organiser validation speaker and every bed any training run has used.

  beds   : python3 long_beds.py beds --out $TRACK2_DATA/beds_long
  valset : python3 long_beds.py valset --out $TRACK2_DATA/valset_long --n 400
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = os.environ.get("TRACK2_DATA", "data")   # data root, see README "Data layout"
SR = 16000
T2 = DATA
TP = os.path.join(DATA, "vaani_tp")
BENIGN = {"breathing", "inhaling", "pause", "lip smack", "lip smacks", "lip smacking"}
TAG = re.compile(r"\[([^\]]+)\]|<([^>]+)>")
LEVEL_DB = -25.7    # organiser references' median RMS, as beds_large


def speaker(clip_id):
    return clip_id.split("_")[5]


def eval_speaker(spk, frac):
    return int(hashlib.md5(spk.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


def used_speakers():
    """Speakers of every organiser validation clip and every bed a training run has drawn from."""
    from data import load_meta
    out = {speaker(r["clip_id"]) for r in load_meta()}
    out |= {r["speaker_id"] for r in map(json.loads, open(os.path.join(T2, "beds_large", "manifest.jsonl")))
            if r["source"] != "indicvoices_r"}
    gold = os.path.join(T2, "beds_gold")   # earlier gold beds, if any
    out |= {speaker(f) for f in os.listdir(gold) if f.endswith(".wav")} if os.path.isdir(gold) else set()
    return out


def heldout_speakers():
    return {speaker(f[:-4]) for f in os.listdir(os.path.join(T2, "heldout", "noisy")) if f.endswith(".wav")}


def candidates(lo, hi, min_words):
    from track2 import text as mt
    ex = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exclude.json")))
    clips, blocked = set(ex["clips"]), set(ex["blocked_recordings"])
    rows, why = [], Counter()
    for r in map(json.loads, open(os.path.join(TP, "manifest.jsonl"))):
        if not lo <= r["duration"] <= hi:
            continue
        tags = {(a or b).strip().lower().lstrip("/") for a, b in TAG.findall(r["transcript"])}
        if r["clip_id"] in clips or r["clip_id"].rsplit("_", 2)[0] in blocked:
            why["excluded"] += 1
        elif not tags <= BENIGN:
            why["tags"] += 1
        elif len(mt.normalize(r["transcript"]).split()) < min_words:
            why["few words"] += 1
        else:
            rows.append(r)
    return rows, why


def audit(path):
    """Pause floor (dB under the 20 ms envelope peak, 10th percentile), longest exact-zero run (s), peak, RMS dB."""
    from scipy.ndimage import uniform_filter1d
    x = sf.read(path, dtype="float64")[0]
    e = uniform_filter1d(x ** 2, 320)
    floor = float(np.percentile(10 * np.log10(e / max(e.max(), 1e-20) + 1e-15), 10))
    z = np.concatenate([[0], (x == 0).astype(np.int8), [0]])
    d = np.flatnonzero(np.diff(z))
    run = float((d[1::2] - d[::2]).max() / SR) if len(d) else 0.0
    return floor, run, float(np.abs(x).max()), float(10 * np.log10(np.mean(x ** 2) + 1e-20)), int((np.abs(x) >= 0.999).sum())


def screen(rows, a):
    with ProcessPoolExecutor(48) as ex:
        res = list(ex.map(audit, [r["wav"] for r in rows], chunksize=16))
    keep, why = [], Counter()
    for r, (fl, run, pk, rms, clip) in zip(rows, res):
        if fl >= a.floor_db:
            why["floor"] += 1
        elif run > a.max_zero_s:
            why["zero run"] += 1
        elif clip > 2:
            why["clipping"] += 1
        else:
            keep.append({**r, "floor_db": fl, "rms_db": rms, "peak": pk})
    return keep, why


def beds(a):
    rows, why = candidates(a.min_s, a.max_s, a.min_words)
    held = heldout_speakers()
    rows = [r for r in rows if speaker(r["clip_id"]) not in held and not eval_speaker(speaker(r["clip_id"]), a.eval_frac)]
    print(f"candidates {len(rows)} after tags/exclusion/held-out/eval-pool | dropped {dict(why)}", flush=True)
    keep, why = screen(rows, a)
    print(f"passed audio screen {len(keep)} | dropped {dict(why)}", flush=True)
    rng = random.Random(a.seed)
    rng.shuffle(keep)
    per, out = Counter(), []
    for r in sorted(keep, key=lambda r: r["floor_db"]):   # quietest first, capped per speaker
        s = speaker(r["clip_id"])
        if per[s] < a.spk_cap and len(out) < a.max_n:
            per[s] += 1
            out.append(r)
    os.makedirs(os.path.join(a.out, "wav"), exist_ok=True)
    with open(os.path.join(a.out, "manifest.jsonl"), "w") as f:
        for r in sorted(out, key=lambda r: r["clip_id"]):
            x = sf.read(r["wav"], dtype="float64")[0]
            x *= 10 ** ((LEVEL_DB - r["rms_db"]) / 20)
            x /= max(1.0, np.abs(x).max() / 0.99)
            p = f"wav/{r['clip_id']}.wav"
            sf.write(os.path.join(a.out, p), x, SR, subtype="PCM_16")
            f.write(json.dumps({"path": p, "source": "vaani_tp_long", "language": r["language"], "speaker_id": speaker(r["clip_id"]),
                                "dur": round(len(x) / SR, 3), "floor_db": round(r["floor_db"], 1), "clip_id": r["clip_id"],
                                "transcript": r["transcript"]}, ensure_ascii=False) + "\n")
    summary(out, "beds_long")


def valset(a):
    rows, why = candidates(a.min_s, a.max_s, a.min_words)
    banned = used_speakers() | heldout_speakers()
    long_m = os.path.join(T2, "beds_long", "manifest.jsonl")
    if os.path.exists(long_m):
        banned |= {r["speaker_id"] for r in map(json.loads, open(long_m))}
    rows = [r for r in rows if eval_speaker(speaker(r["clip_id"]), a.eval_frac) and speaker(r["clip_id"]) not in banned]
    print(f"eval-pool candidates {len(rows)} ({len({speaker(r['clip_id']) for r in rows})} speakers) | dropped {dict(why)}", flush=True)
    keep, why = screen(rows, a)
    print(f"passed audio screen {len(keep)} | dropped {dict(why)}", flush=True)
    rng = random.Random(a.seed)
    rng.shuffle(keep)
    seen, pick = set(), []
    for r in keep:   # one clip per speaker
        if speaker(r["clip_id"]) not in seen and len(pick) < a.n:
            seen.add(speaker(r["clip_id"]))
            pick.append(r)
    ho = os.path.join(T2, "heldout", "clean")
    rms_pool = [10 * np.log10(np.mean(sf.read(os.path.join(ho, f), dtype="float64")[0] ** 2) + 1e-12)
                for f in sorted(os.listdir(ho)) if f.endswith(".wav")]
    bank = os.path.join(T2, "noisebank")
    noise = [sf.read(os.path.join(bank, "noise", r["file"]), dtype="float32")[0] for r in json.load(open(os.path.join(bank, "noisebank.json")))]
    for d in ("noisy", "clean"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    out = []
    for i, r in enumerate(pick):
        bed = sf.read(r["wav"], dtype="float64")[0]
        bed *= 10 ** ((rng.choice(rms_pool) + a.bed_db) / 20) / max(np.sqrt(np.mean(bed ** 2)), 1e-9)
        n, mx, spans = len(bed), bed.copy(), []
        k = max(1, int(np.random.default_rng(a.seed * 100003 + i).poisson(a.rate * n / SR)))
        for _ in range(k):
            ev = rng.choice(noise)[:n] * float(a.gain_median * np.exp(rng.gauss(0, a.gain_sigma)))
            p = rng.randrange(0, n - len(ev) + 1)
            mx[p:p + len(ev)] += ev
            spans.append((p / SR, (p + len(ev)) / SR))
        sc = min(1.0, 0.99 / float(np.abs(mx).max()))
        m = np.zeros(n, bool)
        for s0, s1 in spans:
            m[int(s0 * SR):int(s1 * SR)] = True
        cid = f"vl_{i:04d}"
        sf.write(os.path.join(a.out, "noisy", f"{cid}.wav"), mx * sc, SR, subtype="PCM_16")
        sf.write(os.path.join(a.out, "clean", f"{cid}.wav"), bed * sc, SR, subtype="PCM_16")
        out.append({"clip_id": cid, "source_clip": r["clip_id"], "speaker": speaker(r["clip_id"]), "language": r["language"],
                    "duration": n / SR, "coverage": float(m.mean()), "spans": spans, "floor_db": r["floor_db"], "transcript": r["transcript"]})
    json.dump(out, open(os.path.join(a.out, "meta.json"), "w"), ensure_ascii=False)
    summary(out, "valset_long")
    sp = [s1 - s0 for r in out for s0, s1 in r["spans"]]
    cov = np.array([r["coverage"] for r in out])
    print(f"events {len(sp)} | > 1.2 s {sum(d > 1.2 for d in sp)} | coverage p10/p50/p90 {np.percentile(cov, [10, 50, 90]).round(2)} | "
          f"bands {np.round(np.histogram(cov, [0, .2, .4, .6, .8, 1.01])[0] / len(cov), 3)}")


def summary(rows, name):
    dur = np.array([r.get("dur", r.get("duration")) for r in rows])
    fl = np.array([r["floor_db"] for r in rows])
    print(f"{name}: {len(rows)} clips | {dur.sum() / 3600:.2f} h | duration p10/p50/p90 {np.percentile(dur, [10, 50, 90]).round(1)} | "
          f"floor p50/p90 {np.percentile(fl, [50, 90]).round(1)} dB | speakers {len({r.get('speaker_id') or r.get('speaker') or speaker(r['clip_id']) for r in rows})} | "
          f"languages {len({r['language'] for r in rows})}: {Counter(r['language'] for r in rows).most_common(8)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    # training beds tolerate a louder floor: the long stream's ASR losses only ask for invariance to the pasted events
    for name, lo, fl, out in (("beds", 12.0, -36.0, "beds_long"), ("valset", 10.0, -38.0, "valset_long")):
        p = sub.add_parser(name)
        p.add_argument("--out", default=os.path.join(T2, out))
        p.add_argument("--min-s", type=float, default=lo)
        p.add_argument("--max-s", type=float, default=18.0)
        p.add_argument("--min-words", type=int, default=5)
        p.add_argument("--floor-db", type=float, default=fl, help="keep beds whose pause floor is below this")
        p.add_argument("--max-zero-s", type=float, default=0.3, help="longest run of exact zeros")
        p.add_argument("--eval-frac", type=float, default=0.15, help="share of speakers reserved for evaluation")
        p.add_argument("--seed", type=int, default=0)
    p = sub.choices["beds"]
    p.add_argument("--spk-cap", type=int, default=2)
    p.add_argument("--max-n", type=int, default=5000)
    p = sub.choices["valset"]
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--bed-db", type=float, default=-3.5, help="bed level against the held-out references, as make_valset.py")
    p.add_argument("--rate", type=float, default=0.5, help="events per second")
    p.add_argument("--gain-median", type=float, default=0.93)
    p.add_argument("--gain-sigma", type=float, default=0.80)
    a = ap.parse_args()
    {"beds": beds, "valset": valset}[a.cmd](a)


if __name__ == "__main__":
    main()
