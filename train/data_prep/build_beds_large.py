"""Large clean Indic speech beds from IndicVoices-R (and optionally tagless Vaani clips).

Utterances are resampled to 16 kHz mono, split at VAD pauses into 3-15 s pieces, level-normalised
to the organiser references' median RMS, and kept only if clean: no clipping, no long digital
silence, speech activity >= 50%, and SQUIM-objective PESQ at or above the references' 25th
percentile under the same scoring rule. All candidate scores are logged so thresholds can be
revisited without re-scoring.

  python3 build_beds_large.py calibrate
  python3 build_beds_large.py download --shards 30 --langs Hindi:80,Maithili:45
  CUDA_VISIBLE_DEVICES=0 python3 build_beds_large.py process --device cuda --workers 4
  python3 build_beds_large.py download --source tp --langs Bhojpuri:99,Chhattisgarhi:99
  CUDA_VISIBLE_DEVICES=0 python3 build_beds_large.py process --source tp --device cuda --workers 4
  python3 build_beds_large.py finalize --hours 15 --hours-lang Hindi:35 --spk-cap-min 6
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import soundfile as sf

DATA = os.environ.get("TRACK2_DATA", "data")   # data root, see README "Data layout"
OUT = os.path.join(DATA, "beds_large")
RAW = os.path.join(DATA, "beds_large_raw")
REFS = os.path.join(DATA, "validation", "syntheticCleanRefAudio")
VALSET = os.path.join(DATA, "valset_p2f", "clean")
EXCLUDE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exclude.json")
IVR = "ai4bharat/indicvoices_r"
TP = "ARTPARK-IISc/Vaani-transcription-part"
SR = 16000
LANGS = ["Manipuri", "Malayalam", "Bengali", "Punjabi", "Nepali", "Hindi", "Odia", "Konkani", "Telugu",
         "Maithili", "Marathi", "Assamese", "Kannada", "Tamil", "Gujarati"]
TAG = re.compile(r"(\[[^\]]+\]|<[^>]+>)")



def hf_token():
    """Hugging Face token for the gated datasets, from the environment."""
    return os.environ.get("HF_TOKEN")


# ---------- per-worker models ----------
_M = {}
_GPU_GB = 1.0


def _init(device="cpu"):
    import torch
    torch.set_num_threads(1)
    from silero_vad import load_silero_vad
    from torchaudio.pipelines import SQUIM_OBJECTIVE
    _M["vad"] = load_silero_vad()
    _M["squim"] = SQUIM_OBJECTIVE.get_model().eval().to(device)
    _M["dev"] = device
    if device != "cpu":  # stay inside the GPU 0 budget; overflow falls back to CPU
        torch.cuda.set_per_process_memory_fraction(_GPU_GB / (torch.cuda.get_device_properties(0).total_memory / 2**30))


def vad_spans(x):
    from silero_vad import get_speech_timestamps
    import torch
    return [(s["start"], s["end"]) for s in get_speech_timestamps(torch.from_numpy(x), _M["vad"], sampling_rate=SR)]


def squim(x):
    import torch
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]
    with torch.no_grad():
        try:
            stoi, pesq, sisdr = _M["squim"](t.to(_M["dev"]))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if "squim_cpu" not in _M:
                from torchaudio.pipelines import SQUIM_OBJECTIVE
                _M["squim_cpu"] = SQUIM_OBJECTIVE.get_model().eval()
            stoi, pesq, sisdr = _M["squim_cpu"](t)
    return float(pesq), float(stoi), float(sisdr)


def rms_db(x):
    return float(20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12))


def longest_run(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return int((e - s).max()) if len(s) else 0


def clip_stats(raw):
    """Samples within 1% of the peak, and the longest flat top there (the corpus is peak-normalised,
    so clipping shows as repeated identical values at the peak, not as full scale)."""
    a = np.abs(raw.astype(np.int32)).max(1)
    near = a >= 0.99 * a.max()
    flat = near[1:] & near[:-1] & (np.diff(a) == 0)
    return int(near.sum()), longest_run(flat) + 1 if flat.any() else 1


def pieces(n, spans, lo, hi):
    """Cut [0, n) into lo..hi second pieces at the middle of VAD pauses (hard cut if no pause fits)."""
    lo, hi = int(lo * SR), int(hi * SR)
    cuts = [(spans[i][1] + spans[i + 1][0]) // 2 for i in range(len(spans) - 1)]
    out, s = [], 0
    while n - s > hi:
        ok = [c for c in cuts if lo <= c - s <= hi and n - c >= lo]
        e = max(ok) if ok else s + hi
        out.append((s, e))
        s = e
    out.append((s, n))
    return [(a, b) for a, b in out if b - a >= lo]


def score_audio(x, spans, a):
    """Checks + scores for one 16 kHz mono piece already level-normalised. Returns (row, reason)."""
    vad = sum(max(0, min(e, len(x)) - max(s, 0)) for s, e in spans) / len(x)
    zrun = longest_run(np.abs(x) < 1e-4) / SR
    row = {"vad_frac": round(vad, 3), "zero_run_s": round(zrun, 3)}
    if zrun > a.max_zero_run:
        return row, "digital_silence"
    if vad < a.min_vad:
        return row, "low_speech"
    p, s, d = squim(x)
    row.update(pesq=round(p, 3), stoi=round(s, 3), sisdr=round(d, 2))
    return row, ("ok" if p >= a.pesq_min else "pesq")


def level(x, target_db):
    y = x * 10 ** ((target_db - rms_db(x)) / 20)
    pk = float(np.abs(y).max())
    return (y * 0.99 / pk if pk > 0.99 else y).astype(np.float32)


# ---------- calibrate ----------
def _score_ref(f):
    x = sf.read(f, dtype="float32")[0]
    x = x.mean(1) if x.ndim > 1 else x
    p, s, d = squim(x)
    spans = vad_spans(x)
    return {"file": os.path.basename(f), "dur": len(x) / SR, "pesq": p, "stoi": s, "sisdr": d, "rms_db": rms_db(x),
            "peak": float(np.abs(x).max()), "vad_frac": sum(e - b for b, e in spans) / len(x),
            "zero_frac": float(np.mean(x == 0)), "zero_run_s": longest_run(np.abs(x) < 1e-4) / SR}


def calibrate(a):
    os.makedirs(RAW, exist_ok=True)
    res = {}
    with Pool(a.workers, initializer=_init) as pool:
        for name, d in [("refs695", REFS), ("valset_p2f", VALSET)]:
            rows = pool.map(_score_ref, sorted(glob.glob(f"{d}/*.wav")), chunksize=4)
            q = {k: [round(float(np.percentile([r[k] for r in rows], p)), 3) for p in (5, 25, 50, 75, 95)]
                 for k in ("pesq", "stoi", "sisdr", "rms_db", "peak", "vad_frac", "zero_frac", "zero_run_s", "dur")}
            res[name] = {"n": len(rows), "pct_5_25_50_75_95": q}
            print(name, len(rows), json.dumps(q))
    res["pesq_min"] = res["refs695"]["pct_5_25_50_75_95"]["pesq"][1]
    res["target_rms_db"] = res["refs695"]["pct_5_25_50_75_95"]["rms_db"][2]
    json.dump(res, open(f"{RAW}/calib.json", "w"), indent=1)
    print("pesq_min", res["pesq_min"], "target_rms_db", res["target_rms_db"])


# ---------- download ----------
def download(a):
    from huggingface_hub import HfApi, hf_hub_download
    tok = hf_token()
    repo = IVR if a.source == "ivr" else TP
    files = HfApi(token=tok).list_repo_files(repo, repo_type="dataset")
    per = {k: int(v) for k, v in (s.split(":") for s in a.langs.split(","))} if a.langs else {}
    want = []
    for lang in (list(per) if a.only or a.source == "tp" else LANGS):
        pre = f"{lang}/" if a.source == "ivr" else f"audio/{lang}/"
        tr = sorted(f for f in files if f.startswith(pre + "train-"))
        if a.source == "tp" and len(tr) <= 2:  # thin languages: every split
            tr = sorted(f for f in files if f.startswith(pre) and f.endswith(".parquet"))
        k = min(len(tr), per.get(lang, a.shards))
        idx = sorted({int(round(i)) for i in np.linspace(0, len(tr) - 1, k)}) if k > 1 else [len(tr) // 2]
        want += [tr[i] for i in idx]  # evenly spread over the index for speaker variety
    print(f"{len(want)} shards")

    def get(f):
        rel = f.removeprefix("audio/")
        if os.path.exists(f"{RAW}/{a.source}/{rel}") or os.path.exists(f"{RAW}/scores/{a.source}/{rel}.jsonl"):
            return
        hf_hub_download(repo, f, repo_type="dataset", token=tok, local_dir=f"{RAW}/{a.source}")
        if f != rel:
            os.makedirs(os.path.dirname(f"{RAW}/{a.source}/{rel}"), exist_ok=True)
            os.replace(f"{RAW}/{a.source}/{f}", f"{RAW}/{a.source}/{rel}")
        print("got", f, flush=True)

    with ThreadPoolExecutor(a.dl_workers) as ex:
        list(ex.map(get, want))


# ---------- process ----------
def _rows_ivr(f, lang, shard):
    import pyarrow.parquet as pq
    cols = ["speaker_id", "scenario", "task_name", "gender", "snr", "c50", "duration", "audio"]
    for b in pq.ParquetFile(f).iter_batches(batch_size=16, columns=cols):
        for r in b.to_pylist():
            yield {"source": "indicvoices_r", "language": lang, "speaker_id": r["speaker_id"], "scenario": r["scenario"],
                   "task_name": r["task_name"], "gender": r["gender"], "ivr_snr": r["snr"], "ivr_c50": r["c50"]}, \
                r["audio"]["bytes"], r["duration"]


_EX = {}


def _rows_tp(f, lang, shard):
    """Vaani clips whose transcript has no bracket of any kind, minus every competition clip and every
    recording behind a competition test or validation clip (exclude.json)."""
    import pyarrow.parquet as pq
    if not _EX:
        ex = json.load(open(EXCLUDE))
        _EX.update(clips=set(ex["clips"]), blocked=set(ex["blocked_recordings"]))
    cols = ["audio", "language", "gender", "state", "district", "transcript"]
    for b in pq.ParquetFile(f).iter_batches(batch_size=16, columns=cols):
        for r in b.to_pylist():
            cid = os.path.basename(r["audio"]["path"])[:-4]
            rec = re.sub(r"_\d+_\d+$", "", cid)
            if cid in _EX["clips"] or rec in _EX["blocked"]:
                yield {"reason": "excluded"}, None, 0.0
                continue
            tr = (r["transcript"] or "").strip()
            if not tr or TAG.search(tr):
                yield {"reason": "tagged"}, None, 0.0
                continue
            toks = cid.split("_")  # speaker token follows the district, whose position varies
            dist = (r["district"] or "").replace(" ", "").lower()
            i = next((j for j in range(3, len(toks) - 1) if toks[j].lower() == dist), None)
            yield {"source": "vaani_tp_tagless", "language": "Magadhi" if lang == "Magahi" else lang,
                   "speaker_id": toks[i + 1] if i is not None else rec, "gender": r["gender"], "clip_id": cid}, \
                r["audio"]["bytes"], None


def _process_shard(job):
    f, a = job
    src, lang, shard = f.split("/")[-3], f.split("/")[-2], os.path.basename(f).replace(".parquet", "")
    sc = f"{RAW}/scores/{src}/{lang}/{shard}.parquet.jsonl"
    if os.path.exists(sc):
        return sc
    os.makedirs(os.path.dirname(sc), exist_ok=True)
    try:  # claim the shard so CPU and GPU runs can share the queue
        os.close(os.open(sc + ".claim", os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        return None
    import soxr
    out, skipped, tag = [], Counter(), "ivr" if src == "ivr" else "tp"
    for ri, (base, wav, dur) in enumerate((_rows_ivr if src == "ivr" else _rows_tp)(f, lang, shard)):
        if wav is None:
            skipped[base["reason"]] += 1
            continue
        base.update(shard=f"{src}/{lang}/{shard}", row=ri)
        if dur is not None and dur < a.min_dur:
            out.append({**base, "piece": 0, "dur": dur, "reason": "short"})
            continue
        raw, sr = sf.read(io.BytesIO(wav), dtype="int16", always_2d=True)
        if len(raw) < a.min_dur * sr:
            out.append({**base, "piece": 0, "dur": len(raw) / sr, "reason": "short"})
            continue
        n_near, plateau = clip_stats(raw)
        x = raw.astype(np.float32).mean(1) / 32768.0
        x = soxr.resample(x, sr, SR).astype(np.float32) if sr != SR else x
        spans = vad_spans(x)  # utterance-level VAD, reused for each piece
        for k, (s, e) in enumerate(pieces(len(x), spans, a.min_dur, a.max_dur)):
            y = level(x[s:e], a.target_rms_db)
            row = {**base, "piece": k, "dur": round((e - s) / SR, 3), "src_sr": sr, "clip_near_peak": n_near, "clip_plateau": plateau}
            q, reason = score_audio(y, [(b0 - s, b1 - s) for b0, b1 in spans], a)
            row.update(q, rms_db=round(rms_db(y), 2), peak=round(float(np.abs(y).max()), 3), reason=reason)
            if reason == "ok":
                split, idx = shard.split("-")[:2]
                sid = (idx if split == "train" else split + idx) if tag == "ivr" else f"{lang.lower()}_{split}{idx}"
                row["path"] = f"{base['language']}/{tag}_{sid}_{ri:04d}_{k}.wav"
                os.makedirs(f"{OUT}/{base['language']}", exist_ok=True)
                sf.write(f"{OUT}/{row['path']}", y, SR, subtype="PCM_16")
            out.append(row)
    if skipped:  # one line per shard keeps the exclusion count auditable
        out.append({"source": "vaani_tp_tagless", "language": "Magadhi" if lang == "Magahi" else lang, "shard": f"{src}/{lang}/{shard}",
                    "skipped": dict(skipped), "reason": "skipped_rows", "dur": 0.0})
    with open(sc + ".tmp", "w") as fh:
        fh.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in out)
    os.replace(sc + ".tmp", sc)
    if os.path.exists(sc + ".claim"):
        os.remove(sc + ".claim")
    if a.delete_raw:
        os.remove(f)
    return sc


def process(a):
    cal = json.load(open(f"{RAW}/calib.json"))
    a.pesq_min = a.pesq_min if a.pesq_min is not None else cal["pesq_min"]
    a.target_rms_db = cal["target_rms_db"]
    fs = sorted(glob.glob(f"{RAW}/{a.source}/*/*.parquet"))
    print(f"{len(fs)} shards | pesq_min {a.pesq_min} | target {a.target_rms_db} dB", flush=True)
    with Pool(a.workers, initializer=_init, initargs=(a.device,)) as pool:
        for i, sc in enumerate(pool.imap_unordered(_process_shard, [(f, a) for f in fs])):
            if sc is None:
                continue
            rows = [json.loads(l) for l in open(sc)]
            ok = [r for r in rows if r["reason"] == "ok"]
            print(f"[{i+1}/{len(fs)}] {sc.split('/')[-2]}/{os.path.basename(sc)[:11]} pieces {len(rows)} ok {len(ok)} "
                  f"{sum(r['dur'] for r in ok)/3600:.2f} h", flush=True)


# ---------- finalize ----------
def finalize(a):
    cal = json.load(open(f"{RAW}/calib.json"))
    pesq_min = a.pesq_min if a.pesq_min is not None else cal["pesq_min"]
    cap_h = {k: float(v) for k, v in (s.split(":") for s in a.hours_lang.split(","))} if a.hours_lang else {}
    rows = [json.loads(l) for f in sorted(glob.glob(f"{RAW}/scores/*/*/*.jsonl")) for l in open(f)]
    stats = defaultdict(Counter)
    for r in rows:  # clipping is judged here so its threshold can change without re-scoring
        if r["reason"] == "ok" and r["clip_plateau"] >= a.max_plateau:
            r["reason"] = "clipping"
    for r in rows:
        stats[(r["source"], r["language"])][r["reason"]] += 1
        stats[(r["source"], r["language"])]["sec_" + r["reason"]] += r["dur"]
        for k, v in r.get("skipped", {}).items():
            stats[(r["source"], r["language"])]["rows_" + k] += v
    rows = [r for r in rows if r["reason"] != "skipped_rows"]
    ok = [r for r in rows if r["reason"] == "ok" and r["pesq"] >= pesq_min]
    random.Random(0).shuffle(ok)
    # round-robin over speakers so the hour cap keeps speaker spread
    by = defaultdict(lambda: defaultdict(list))
    for r in ok:
        by[(r["source"], r["language"])][r["speaker_id"]].append(r)
    keep = []
    for key, spk in sorted(by.items()):
        limit, tot, used = cap_h.get(key[1], a.hours) * 3600, 0.0, Counter()
        queues = [list(v) for v in spk.values()]
        while tot < limit and any(queues):
            for qu in queues:
                if qu and tot < limit:
                    r = qu.pop()
                    if used[r["speaker_id"]] + r["dur"] <= a.spk_cap_min * 60:
                        used[r["speaker_id"]] += r["dur"]
                        tot += r["dur"]
                        keep.append(r)
    kept = {r["path"] for r in keep}
    for r in rows:  # unselected wavs move to a spare pool, so a rerun with other caps can take them back
        if r.get("path"):
            src, dst = f"{OUT}/{r['path']}", f"{RAW}/spare/{r['path']}"
            if r["path"] in kept:
                src, dst = dst, src
            if os.path.exists(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.replace(src, dst)
    missing = [r["path"] for r in keep if not os.path.exists(f"{OUT}/{r['path']}")]
    assert not missing, f"{len(missing)} selected wavs missing, e.g. {missing[:3]}"
    fields = ["path", "source", "language", "speaker_id", "dur", "pesq", "stoi", "sisdr", "vad_frac", "rms_db", "peak",
              "zero_run_s", "clip_plateau", "scenario", "task_name", "gender", "ivr_snr", "ivr_c50", "clip_id", "shard", "row", "piece"]
    keep.sort(key=lambda r: r["path"])
    with open(f"{OUT}/manifest.jsonl", "w") as fh:
        for r in keep:
            fh.write(json.dumps({k: r.get(k) for k in fields}, ensure_ascii=False) + "\n")
    summ = {}
    for key in sorted(stats):
        sel = [r for r in keep if (r["source"], r["language"]) == key]
        st = stats[key]
        n_scored = st["ok"] + st["pesq"]
        summ["/".join(key)] = {
            "candidates": sum(v for k, v in st.items() if not k.startswith(("sec_", "rows_", "skipped"))),
            "reasons": {k: v for k, v in st.items() if not k.startswith(("sec_", "skipped"))},
            "cand_hours": round(sum(v for k, v in st.items() if k.startswith("sec_")) / 3600, 2),
            "pesq_pass_rate": round(st["ok"] / max(1, n_scored), 3), "pass_hours": round(st["sec_ok"] / 3600, 2),
            "kept_clips": len(sel), "kept_hours": round(sum(r["dur"] for r in sel) / 3600, 2),
            "kept_speakers": len({r["speaker_id"] for r in sel}),
            "kept_pesq_p50": round(float(np.median([r["pesq"] for r in sel])), 2) if sel else None}
    json.dump({"pesq_min": pesq_min, "calib": cal, "per_source_language": summ}, open(f"{OUT}/stats.json", "w"), indent=1, ensure_ascii=False)
    for k, v in summ.items():
        print(f"{k:28s} cand {v['cand_hours']:6.1f} h  pesq-pass {v['pesq_pass_rate']:.2f}  kept {v['kept_clips']:6d} clips "
              f"{v['kept_hours']:5.1f} h  {v['kept_speakers']:4d} spk  {v['reasons']}")
    print(f"total {len(keep)} clips | {sum(r['dur'] for r in keep)/3600:.1f} h | {len(speakers(keep))} speakers")
    write_readme(a, keep, summ, cal, pesq_min)


def speakers(rows):
    return {(r["source"], r["language"], r["speaker_id"]) for r in rows}  # Vaani tokens are not unique across languages


def write_readme(a, keep, summ, cal, pesq_min):
    ref = cal["refs695"]["pct_5_25_50_75_95"]
    tot_h = sum(r["dur"] for r in keep) / 3600
    lines = [
        "# beds_large: clean Indic speech beds for Track 2 enhancer training", "",
        f"{len(keep)} clips, {tot_h:.1f} h, {len(speakers(keep))} speakers, 16 kHz mono PCM16, 3-15 s each.",
        "Built by `train/data_prep/build_beds_large.py` (calibrate, download, process, finalize); `manifest.jsonl` has one row per wav.",
        "", "## Counts", "",
        "| source | language | clips | hours | speakers | candidate h | SQUIM-PESQ pass | median PESQ |", "|---|---|---|---|---|---|---|---|"]
    for k, v in summ.items():
        s, lang = k.split("/")
        lines.append(f"| {s} | {lang} | {v['kept_clips']} | {v['kept_hours']:.2f} | {v['kept_speakers']} | {v['cand_hours']:.1f} | "
                     f"{v['pesq_pass_rate']:.0%} | {v['kept_pesq_p50']} |")
    for s in sorted({r["source"] for r in keep}):
        sel = [r for r in keep if r["source"] == s]
        lines.append(f"| **{s}** | all | {len(sel)} | {sum(r['dur'] for r in sel)/3600:.2f} | {len(speakers(sel))} | | | |")
    lines += [
        "", "Candidate hours count every 3-15 s piece considered; the pass rate is SQUIM-PESQ passes over pieces that reached SQUIM.",
        "Per-shard scores for every candidate (pass and fail) are in `beds_large_raw/scores/`; `stats.json` has the rejection reasons.",
        "", "## Processing and filters", "",
        "1. Decode, average channels, resample to 16 kHz (soxr). IndicVoices-R ships 48 kHz stereo, peak-normalised to -0.1 dBFS.",
        f"2. Split utterances longer than {a.max_dur:g} s at the middle of Silero-VAD pauses; drop pieces under {a.min_dur:g} s.",
        f"3. Level: each piece scaled to {cal['target_rms_db']:.1f} dB RMS (median of the 695 organiser clean references; their p5-p95 is "
        f"{ref['rms_db'][0]:.1f} to {ref['rms_db'][4]:.1f} dB), peak capped at 0.99.",
        f"4. Digital silence: drop if the longest run of |x| < 1e-4 exceeds {a.max_zero_run:g} s.",
        f"5. Speech activity: drop if the Silero-VAD speech fraction is under {a.min_vad:g}.",
        f"6. Clipping: drop if the source has a flat top of {a.max_plateau} or more identical samples at its peak (none found in IndicVoices-R).",
        f"7. Quality: keep if torchaudio SQUIM-objective predicted PESQ >= {pesq_min:.3f}, the 25th percentile of the 695 organiser clean "
        f"references scored with the same estimator on full-length input (SQUIM is level-invariant here: the refs give the "
        f"same p25 at native level and at {cal['target_rms_db']:.1f} dB; refs PESQ p25/p50/p75 "
        f"{ref['pesq'][1]:.2f}/{ref['pesq'][2]:.2f}/{ref['pesq'][3]:.2f}; valset_p2f clean "
        f"{cal['valset_p2f']['pct_5_25_50_75_95']['pesq'][1]:.2f}/{cal['valset_p2f']['pct_5_25_50_75_95']['pesq'][2]:.2f}/"
        f"{cal['valset_p2f']['pct_5_25_50_75_95']['pesq'][3]:.2f}). SQUIM STOI and SI-SDR are recorded but not filtered on.",
        f"8. Selection: per source and language, round-robin over speakers up to the hour cap ({a.hours:g} h default"
        f"{', ' + a.hours_lang if a.hours_lang else ''}), at most {a.spk_cap_min:g} min per speaker; shards are spread evenly over each "
        "language's index for speaker variety. Unselected passing wavs are kept in `beds_large_raw/spare/`.",
        "", "Not done: the organiser references are hard-gated (about 10% exact zeros, residual floor near zero); these beds are not gated.",
        "", "## Caveats", "",
        "- The per-speaker cap binds where IndicVoices-R has few speakers: Hindi (about 360 speakers in the shards pulled), Konkani, "
        "Manipuri and Gujarati (the whole Gujarati release is 7 shards, about 30 speakers). `spare/` holds the rest.",
        "- The digital-silence rule is stricter than the references themselves (their longest near-zero run has p75 0.56 s), because "
        "IndicVoices-R pauses are often gated to zero by its own restoration chain.",
        "- `exclude.json` holds competition clips by name. Phase 2 test clips are anonymised (`indoMLDatathon_*`), so whether recordings "
        "behind them are covered cannot be checked by name; drop `source == vaani_tp_tagless` if that matters. IndicVoices-R is a "
        "different corpus and does not share recordings with Vaani.",
        "- No Vaani bed is a competition clip (checked against validation metadata, the Track 1 training release and the Phase 1 "
        "test names). 43 of the Vaani recordings used also have other clips in the Track 1 training release, which is training data.",
        "- Vaani `Magahi` and `Magadhi` folders are merged as Magadhi. Manipuri and Santali have almost no tagless Vaani clips.",
        "", "## Rebuild", "",
        "```",
        "python3 build_beds_large.py calibrate",
        "python3 build_beds_large.py download --shards 30 --langs Hindi:80,Maithili:45,Nepali:42,Tamil:40,Marathi:38,Konkani:36",
        "python3 build_beds_large.py process --device cuda --workers 4   # CUDA_VISIBLE_DEVICES=0; CPU runs can share the queue",
        "python3 build_beds_large.py download --source tp --langs Chhattisgarhi:99,Magadhi:99,Magahi:99,Bhojpuri:99,Hindi:3,...",
        "python3 build_beds_large.py process --source tp --device cuda --workers 4",
        f"python3 build_beds_large.py finalize --hours {a.hours:g}{' --hours-lang ' + a.hours_lang if a.hours_lang else ''} "
        f"--spk-cap-min {a.spk_cap_min:g}",
        "```",
        "", "## Sources and licences", "",
        "- `indicvoices_r`: ai4bharat/indicvoices_r (IndicVoices-R, arXiv 2409.05356), CC-BY-4.0. Field-collected IndicVoices speech already "
        "restored by the authors with HTDemucs + VoiceFixer + DeepFilterNet3 (VoiceFixer is generative, so artifacts are possible). "
        "Hindi stands in for Magadhi, Chhattisgarhi and Bhojpuri, which IndicVoices-R lacks.",
        "- `vaani_tp_tagless`: ARTPARK-IISc/Vaani-transcription-part, CC-BY-4.0, clips whose transcript has no bracketed tag of any kind. "
        "Every competition clip and every recording behind a competition test or validation clip "
        "(`train/data_prep/exclude.json`) is excluded before scoring. Same project and channel as the organisers' beds.",
        "- Both must be disclosed in the system description as external training data.",
        "", "## Manifest fields", "",
        "`path` (relative to this folder), `source`, `language`, `speaker_id` (IndicVoices-R id, or the Vaani speaker token after the "
        "district), `dur` (s), `pesq`/`stoi`/`sisdr` (SQUIM predictions), `vad_frac`, `rms_db`, `peak`, `zero_run_s`, `clip_plateau`, "
        "IndicVoices-R `scenario`/`task_name`/`gender`/`ivr_snr`/`ivr_c50`, Vaani `clip_id`, and provenance `shard`/`row`/`piece`.", ""]
    open(f"{OUT}/README.md", "w").write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["calibrate", "download", "process", "finalize"])
    ap.add_argument("--source", choices=["ivr", "tp"], default="ivr", help="IndicVoices-R or tagless Vaani clips")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--dl-workers", type=int, default=4)
    ap.add_argument("--device", default="cpu", help="SQUIM device; cuda uses ~1.5 GB per worker (1 GB cap + context)")
    ap.add_argument("--shards", type=int, default=20, help="train shards per language, spread over the index")
    ap.add_argument("--langs", default="", help="per-language shard counts, e.g. Hindi:50,Gujarati:99")
    ap.add_argument("--only", action="store_true", help="download only the languages named in --langs")
    ap.add_argument("--min-dur", type=float, default=3.0)
    ap.add_argument("--max-dur", type=float, default=15.0)
    ap.add_argument("--min-vad", type=float, default=0.5)
    ap.add_argument("--max-zero-run", type=float, default=0.5, help="longest |x|<1e-4 run, seconds")
    ap.add_argument("--max-plateau", type=int, default=4, help="identical consecutive samples at the peak (48 kHz)")
    ap.add_argument("--pesq-min", type=float, default=None, help="default: refs' 25th percentile from calibrate")
    ap.add_argument("--hours", type=float, default=12.0, help="hour cap per language")
    ap.add_argument("--hours-lang", default="", help="per-language caps, e.g. Hindi:30")
    ap.add_argument("--spk-cap-min", type=float, default=3.0, help="minutes per speaker per language")
    ap.add_argument("--delete-raw", action="store_true", help="remove each parquet once scored")
    a = ap.parse_args()
    {"calibrate": calibrate, "download": download, "process": process, "finalize": finalize}[a.stage](a)


if __name__ == "__main__":
    main()
