"""Track 2 inference: noisy 16 kHz wavs in, enhanced wavs (and optionally the Phase 2 submission zip) out.

Every clip goes through the same pipeline:
  1. three members: MossFormerGAN and BSRNN-104M with 4-shift test-time augmentation, TF-Locoformer (weight soup)
  2. learned per-cell fusion of the three estimates
  3. post-processing (span-limit, -50 dB silence gate, 0.99 peak guard), PCM16
  4. with --zip: the mandated ASR (SraVaani-1.0) on exactly those files, then a zip with the wavs and transcripts.jsonl

  python inference/infer.py --input-dir input_data/audio --output-dir out --checkpoints checkpoints --zip submission.zip
"""
import argparse
import io
import json
import os
import sys
import time
import wave
import zipfile

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from track2.audio import SR, fit, ls_scale  # noqa: E402
from track2.fusion import load_net, run_fusion  # noqa: E402
from track2.members import BsrnnMember, LocoMember, MossganMember, enhance_tta  # noqa: E402
from track2.post import post  # noqa: E402

TTA = (0, 41, 83, 124)
CKPTS = {"mossgan": "mossformergan.pt", "bsrnn": "bsrnn_xl.pt", "loco": "tflocoformer_soup.pt", "fusion": "fusion.pt"}
ASR_REPO, ASR_REV = "ARTPARK-IISc/SraVaani-1.0", "f5dd5358325a5208775b91dad98918e079ea2b27"


def pcm16(x):
    """Inverse of libsndfile's PCM16 read (x / 32768), so the zip holds exactly the samples the ASR transcribed."""
    return np.clip(np.round(np.asarray(x, np.float64) * 32768.0), -32768, 32767).astype("<i2")


def enhance_dir(a, ids, dev):
    ck = lambda k: os.path.join(a.checkpoints, CKPTS[k])
    torch.backends.cuda.matmul.allow_tf32 = True
    members = [MossganMember(dev, ck("mossgan")), BsrnnMember(dev, ck("bsrnn")), LocoMember(dev, ck("loco"))]
    for m in members:
        m.net.eval()
    net = load_net(ck("fusion"), dev)
    os.makedirs(a.output_dir, exist_ok=True)
    t0 = time.time()
    for i, c in enumerate(ids):
        path = os.path.join(a.input_dir, f"{c}.wav")
        y, sr = sf.read(path, dtype="float32")
        assert sr == SR and y.ndim == 1, f"{c}: expected 16 kHz mono"
        torch.backends.cuda.matmul.allow_tf32 = True
        with torch.no_grad():
            est = [enhance_tta(members[0], y, TTA), enhance_tta(members[1], y, TTA), enhance_tta(members[2], y, (0,))]
        y64 = y.astype(np.float64)
        X = np.stack([ls_scale(fit(e.astype(np.float64), len(y)), y64) for e in est]).astype(np.float32)
        torch.backends.cuda.matmul.allow_tf32 = False
        e = run_fusion(net, dev, y, X)
        sf.write(os.path.join(a.output_dir, f"{c}.wav"), post(y, e), SR, subtype="PCM_16")
        if (i + 1) % 100 == 0 or i + 1 == len(ids):
            print(f"  {i + 1}/{len(ids)} clips ({time.time() - t0:.0f} s)", flush=True)


def transcribe_and_zip(a, ids):
    from huggingface_hub import snapshot_download
    from transformers import AutoModel
    path = os.environ.get("SRAVAANI") or snapshot_download(ASR_REPO, revision=ASR_REV)
    asr = AutoModel.from_pretrained(path, trust_remote_code=True).to("cuda").eval()
    hyps = asr.transcribe([os.path.join(a.output_dir, f"{c}.wav") for c in ids], return_hypotheses=True)
    texts = {c: getattr(h, "text", str(h)) for c, h in zip(ids, hyps)}
    print(f"ASR done | empty transcripts: {sum(1 for t in texts.values() if not t.strip())}", flush=True)
    with zipfile.ZipFile(a.zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("transcripts.jsonl", "".join(json.dumps({"clip_id": c, "text": texts[c]}, ensure_ascii=False) + "\n" for c in ids))
        for c in ids:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes(pcm16(sf.read(os.path.join(a.output_dir, f"{c}.wav"), dtype="float32")[0]).tobytes())
            z.writestr(f"{c}.wav", buf.getvalue())
    print(f"wrote {a.zip} ({len(ids)} clips)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="directory of noisy 16 kHz mono wavs (<clip_id>.wav)")
    ap.add_argument("--output-dir", required=True, help="enhanced PCM16 wavs are written here, same names")
    ap.add_argument("--checkpoints", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints"))
    ap.add_argument("--zip", default="", help="also transcribe with SraVaani-1.0 and write the Phase 2 submission zip here")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    ids = sorted(f[:-4] for f in os.listdir(a.input_dir) if f.endswith(".wav"))
    print(f"{len(ids)} clips from {a.input_dir}", flush=True)
    enhance_dir(a, ids, a.device)
    if a.zip:
        transcribe_and_zip(a, ids)


if __name__ == "__main__":
    main()
