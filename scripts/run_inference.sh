#!/bin/bash
# Phase 2 inference: noisy wavs in, enhanced wavs + the submission zip (wavs and SraVaani-1.0 transcripts) out.
#   bash scripts/run_inference.sh <input_data/audio> <output_dir> [submission.zip]
set -e
cd "$(dirname "$0")/.."
IN=${1:?input dir of 16 kHz wavs}; OUT=${2:?output dir}; ZIP=${3:-$OUT/submission.zip}
for f in mossformergan bsrnn_xl tflocoformer_soup fusion; do [ -f checkpoints/$f.pt ] || { bash scripts/download_checkpoints.sh; break; }; done
python inference/infer.py --input-dir "$IN" --output-dir "$OUT" --checkpoints checkpoints --zip "$ZIP"
