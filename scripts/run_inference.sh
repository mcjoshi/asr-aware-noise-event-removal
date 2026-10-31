#!/bin/bash
# Phase 2 inference: noisy wavs in, enhanced wavs + the submission zip (wavs and SraVaani-1.0 transcripts) out.
#   bash scripts/run_inference.sh <input_data/audio> <output_dir> [submission.zip | none]
# The zip (default <output_dir>/submission.zip) needs gated SraVaani-1.0 access; pass "none" for the enhanced wavs only.
set -e
abs() { python -c "import os, sys; print(os.path.abspath(sys.argv[1]))" "$1"; }
IN=$(abs "${1:?input dir of 16 kHz wavs}"); OUT=$(abs "${2:?output dir}"); ZIP=${3:-$OUT/submission.zip}
[ "$ZIP" = none ] || ZIP=$(abs "$ZIP")
cd "$(dirname "$0")/.."
for f in mossformergan bsrnn_xl tflocoformer_soup fusion; do [ -f checkpoints/$f.pt ] || { bash scripts/download_checkpoints.sh; break; }; done
if [ "$ZIP" = none ]; then
  python inference/infer.py --input-dir "$IN" --output-dir "$OUT" --checkpoints checkpoints
else
  python inference/infer.py --input-dir "$IN" --output-dir "$OUT" --checkpoints checkpoints --zip "$ZIP"
fi
