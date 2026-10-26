#!/bin/bash
# Final checkpoints (MossFormerGAN, BSRNN-104M, TF-Locoformer soup, fusion) from the public Hugging Face model repo.
set -e
cd "$(dirname "$0")/.."
REPO=${HF_REPO:-manishjoshiai/indoml2026-track2-noise-removal}
python -c "
from huggingface_hub import hf_hub_download
for f in ['mossformergan.pt', 'bsrnn_xl.pt', 'tflocoformer_soup.pt', 'fusion.pt']:
    print(hf_hub_download('$REPO', f'checkpoints/{f}', local_dir='.'))
"
