#!/bin/bash
# Every intermediate checkpoint of the shipped training chain (19 files, about 2.9 GB) into runs/, from the public Hugging
# Face model repo. With them, any single stage of scripts/train_all.sh can be run (or checked) on its own: its --init
# file is already in place. See docs/REPRODUCE.md.
set -e
cd "$(dirname "$0")/.."
REPO=${HF_REPO:-manishjoshiai/asr-aware-noise-event-removal}
python -c "
import os
from huggingface_hub import hf_hub_download, list_repo_files
for f in sorted(x for x in list_repo_files('$REPO') if x.startswith('stages/')):
    print(hf_hub_download('$REPO', f, local_dir='.cache_stages'))
"
mkdir -p runs
cp -r .cache_stages/stages/runs/. runs/
rm -rf .cache_stages
