#!/bin/bash
# Create the environment: bash scripts/setup_env.sh [inference|train]   (Python 3.12 venv in .venv)
set -e
cd "$(dirname "$0")/.."
python3.12 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install torch==2.9.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements_${1:-inference}.txt
pip install --no-deps espnet==202610.post2
python -c "import espnet2.enh.separator.bsrnn_separator, clearvoice; print('environment ready')"
