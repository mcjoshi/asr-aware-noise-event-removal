#!/bin/bash
# Public pretrained start points for training. MossFormerGAN (ClearVoice) and BSRNN (ESPnet, Hugging Face) download
# themselves on first use; TF-Locoformer's DNS-2020 checkpoint comes from MERL's repository (Apache-2.0, git LFS),
# pinned to commit 7a61546.
set -e
cd "$(dirname "$0")/.."
curl -L -o track2/third_party/tflocoformer/model.pth \
  https://media.githubusercontent.com/media/merlresearch/tf-locoformer/7a615460d347ff7334a13dbb831d16280da72cdc/egs2/dns_ins20/enh1/exp/enh_train_enh_tflocoformer_raw/valid.loss.ave_5best.pth
