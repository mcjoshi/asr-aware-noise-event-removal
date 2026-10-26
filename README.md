# IndoML Datathon 2026, Track 2 (Noise Event Removal): unified ASR-aware three-model fusion

Final Phase 2 entry `p2_44_soupmwer`: **Combined 19.8973** (SI-SDR 15.79 dB on the synthetic clips; ΔWER +11.66 pp synthetic, +0.54 pp natural, +4.10 pp pooled). The organisers' ASR audit passed.

Every clip goes through the same pipeline. There is no clip classifier or router, no natural/synthetic branch, no pass-through and no template, noise-dictionary or correlation matching.

```
noisy clip -> MossFormerGAN (4-shift TTA) -+
           -> BSRNN 104M    (4-shift TTA) -+-> learned per-cell fusion -> post-processing -> enhanced wav
           -> TF-Locoformer (weight soup) -+    (0.35M params)            (span-limit, -50 dB gate, peak guard)
```

- **Members:** all three are fine-tuned through the frozen scoring ASR (SraVaani-1.0).
  - Losses: SI-SDR, ASR encoder distance, TDT transcript NLL, and a deletion-weighted MWER term for TF-Locoformer.
  - Training data: the organisers' validation pairs, plus a long single-recording (12-18 s) ASR stream.
- **Fusion:** predicts per time-frequency cell softmax weights over the members, plus a bounded readmission of the input.
  - Trained on out-of-sample member outputs, through the shipped post-processing.
  - Losses: SI-SDR, encoder distance, transcript NLL and MWER.

The technical report is `docs/technical_report.pdf`.

## Repository layout

```
README.md
requirements_inference.txt     inference environment
requirements_train.txt         training environment (inference + data preparation)
inference/infer.py             Phase 2 inference: wav dir -> enhanced wavs (+ submission zip with SraVaani transcripts)
track2/                        shared code
  members.py                   MossFormerGAN, BSRNN-104M, TF-Locoformer wrappers (enhance + training losses)
  mossgan.py                   ClearVoice MossFormerGAN decode path, differentiable
  fusion.py                    FusionNet (per-cell weights + bounded readmission)
  post.py                      post-processing (identical for every clip)
  losses.py                    losses through the scoring ASR: encoder distance, TDT NLL, MWER
  audio.py, text.py            signal helpers, the organisers' transcript normaliser
  third_party/tflocoformer/    MERL TF-Locoformer separator (Apache-2.0) and its config
train/
  train_member.py              member fine-tuning (configs/<stage>.yaml)
  train_fusion.py              fusion training (configs/fusion.yaml)
  data.py                      training streams (organiser pairs, re-mixes, long single-recording stream)
  prepare_noisebank.py         noise events from the validation pairs (training augmentation only)
  dump_members.py              member outputs on a set (fusion training inputs)
  soup.py                      weight soup of TF-Locoformer fine-tunes
  data_prep/                   external data ingest and the simulated sets (see "Training")
configs/                       one yaml per training stage (exact hyper-parameters of the shipped models)
scripts/
  run_inference.sh             one-command inference
  download_checkpoints.sh      final checkpoints from Hugging Face
  download_pretrained.sh       MERL TF-Locoformer start point (training only)
  train_all.sh                 the full training pipeline in order
checkpoints/                   final checkpoints (downloaded; see below)
docs/                          technical report, figures
```

## Environment and hardware

| item | value |
|---|---|
| Python | 3.12.4 |
| PyTorch | 2.9.0 (CUDA 12.8 build), cuDNN 9.10 |
| NVIDIA driver | 535.288.01 |
| GPU | NVIDIA A100 80 GB PCIe (4 in the machine; each training run used one) |
| GPU memory | member training 20-40 GB per run (activation checkpointing, bf16 attention); fusion training about 10 GB per run; inference runs on one GPU |
| CPU / RAM | 96 cores; training streams load the organiser pairs and long beds into RAM (about 20 GB) |
| OS | Linux (Ubuntu, kernel 6.8) |

```bash
bash scripts/setup_env.sh inference      # or: train   (creates .venv; see requirements_inference.txt for the espnet note)
. .venv/bin/activate
```

The environment was tested from scratch: in a fresh venv, `scripts/setup_env.sh` followed by the inference command reproduced the shipped audio (see "Reproduction check").

### Pretrained models

| model | source | version / revision | license | role |
|---|---|---|---|---|
| SraVaani-1.0 (TDT ASR) | [ARTPARK-IISc/SraVaani-1.0](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0) | `f5dd5358325a5208775b91dad98918e079ea2b27` | as published | mandated scorer; transcripts in the zip; frozen training loss (never updated) |
| MossFormerGAN_SE_16K | [ClearVoice](https://github.com/modelscope/ClearerVoice-Studio) (`clearvoice==0.1.2`) | ClearVoice release checkpoint | Apache-2.0 | start point of member 1 |
| BSRNN large-double (104M), VCTK+DNS-2020+WHAMR! | [wyz/vctk_dns2020_whamr_bsrnn_large_double_noncausal](https://huggingface.co/wyz/vctk_dns2020_whamr_bsrnn_large_double_noncausal) (ESPnet) | `91epoch.pth` | CC-BY-4.0 | start point of member 2 |
| TF-Locoformer M, DNS-2020 | [merlresearch/tf-locoformer](https://github.com/merlresearch/tf-locoformer) | `egs2/dns_ins20/enh1/.../valid.loss.ave_5best.pth` | Apache-2.0 | start point of member 3 |

The first two download automatically on first use.
TF-Locoformer is needed only for training (`scripts/download_pretrained.sh`), because the final checkpoint contains all its weights.

## Inference

```bash
bash scripts/download_checkpoints.sh             # 4 files, about 490 MB
bash scripts/run_inference.sh /path/to/input_data/audio out/ out/submission.zip
# or
python inference/infer.py --input-dir /path/to/input_data/audio --output-dir out --checkpoints checkpoints --zip out/submission.zip
```

- **Input:** a directory of noisy 16 kHz mono wavs named `<clip_id>.wav` (the Phase 2 `input_data/audio`).
- **Output:**
  - `out/<clip_id>.wav`: enhanced audio, 16 kHz mono PCM16, same length as the input.
  - `--zip`: the Phase 2 submission zip. The wavs and `transcripts.jsonl` (`{"clip_id", "text"}`) sit at the zip root. The transcripts are SraVaani-1.0 run on exactly the zipped audio.
- **Time:** a few seconds per clip on one A100, dominated by the two 4-shift TTA members (7 clips totalling 2.4 minutes of audio took 79 s).

**Reproduction check:** in a fresh environment built by `scripts/setup_env.sh`, the script reproduced the shipped `p2_44_soupmwer` audio on 7 Phase 2 clips (4 s to 39 s) to within 1 LSB of 16-bit PCM on every sample. 6 of the 7 transcripts were identical. In the seventh, a 1-LSB difference changed the ASR's greedy decode.

Checkpoints are on Hugging Face, [manishjoshiai/indoml2026-track2-noise-removal](https://huggingface.co/manishjoshiai/indoml2026-track2-noise-removal):
- `mossformergan.pt`
- `bsrnn_xl.pt`
- `tflocoformer_soup.pt`
- `fusion.pt` (its `cfg` holds the network configuration)

## Training

`scripts/train_all.sh` lists every stage in order. Each member stage is `python train/train_member.py --config configs/<stage>.yaml`.

**Data layout (`$TRACK2_DATA`, default `./data`):**

| path | content | made by |
|---|---|---|
| `validation/` | organiser validation set (`validationMetadata.json`, synthetic noisy and clean pairs, natural clips) | organisers |
| `noisebank/` | 2,213 noise events (noisy minus clean of the validation pairs) and the speaker split | `train/prepare_noisebank.py` |
| `heldout/` | the 209 held-out validation pairs | `train/data_prep/make_heldout.py` |
| `vaani_tp/` | Vaani transcription release, after exclusions | `train/data_prep/ingest_vaani_tp.py` |
| `beds_long/` | 1,450 single-speaker 12-18 s recordings (long ASR stream) | `train/data_prep/long_beds.py beds` |
| `beds_large/` | clean Vaani tagless and IndicVoices-R beds | `train/data_prep/build_beds_large.py` |
| `valset_p2f/` | 1,200 simulated Phase-2-shaped clips on held-out speakers | `train/data_prep/make_valset.py` |
| `fresh2/` | 5,000 test-shaped fusion mixtures (first 2,500 used) | `train/data_prep/make_fusion_mixtures.py` |
| `tdt_teacher/` | SraVaani greedy tokens of every clean bed | `train/data_prep/make_teacher.py` |

**Training data and augmentation:**
- **Organiser validation set (Vaani):**
  - 695 synthetic pairs train the members.
  - In the held-out protocol, 209 pairs from 30% of the speakers are excluded and used only for scoring and fusion training.
  - The shipped MossFormerGAN and BSRNN use all 695 pairs.
- **Augmentation (`train/data.py`):**
  - Half of the 2-4 s crops are real pairs.
  - The other half re-mix a clean reference with 1-3 validation noise events, at an SNR drawn from the real pairs.
- **Long stream:** each step adds a micro-batch of 12-18 s windows from `beds_long`. Validation-pool events are pasted as the organisers' generator does (Poisson 0.5 events/s, log-normal gain around 0.93). These windows train only the ASR losses.
- **External data:**
  - [ARTPARK-IISc/Vaani-transcription-part](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part) (gated) supplies the long beds and part of the fusion beds.
  - Every competition clip and every recording behind a competition clip was excluded (`train/data_prep/exclude.json`).
  - [ai4bharat/indicvoices_r](https://huggingface.co/datasets/ai4bharat/indicvoices_r) supplies fusion beds only.
- **No test audio** is used anywhere in training.

**Final members** (`configs/`):
- **MossFormerGAN:** 9 stages.
  - Held-out twin `mg_LB`, used to produce the fusion's training inputs.
  - Shipped `mg_LB_full`.
- **BSRNN-104M:** 6 stages.
  - Held-out twin `bsrnnxl_LB`.
  - Shipped `bsrnnxl_LB_full`.
- **TF-Locoformer:** `loco_LS` (20k steps from MERL's checkpoint, long stream from step 0), then three 4k continuations (MWER, plain, louder events), averaged by `train/soup.py`.

**Fusion (`configs/fusion.yaml`):**
- Trained on held-out-protocol member outputs over:
  - 2,500 fresh mixtures;
  - the 209 held-out pairs;
  - the 1,200 simulated clips.
- 3,000 steps, 8 s crops, SI-SDR plus encoder (30), transcript NLL (2.5) and MWER (3).
- Every step goes through the shipped post-processing.

## Notes on compliance

- One approach for every clip; nothing inspects a clip to choose its processing.
- The validation noise events are used only to synthesise training mixtures. Nothing at inference reads them, matches against them or subtracts them.
- Every clip ships model-processed audio.
  - The post-processing's span-limit keeps the input only where the model left the audio essentially unchanged.
  - On the test clips, the shipped audio differs from the pure model output by -41 dB (median).
- Pretrained models and external datasets are listed above.

## License

Our code is released under the Apache License 2.0 (`LICENSE`).
`track2/third_party/tflocoformer/` is MERL's code under Apache-2.0.
The released checkpoints derive from the pretrained models listed above and follow their licences.
