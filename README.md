# ASR-aware noise event removal: unified fusion of three speech enhancers

Team Hydra's system for the IndoML Datathon 2026, Track 2 (noise event removal on Vaani speech).

Final Phase 2 entry `p2_44_soupmwer`: **Combined 19.8973** (SI-SDR 15.7934 dB on the synthetic clips; ΔWER +11.66 pp synthetic, +0.54 pp natural, +4.1039 pp pooled). The organisers' ASR audit passed.

Every clip goes through the same pipeline. There is no clip classifier or router, no natural/synthetic branch, no pass-through and no template, noise-dictionary or correlation matching.

```
noisy clip -> MossFormerGAN (4-shift TTA) -+
           -> BSRNN 104M    (4-shift TTA) -+-> learned per-cell fusion -> post-processing -> enhanced wav
           -> TF-Locoformer (weight soup) -+    (0.34M params)            (span-limit, -50 dB gate, peak guard)
```

- **Members:** all three are fine-tuned through the frozen scoring ASR (SraVaani-1.0).
  - Losses: SI-SDR, ASR encoder distance and TDT transcript NLL; one of the three averaged TF-Locoformer fine-tunes also uses a deletion-weighted MWER term.
  - Training data: the organisers' validation pairs, plus a long single-recording (12-18 s) ASR stream.
- **Fusion:** predicts per time-frequency cell softmax weights over the members, plus a bounded readmission of the input.
  - Trained on out-of-sample member outputs, through the shipped post-processing.
  - Losses: SI-SDR, encoder distance, transcript NLL and MWER.

The technical report is `docs/technical_report.pdf`.
**To retrain and check the result stage by stage, follow [`docs/REPRODUCE.md`](docs/REPRODUCE.md)**: every stage, its expected numbers, the checks we ran and the known differences.

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
  download_stage_checkpoints.sh  all 19 intermediate training checkpoints (to re-run or check any single stage)
  train_all.sh                 the full training pipeline in order
checkpoints/                   final checkpoints (downloaded; see below)
docs/                          technical report, REPRODUCE.md (reproduction guide), figures
```

## Environment and hardware

| item | value |
|---|---|
| OS | Ubuntu 22.04.4 LTS, Linux kernel 6.8 |
| Python | 3.12.4 |
| PyTorch | 2.9.0 (CUDA 12.8 wheels; they bundle the CUDA runtime and cuDNN 9.10, so no system CUDA toolkit is needed) |
| NVIDIA driver | 535.288.01 (tested) |
| GPU | NVIDIA A100 80 GB PCIe (4 in the machine; each training run used one) |
| GPU memory | member training 20-40 GB per run (activation checkpointing, bf16 attention); fusion training about 10 GB; inference fits on one GPU |
| CPU / RAM | 96 cores, 866 GB RAM; training streams hold the organiser pairs and long beds in RAM (about 20 GB); data preparation uses 24-48 worker processes |

**System dependencies:** an NVIDIA GPU with a CUDA 12 driver (inference and training both run on CUDA), Python 3.12 with `venv` (we used Miniconda's Python 3.12.4; Ubuntu 22.04 ships Python 3.10, so use conda, pyenv or the deadsnakes PPA: `add-apt-repository ppa:deadsnakes/ppa && apt install python3.12 python3.12-venv`), `curl` and `git`.
Internet access to huggingface.co is needed on first use (ClearVoice and ESPnet download their base networks before our weights are loaded).
`soundfile` ships its own libsndfile, so no audio system library is needed.

**Disk:** inference needs about 2 GB (final checkpoints 0.5 GB, SraVaani-1.0 0.9 GB, the ESPnet BSRNN base network 0.42 GB, ClearVoice's MossFormerGAN 0.14 GB).
Rebuilding the training data needs several hundred GB at peak: the Vaani ingest streams the whole release (its kept wavs are about 60 GB), the raw bed shards take about 300 GB (`build_beds_large.py process --delete-raw` frees each shard once scored), the final beds 23 GB, and `fresh2` with its member outputs about 64 GB.

```bash
bash scripts/setup_env.sh inference      # or: train   (creates .venv; see requirements_inference.txt for the espnet note)
. .venv/bin/activate
```

Both environments were built from scratch with `scripts/setup_env.sh`.
The inference environment reproduced the shipped audio (see "Reproduction check"); in the training environment every training and data-preparation entry point starts.
The rendered figures need `matplotlib` only to be redrawn (`docs/figures/src/system.py`).

### Access (gated Hugging Face repositories)

Three resources are gated: request access on each page, then log in (`huggingface-cli login`) or `export HF_TOKEN=...` (the data-preparation scripts read `HF_TOKEN`).

| resource | needed for |
|---|---|
| [ARTPARK-IISc/SraVaani-1.0](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0) | the transcripts in the submission zip (`infer.py --zip`), and all ASR losses in training |
| [ARTPARK-IISc/Vaani-transcription-part](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part) | training data (long beds, fusion beds) |
| [ai4bharat/indicvoices_r](https://huggingface.co/datasets/ai4bharat/indicvoices_r) | training data (fusion beds) |

`SRAVAANI=/path/to/snapshot` points inference and training at a local SraVaani-1.0 copy instead of downloading it.
Enhanced wavs alone (`infer.py` without `--zip`, or `run_inference.sh <in> <out> none`) need no gated access.

### Pretrained models

| model | source | version / revision | license | role |
|---|---|---|---|---|
| SraVaani-1.0 (TDT ASR) | [ARTPARK-IISc/SraVaani-1.0](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0) | `f5dd5358325a5208775b91dad98918e079ea2b27` | MIT | mandated scorer; transcripts in the zip; frozen training loss (never updated) |
| MossFormerGAN_SE_16K | [alibabasglab/MossFormerGAN_SE_16K](https://huggingface.co/alibabasglab/MossFormerGAN_SE_16K), fetched by [ClearVoice](https://github.com/modelscope/ClearerVoice-Studio) `clearvoice==0.1.2` into `./checkpoints/` | `dcb1b20e1e44d435090072ab9b223c792ea1f624` | Apache-2.0 | start point of member 1 |
| BSRNN large-double (104M), VCTK+DNS-2020+WHAMR! | [wyz/vctk_dns2020_whamr_bsrnn_large_double_noncausal](https://huggingface.co/wyz/vctk_dns2020_whamr_bsrnn_large_double_noncausal) (ESPnet) | `da049f2c9fca1b6f0c09c700694dfcec03703008`, `91epoch.pth` | CC-BY-4.0 | start point of member 2 |
| TF-Locoformer M, DNS-2020 | [merlresearch/tf-locoformer](https://github.com/merlresearch/tf-locoformer) | commit `7a615460d347ff7334a13dbb831d16280da72cdc`, `egs2/dns_ins20/enh1/exp/enh_train_enh_tflocoformer_raw/valid.loss.ave_5best.pth` | Apache-2.0 | start point of member 3 |
| Silero VAD | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | `silero-vad==6.2.3` | MIT | bed screening (data preparation only) |
| SQUIM objective | `torchaudio.pipelines.SQUIM_OBJECTIVE` | `torchaudio==2.9.0` | BSD-2-Clause | bed quality screening (data preparation only) |

MossFormerGAN and BSRNN download automatically on first use.
TF-Locoformer is needed only for training (`scripts/download_pretrained.sh`), because the final checkpoint contains all its weights.

## Inference

```bash
bash scripts/download_checkpoints.sh             # 4 files, about 490 MB
bash scripts/run_inference.sh /path/to/input_data/audio out/ out/submission.zip    # "none" instead of the zip: wavs only
# or
python inference/infer.py --input-dir /path/to/input_data/audio --output-dir out --checkpoints checkpoints --zip out/submission.zip
```

- **Input:** a directory of noisy 16 kHz mono wavs named `<clip_id>.wav` (the Phase 2 `input_data/audio`).
- **Output:**
  - `out/<clip_id>.wav`: enhanced audio, 16 kHz mono PCM16, same length as the input.
  - `--zip`: the Phase 2 submission zip. The wavs and `transcripts.jsonl` (`{"clip_id", "text"}`) sit at the zip root. The transcripts are SraVaani-1.0 run on exactly the zipped audio.
- **Time and memory:** about 10 s per clip on one A100 including model loading, dominated by the two 4-shift TTA members (20 test clips, 5.9 minutes of audio, took 194 s); peak GPU memory 16.8 GB.

**Reproduction check:** from a fresh clone of this repository, with the environment built by `scripts/setup_env.sh` and the checkpoints downloaded from Hugging Face, `run_inference.sh` reproduced the shipped `p2_44_soupmwer` audio on 20 Phase 2 clips (4 s to 39 s) to within 1 LSB of 16-bit PCM on every sample, and all 20 transcripts were identical. (In an earlier 7-clip check, 6 transcripts were identical; in the seventh a 1-LSB difference changed the ASR's greedy decode, which can happen.)

Checkpoints are on Hugging Face, [manishjoshiai/asr-aware-noise-event-removal](https://huggingface.co/manishjoshiai/asr-aware-noise-event-removal):
- `mossformergan.pt`
- `bsrnn_xl.pt`
- `tflocoformer_soup.pt`
- `fusion.pt` (its `cfg` holds the network configuration)

## Training

`scripts/train_all.sh` lists every stage in order. Each member stage is `python train/train_member.py --config configs/<stage>.yaml`.
See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for the stage table, expected results and how to check a stage against our intermediate checkpoints.

**Data layout (`./data`; symlink it if the data must live elsewhere, since configs refer to `data/...`; teacher files are keyed by absolute paths, so build them where the data will stay):**

| path | content | made by |
|---|---|---|
| `validation/` | organiser validation release, unpacked as `validationMetadata.json`, `syntheticNoiseAudio/`, `syntheticCleanRefAudio/` and `naturalNoisyAudio/` | organisers |
| `noisebank/` | 2,213 noise events (noisy minus clean of the validation pairs) and the speaker split | `train/prepare_noisebank.py` |
| `heldout/` | the 209 held-out validation pairs | `train/data_prep/make_heldout.py` |
| `vaani_tp/` | Vaani transcription release, after exclusions | `train/data_prep/ingest_vaani_tp.py` |
| `beds_long/` | 1,450 single-speaker 12-18 s recordings (long ASR stream) | `train/data_prep/long_beds.py beds` |
| `beds_large/` | clean Vaani tagless and IndicVoices-R beds, from the exact shards and dataset revisions in `train/data_prep/bed_shards.json` | `train/data_prep/build_beds_large.py` |
| `valset_p2f/` | 1,200 simulated Phase-2-shaped clips on held-out speakers | `train/data_prep/make_valset.py` |
| `valset_long/` | 360 test-length clips, one per unseen speaker (evaluation only) | `train/data_prep/long_beds.py valset` |
| `fresh2/` | 5,000 fusion mixtures, 8-18 s, each 1-4 beds of one speaker joined with 50 ms gaps (first 2,500 used) | `train/data_prep/make_fusion_mixtures.py` |
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
- **External data:** two public speech corpora supply clean speech only. Every competition clip and every recording behind a competition clip was excluded (`train/data_prep/exclude.json`).

| data (version) | what we used | how | in the final solution |
|---|---|---|---|
| organiser synthetic pairs | 695 pairs (1.0 h) | member training; 209 held out for scoring and fusion training | yes |
| organiser noise events (noisy minus clean) | 2,213 events (0.4 h) | pasted into training mixtures only, all of them (the held-out pairs have unseen speakers, not unseen noises); never used at inference | yes |
| organiser natural recordings | 555 of 1,806 | evaluation only | no |
| [ARTPARK-IISc/Vaani-transcription-part](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part) (`d2acadff`) | 1,450 long recordings (5.7 h) | long single-recording ASR stream for all three members | yes |
| same | 710 clean beds (1.4 h), in 908 of the 2,500 fusion clips (3.2 h) | fusion training mixtures (`fresh2`) | yes |
| [ai4bharat/indicvoices_r](https://huggingface.co/datasets/ai4bharat/indicvoices_r) (`5f4495c9`) | 2,793 clean beds (6.5 h), in 1,592 of the 2,500 fusion clips (6.7 h) | fusion training mixtures (`fresh2`) | yes |
| same | 198 h more clean speech | member training experiment; no gain | no |

  No external noise data, no synthetic (TTS) speech and no test audio is used. A transcript check found no test clip that shares 30% of its word 8-grams with any of the 6,450 training recordings.
- **No test audio** is used anywhere in training. The only test statistics we used are the clip-length distribution and an estimated SNR distribution of the unlabelled test inputs, to shape the simulated sets and to weight held-out scores.
- **`train/data_prep/exclude.json`** lists the competition clip names (validation, Track 1 training release, Phase 1 test) and the Vaani recordings behind competition validation and test clips; ingestion drops every clip on it.

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
  - 2,500 fresh mixtures (153 of them held back as the dev set, so 2,347 train);
  - the 209 held-out pairs;
  - the 1,200 simulated clips.
- MossFormerGAN's inputs are single-pass on fresh2 and valset and 4-shift TTA on the held-out pairs, as in our run; at inference all MossFormerGAN and BSRNN outputs use 4-shift TTA.
- 3,000 steps, 8 s crops, SI-SDR plus encoder (30), transcript NLL (2.5) and MWER (3).
- Every step goes through the shipped post-processing.

**Training time** (wall-clock on one A100 80 GB shared with other jobs, so an upper bound):

| stage | configs | time |
|---|---|---|
| MossFormerGAN, held-out twin | `mg_sn_a1`, `mg_S`, `mg_SE`, `mg_SEN`, `mg_LB` | 3 h + 12 h + 1 h + 1 h + 6 h |
| MossFormerGAN, shipped | `mg_SE_full`, `mg_SE2_full`, `mg_SEN_full`, `mg_LB_full` | 1 h + 5 h + 3 h + 6 h |
| BSRNN-104M, held-out twin | `bsrnnxl_S`, `bsrnnxl_SEN`, `bsrnnxl_LB` | 11 h + 4 h + 18 h |
| BSRNN-104M, shipped | `bsrnnxl_SE_full`, `bsrnnxl_SEN30_full`, `bsrnnxl_LB_full` | 21 h + 4 h + 8 h |
| TF-Locoformer | `loco_LS`, then `loco_MWm`, `loco_MWc`, `loco_MWch6` | 25 h, then 8 h + 5 h + 8 h |
| Fusion | `fusion.yaml` | about 5 h |

Data preparation (Vaani ingest, bed scoring, member outputs) was not timed end to end; it is dominated by downloads and SQUIM scoring.

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
