# Reproducing the Track 2 system

This guide takes you from the public pretrained models to the four shipped checkpoints, stage by stage, and says what to expect at each step.
The README covers the environment, hardware and gated-data access; set those up first.

## What can be reproduced, and how exactly

| level | what you run | what you get | verified by us |
|---|---|---|---|
| 1. Inference | `scripts/run_inference.sh` with the released checkpoints | the submitted Phase 2 audio, within 1 LSB of 16-bit PCM; the same transcripts | yes (see "Checks we ran") |
| 2. One training stage | any stage of `scripts/train_all.sh`, started from our released checkpoint of the previous stage | the same training run as ours: same data order, losses and evaluation, up to GPU floating-point nondeterminism | yes, one stage per model family |
| 3. The whole chain | `scripts/train_all.sh` from the public pretrained models | models of the same quality; not bit-identical weights | the data steps and the trainers are checked; the full 150 GPU-hour chain was not re-run |

GPU training is not bit-reproducible across runs, cards or library versions: nondeterministic GPU kernels change the last bits, and the differences grow over thousands of steps.
For a reference point, two seeds of the same 6k-step continuation differ by about 0.35 pp ΔWER and a few hundredths of a dB.
The intermediate checkpoints make each stage checkable on its own, so a difference can be located instead of compounding over the chain.

## 1. Data

Run the data section of `scripts/train_all.sh` in order. Everything lands in `./data` (symlink it elsewhere if needed).

| step | command | output | deterministic | notes |
|---|---|---|---|---|
| noise events | `train/prepare_noisebank.py` | `data/noisebank/` (2,213 events, speaker split) | yes | rebuilt bit-identical to ours |
| held-out pairs | `train/data_prep/make_heldout.py` | `data/heldout/` (209 pairs) | yes | rebuilt bit-identical to ours |
| valset | `train/data_prep/make_valset.py` | `data/valset_p2f/` (1,200 clips) | yes (since the fix below) | see "Known differences" |
| Vaani ingest | `train/data_prep/ingest_vaani_tp.py` | `data/vaani_tp/` | yes, given the same dataset revision | streams the whole gated release |
| long beds | `train/data_prep/long_beds.py beds` | `data/beds_long/` (1,450 recordings) | yes | |
| fusion beds | `train/data_prep/build_beds_large.py` (calibrate, download, process, finalize) | `data/beds_large/` | yes, given `bed_shards.json` | the exact shards and dataset revisions are pinned |
| valset_long | `train/data_prep/long_beds.py valset` | `data/valset_long/` (360 clips) | yes | evaluation only |
| fresh2 | `train/data_prep/make_fusion_mixtures.py` | `data/fresh2/` (5,000 clips, first 2,500 used) | yes | rebuilt bit-identical to ours (all 5,000 clips) |
| teachers | `train/data_prep/make_teacher.py` | `data/tdt_teacher/` | yes (greedy decoding) | keys are absolute paths: build them where the data will stay |

## 2. Members

Each stage is `python train/train_member.py --config configs/<stage>.yaml`.
A stage starts from the `init` checkpoint named in its config, so stages must run in the order below, or you can download our checkpoints of the earlier stages first:

```bash
bash scripts/download_stage_checkpoints.sh     # all 19 intermediate checkpoints into runs/
python train/train_member.py --config configs/loco_MWm.yaml   # e.g. re-run one stage on its own
```

Every run writes `runs/<stage>/log.jsonl` with an evaluation on the 209 held-out pairs every `eval_every` steps.
`ema_raw` is the SI-SDR (dB) of the EMA weights on those pairs, `enc_dist` and `nll` the ASR losses there.
For stages trained on all 695 pairs (`*_full`) the held-out pairs are in the training set, so their numbers are a consistency check, not a held-out score.

| stage | starts from | steps | wall-clock* | our final `ema_raw` (dB) |
|---|---|---|---|---|
| `mg_sn_a1` | ClearVoice MossFormerGAN | 6,000 used (schedule 40k) | 3 h | 18.39 at 10k |
| `mg_S` | `mg_sn_a1/ema_006000` | 34,000 | 12 h | 19.04 |
| `mg_SE` | `mg_S/ema_034000` | 4,000 | 1 h | 18.99 |
| `mg_SEN` | `mg_SE/ema_004000` | 4,000 | 1 h | 19.01 |
| `mg_LB` (held-out twin) | `mg_SEN/ema_004000` | 4,000 | 6 h | 19.01 |
| `mg_SE_full` | `mg_S/ema_034000` | 4,000 | 1 h | 19.14 |
| `mg_SE2_full` | `mg_SE_full/ema_004000` | 8,000 | 5 h | 19.35 |
| `mg_SEN_full` | `mg_SE2_full/ema_008000` | 4,000 | 3 h | 19.39 |
| `mg_LB_full` (shipped) | `mg_SEN_full/ema_004000` | 4,000 | 6 h | 19.41 |
| `bsrnnxl_S` | ESPnet BSRNN `91epoch.pth` | 30,000 | 11 h | 21.90 |
| `bsrnnxl_SEN` | `bsrnnxl_S/ema_030000` | 6,000 | 4 h | 21.89 |
| `bsrnnxl_LB` (held-out twin) | `bsrnnxl_SEN/ema_006000` | 6,000 | 18 h | 21.95 |
| `bsrnnxl_SE_full` | ESPnet BSRNN `91epoch.pth` | 30,000 | 21 h | 24.95 |
| `bsrnnxl_SEN30_full` | `bsrnnxl_SE_full/ema_030000` | 6,000 | 4 h | 25.11 |
| `bsrnnxl_LB_full` (shipped) | `bsrnnxl_SEN30_full/ema_006000` | 6,000 | 8 h | 25.17 |
| `loco_LS` | MERL TF-Locoformer (`scripts/download_pretrained.sh`) | 20,000 | 25 h | 22.28 |
| `loco_MWm`, `loco_MWc`, `loco_MWch6` | `loco_LS/ema_020000` | 4,000 each | 8 h, 5 h, 8 h | 22.20, 22.25, 22.20 |

\* On an A100 80 GB shared with other jobs, so an upper bound.

The shipped TF-Locoformer is the weight average of the last three: `python train/soup.py --out runs/loco_soup.pt runs/loco_MWm/ema_004000.pt runs/loco_MWc/ema_004000.pt runs/loco_MWch6/ema_004000.pt`.
From our three checkpoints this reproduces the shipped `tflocoformer_soup.pt` bit-exactly.

## 3. Fusion

1. Run the three held-out-protocol members (`mg_LB`, `bsrnnxl_LB`, the soup) over `data/heldout`, `data/valset_p2f` and the first 2,500 `data/fresh2` clips (section 2 of `train_all.sh`).
2. `python train/train_fusion.py --config configs/fusion.yaml --out runs/fusion` (3,000 steps, about 5 h, about 10 GB).
3. Copy the four checkpoints into `checkpoints/` (end of `train_all.sh`) and run inference.

## Checks we ran

We re-ran the released scripts in a fresh folder and compared them with the runs that produced the shipped checkpoints (October 2026, on the machine described in the README).

**Data.** Each script was re-run and its output compared file by file with the data our models were trained on:

| data | files compared | result |
|---|---|---|
| noise bank (`prepare_noisebank.py`) | 2,214 | identical, sample for sample |
| held-out pairs (`make_heldout.py`) | 418 | identical |
| fresh2 (`make_fusion_mixtures.py`) | 10,002 (5,000 noisy, 5,000 clean, metadata) | identical |
| valset (`make_valset.py`) | 2,401 | same clips, beds, events and placements; low-level gap noise differs (see "Known differences"); two runs of the released script are identical |

**Trainers.** For one stage of each model and for the fusion, the released trainer and the research trainer that produced our checkpoints were started from the same checkpoint, with the same seed, data and arguments, for 60 steps with an evaluation every 30:

| stage | metric at step 60 | research trainer | released trainer | research trainer, second run |
|---|---|---|---|---|
| `bsrnnxl_LB_full` | training loss | −21.3692 | −21.3671 | |
| | held-out `ema_raw` (dB) | 25.10591 | 25.10589 | |
| `loco_MWm` (with MWER) | training loss | −21.7240 | −21.7274 | −21.7167 |
| | held-out `ema_raw` (dB) | 22.27917 | 22.27926 | |
| | long-window ASR loss | 0.975 | 0.884 | 0.991 |
| fusion (`fusion.yaml`) | training loss | −15.696 | −15.706 | −15.614 |
| `mg_LB_full` | held-out `raw` at step 0 | 19.369 | 19.369 | |

The two trainers agree to the precision that GPU nondeterminism allows: their gap is no larger than the gap between two runs of the same trainer.
The MWER loss samples hypotheses, so it varies most between runs.
Logged `post` columns of member runs are not comparable with our logs (our research trainer evaluated with an older post-processing setting); `raw` and `ema_raw` are.

**Inference.** In a fresh environment, `infer.py` with the released checkpoints reproduced the submitted audio to within 1 LSB of 16-bit PCM on every sample and gave identical transcripts.

**Environments.** `scripts/setup_env.sh inference` and `scripts/setup_env.sh train` both build from scratch, and every training and data-preparation script starts in the training environment.

## Known differences from our original run

- **valset noise floor.** Our original `valset_p2f` was built by a version of `make_valset.py` that seeded the noise filling each bed's silent gaps by the order of a Python `set`, which changes from run to run. The released script visits the beds in sorted order, so it is now deterministic, but no run can reproduce our original copy bit-exactly. The clips, beds, events and placements are identical; the differences are in that low-level gap noise (at most 0.4% of full scale in 1,307 of 2,400 files). valset is one of the three fusion training sets.
- **Vaani revision.** The fusion and long beds were downloaded at dataset revision `d2acadff`; the earlier Vaani ingest's revision was not recorded. If the release has changed since, a few beds may differ.
- **MossFormerGAN `mg_sn_a1`** ran 10k of its 40k-step schedule; only `ema_006000` is used, so the run can be stopped there.
- **GPU nondeterminism**, as above.
