#!/bin/bash
# The full training pipeline that produced the shipped checkpoints, in order (docs/REPRODUCE.md explains each stage).
#   bash scripts/train_all.sh [all|data|members|dumps|fusion]      (default: all)
# Re-running is safe: a step whose output already exists is skipped, so after a failure run the same command again.
# Data lives in ./data (symlink it if it must sit elsewhere). Gated Hugging Face access and HF_TOKEN are required
# (README, "Access").
set -e
cd "$(dirname "$0")/.."
D=data
PART=${1:-all}
want() { [ "$PART" = all ] || [ "$PART" = "$1" ]; }
# once <output that marks the step done> <command...>
once() { local mark=$1; shift; if [ -e "$mark" ]; then echo "skip (done): $*"; else echo "run: $*"; "$@"; fi; }

if want data; then
  # ---- 0. data (README "Data layout"); the organiser validation set must be in data/validation
  once $D/noisebank/noisebank.json python train/prepare_noisebank.py --val-root $D/validation --out $D/noisebank   # noise events
  once $D/heldout/meta.json        python train/data_prep/make_heldout.py --out $D/heldout                         # the 209 held-out pairs
  once $D/valset_p2f/meta.json     python train/data_prep/make_valset.py --out $D/valset_p2f --n 1200              # valset
  python train/data_prep/ingest_vaani_tp.py --root $D/vaani_tp                                                     # resumable by itself
  once $D/beds_long/manifest.jsonl python train/data_prep/long_beds.py beds --out $D/beds_long                     # long-stream beds
  once $D/beds_large_raw/calib.json python train/data_prep/build_beds_large.py calibrate                          # fusion beds:
  python train/data_prep/build_beds_large.py download --shard-list train/data_prep/bed_shards.json                 # exact shards (skips present ones)
  python train/data_prep/build_beds_large.py process --device cuda --workers 4                                     # skips scored shards
  python train/data_prep/build_beds_large.py download --source tp --shard-list train/data_prep/bed_shards.json
  python train/data_prep/build_beds_large.py process --source tp --device cuda --workers 4
  once $D/beds_large/manifest.jsonl python train/data_prep/build_beds_large.py finalize --hours 15 --hours-lang Hindi:35 --spk-cap-min 6
  # evaluation only; --n 400 is a cap, the eligible speakers give 360 clips
  once $D/valset_long/meta.json    python train/data_prep/long_beds.py valset --out $D/valset_long --n 400
  once $D/fresh2/meta.json         python train/data_prep/make_fusion_mixtures.py --out $D/fresh2 --n 5000 --seed 1 --rate 0.65 \
                                     --dur-median 13 --min-s 8 --max-s 18 --ivr-scan 60000
  once $D/tdt_teacher/val695.pt    python train/data_prep/make_teacher.py --set val                               # SraVaani teacher tokens
  once $D/tdt_teacher/beds_long.pt python train/data_prep/make_teacher.py --set $D/beds_long/manifest.jsonl
  once $D/tdt_teacher/fresh2.pt    python train/data_prep/make_teacher.py --set $D/fresh2/clean_manifest.jsonl --name fresh2
  once track2/third_party/tflocoformer/model.pth bash scripts/download_pretrained.sh
fi

if want members; then
  # ---- 1. members (configs/*.yaml; held-out twins feed the fusion, *_full are shipped). A stage is done when the
  #      checkpoint of its last step exists (mg_sn_a1 ends at its stop_at, 6000).
  for c in mg_sn_a1 mg_S mg_SE mg_SEN mg_LB mg_SE_full mg_SE2_full mg_SEN_full mg_LB_full \
           bsrnnxl_S bsrnnxl_SEN bsrnnxl_LB bsrnnxl_SE_full bsrnnxl_SEN30_full bsrnnxl_LB_full \
           loco_LS loco_MWm loco_MWc loco_MWch6; do
    last=$(python -c "import yaml; y = yaml.safe_load(open('configs/$c.yaml')); print('%06d' % (y.get('stop_at') or y['steps']))")
    once runs/$c/ema_$last.pt python train/train_member.py --config configs/$c.yaml
  done
  once runs/loco_soup.pt python train/soup.py --out runs/loco_soup.pt runs/loco_MWm/ema_004000.pt runs/loco_MWc/ema_004000.pt \
                                              runs/loco_MWch6/ema_004000.pt
fi

if want dumps; then
  # ---- 2. held-out-protocol member outputs on the fusion's training sets (finished clips are skipped). BSRNN always uses
  #      4-shift TTA, MossFormerGAN only on the held-out pairs (as in our run); fresh2: the 2,500 clips the fusion uses.
  TTA=0,41,83,124
  for s in heldout valset_p2f fresh2; do
    mgs=0; [ $s = heldout ] && mgs=$TTA
    lim=0; [ $s = fresh2 ] && lim=2500
    python train/dump_members.py --model mossgan --ckpt runs/mg_LB/ema_004000.pt --in-dir $D/$s/noisy --out $D/$s/members/mg_LB$([ $s = heldout ] && echo _tta) --shifts $mgs --limit $lim
    python train/dump_members.py --model bsrnn_xl --ckpt runs/bsrnnxl_LB/ema_006000.pt --in-dir $D/$s/noisy --out $D/$s/members/bsrnnxl_LB_tta --shifts $TTA --limit $lim
    python train/dump_members.py --model loco --ckpt runs/loco_soup.pt --in-dir $D/$s/noisy --out $D/$s/members/loco_soup --limit $lim
  done
fi

if want fusion; then
  # ---- 3. fusion, then the shipped checkpoints
  once runs/fusion/step_003000.pt python train/train_fusion.py --config configs/fusion.yaml --out runs/fusion
  mkdir -p checkpoints
  cp runs/mg_LB_full/ema_004000.pt checkpoints/mossformergan.pt
  cp runs/bsrnnxl_LB_full/ema_006000.pt checkpoints/bsrnn_xl.pt
  cp runs/loco_soup.pt checkpoints/tflocoformer_soup.pt
  cp runs/fusion/step_003000.pt checkpoints/fusion.pt
fi
