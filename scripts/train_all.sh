#!/bin/bash
# The full training pipeline that produced the shipped checkpoints, in order. Data lives in ./data (symlink it if it
# must sit elsewhere); configs and teachers refer to data/... paths. Gated Hugging Face access and HF_TOKEN are required
# (README, "Access"). Wall-clock times per stage are in the README ("Training time").
set -e
cd "$(dirname "$0")/.."
D=data
T=train/train_member.py

# ---- 0. data (README "Data layout"); the organiser validation set must be in data/validation
python train/prepare_noisebank.py --val-root $D/validation --out $D/noisebank        # noise events from the validation pairs
python train/data_prep/make_heldout.py --out $D/heldout                              # the 209 held-out pairs
python train/data_prep/make_valset.py --out $D/valset_p2f --n 1200                   # simulated Phase-2-shaped clips (valset)
python train/data_prep/ingest_vaani_tp.py --root $D/vaani_tp                         # Vaani transcription release, exclusions applied
python train/data_prep/long_beds.py beds --out $D/beds_long                          # 12-18 s single-speaker beds (long stream)
python train/data_prep/build_beds_large.py calibrate                                 # clean beds for the fusion mixtures,
python train/data_prep/build_beds_large.py download --shard-list train/data_prep/bed_shards.json   # exact shards and revisions
python train/data_prep/build_beds_large.py process --device cuda --workers 4
python train/data_prep/build_beds_large.py download --source tp --shard-list train/data_prep/bed_shards.json
python train/data_prep/build_beds_large.py process --source tp --device cuda --workers 4
python train/data_prep/build_beds_large.py finalize --hours 15 --hours-lang Hindi:35 --spk-cap-min 6
python train/data_prep/long_beds.py valset --out $D/valset_long --n 400              # evaluation only (needs beds_large)
python train/data_prep/make_fusion_mixtures.py --out $D/fresh2 --n 5000 --seed 1 --rate 0.65 --dur-median 13 --min-s 8 --max-s 18 --ivr-scan 60000
python train/data_prep/make_teacher.py --set val                                     # SraVaani teacher tokens
python train/data_prep/make_teacher.py --set $D/beds_long/manifest.jsonl
python train/data_prep/make_teacher.py --set $D/fresh2/clean_manifest.jsonl --name fresh2
bash scripts/download_pretrained.sh

# ---- 1. members (configs/*.yaml; held-out twins feed the fusion, *_full are shipped). mg_sn_a1 is only used up to
#      step 6000 (its 40k-step cosine schedule is kept for the learning-rate curve; it may be stopped after ema_006000).
for c in mg_sn_a1 mg_S mg_SE mg_SEN mg_LB mg_SE_full mg_SE2_full mg_SEN_full mg_LB_full \
         bsrnnxl_S bsrnnxl_SEN bsrnnxl_LB bsrnnxl_SE_full bsrnnxl_SEN30_full bsrnnxl_LB_full \
         loco_LS loco_MWm loco_MWc loco_MWch6; do
  python $T --config configs/$c.yaml
done
python train/soup.py --out runs/loco_soup.pt runs/loco_MWm/ema_004000.pt runs/loco_MWc/ema_004000.pt runs/loco_MWch6/ema_004000.pt

# ---- 2. held-out-protocol member outputs on the fusion's training sets (shifts = 4-shift TTA; fresh2: the 2,500 used)
TTA=0,41,83,124
for s in heldout valset_p2f fresh2; do
  mgs=0; [ $s = heldout ] && mgs=$TTA
  lim=0; [ $s = fresh2 ] && lim=2500
  python train/dump_members.py --model mossgan --ckpt runs/mg_LB/ema_004000.pt --in-dir $D/$s/noisy --out $D/$s/members/mg_LB$([ $s = heldout ] && echo _tta) --shifts $mgs --limit $lim
  python train/dump_members.py --model bsrnn_xl --ckpt runs/bsrnnxl_LB/ema_006000.pt --in-dir $D/$s/noisy --out $D/$s/members/bsrnnxl_LB_tta --shifts $TTA --limit $lim
  python train/dump_members.py --model loco --ckpt runs/loco_soup.pt --in-dir $D/$s/noisy --out $D/$s/members/loco_soup --limit $lim
done

# ---- 3. fusion, then the shipped checkpoints
python train/train_fusion.py --config configs/fusion.yaml --out runs/fusion
mkdir -p checkpoints
cp runs/mg_LB_full/ema_004000.pt checkpoints/mossformergan.pt
cp runs/bsrnnxl_LB_full/ema_006000.pt checkpoints/bsrnn_xl.pt
cp runs/loco_soup.pt checkpoints/tflocoformer_soup.pt
cp runs/fusion/step_003000.pt checkpoints/fusion.pt
