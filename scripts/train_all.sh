#!/bin/bash
# The full training pipeline that produced the shipped checkpoints, in order. Each stage resumes nothing: run the
# sections you need. Times are for one A100 80 GB. Set TRACK2_DATA (default ./data) and HF_TOKEN (gated datasets).
set -e
cd "$(dirname "$0")/.."
export TRACK2_DATA=${TRACK2_DATA:-data}
D=$TRACK2_DATA
T=train/train_member.py

# ---- 0. data (README "Data layout"); the organiser validation set must be in $D/validation
python train/prepare_noisebank.py --val-root $D/validation --out $D/noisebank        # noise events from the validation pairs
python train/data_prep/make_heldout.py --out $D/heldout                              # the 209 held-out pairs
python train/data_prep/ingest_vaani_tp.py --root $D/vaani_tp                         # Vaani transcription release, exclusions applied
python train/data_prep/long_beds.py beds --out $D/beds_long                          # 12-18 s single-speaker beds (long stream)
python train/data_prep/long_beds.py valset --out $D/valset_long --n 400              # 360-clip test-length evaluation set
python train/data_prep/build_beds_large.py calibrate                                 # clean beds for the fusion mixtures:
python train/data_prep/build_beds_large.py download --shards 30 --langs Hindi:80,Maithili:45,Nepali:42,Tamil:40,Marathi:38,Konkani:36
python train/data_prep/build_beds_large.py process --device cuda --workers 4
python train/data_prep/build_beds_large.py download --source tp --langs Chhattisgarhi:99,Magadhi:99,Magahi:99,Bhojpuri:99
python train/data_prep/build_beds_large.py process --source tp --device cuda --workers 4
python train/data_prep/build_beds_large.py finalize --hours 15 --hours-lang Hindi:35 --spk-cap-min 6
python train/data_prep/make_valset.py --out $D/valset_p2f --n 1200                   # simulated Phase-2-shaped clips
python train/data_prep/make_fusion_mixtures.py --out $D/fresh2 --n 5000 --seed 1 --rate 0.65 --dur-median 13 --min-s 8 --max-s 18 --ivr-scan 60000
python train/data_prep/make_teacher.py --set val                                     # SraVaani teacher tokens
python train/data_prep/make_teacher.py --set $D/beds_long/manifest.jsonl
python train/data_prep/make_teacher.py --set $D/fresh2/clean_manifest.jsonl --name fresh2
bash scripts/download_pretrained.sh

# ---- 1. members (configs/*.yaml; held-out twins feed the fusion, *_full are shipped)
for c in mg_sn_a1 mg_S mg_SE mg_SEN mg_LB mg_SE_full mg_SE2_full mg_SEN_full mg_LB_full \
         bsrnnxl_S bsrnnxl_SEN bsrnnxl_LB bsrnnxl_SE_full bsrnnxl_SEN30_full bsrnnxl_LB_full \
         loco_LS loco_MWm loco_MWc loco_MWch6; do
  python $T --config configs/$c.yaml
done
python train/soup.py --out runs/loco_soup.pt runs/loco_MWm/ema_004000.pt runs/loco_MWc/ema_004000.pt runs/loco_MWch6/ema_004000.pt

# ---- 2. held-out-protocol member outputs on the fusion's training sets (shifts = 4-shift TTA)
TTA=0,41,83,124
for s in heldout valset_p2f fresh2; do
  mgs=0; [ $s = heldout ] && mgs=$TTA
  python train/dump_members.py --model mossgan --ckpt runs/mg_LB/ema_004000.pt --in-dir $D/$s/noisy --out $D/$s/members/mg_LB$([ $s = heldout ] && echo _tta) --shifts $mgs
  python train/dump_members.py --model bsrnn_xl --ckpt runs/bsrnnxl_LB/ema_006000.pt --in-dir $D/$s/noisy --out $D/$s/members/bsrnnxl_LB_tta --shifts $TTA
  python train/dump_members.py --model loco --ckpt runs/loco_soup.pt --in-dir $D/$s/noisy --out $D/$s/members/loco_soup
done

# ---- 3. fusion, then the shipped checkpoints
python train/train_fusion.py --config configs/fusion.yaml --out runs/fusion
mkdir -p checkpoints
cp runs/mg_LB_full/ema_004000.pt checkpoints/mossformergan.pt
cp runs/bsrnnxl_LB_full/ema_006000.pt checkpoints/bsrnn_xl.pt
cp runs/loco_soup.pt checkpoints/tflocoformer_soup.pt
cp runs/fusion/step_003000.pt checkpoints/fusion.pt
