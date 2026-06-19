#!/usr/bin/env bash
# Union-framing centered-window vision arm (2026-06-18 follow-up): isolates the
# WINDOW variable — union crop (vehicle ∪ top-1 ped, event-static) over the fixed
# ±32 centered window, vs the 0.659 anchor's whole-track linspace. Same protocol
# as the vehicle/ped tight arms (r2plus1d, 64 frames, size 112). Detached.
set -euo pipefail

ROOT=/home/satria/Project/ATLAS
EXP=$ROOT/artifacts/experiments/2026-06-18_centered_crop_vision
mkdir -p "$EXP"
cd "$ROOT"

EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-4}
H5=frames_union_centered.h5

echo "########## UNION START $(date '+%F %T')  epochs=$EPOCHS batch=$BATCH ##########"
echo "=== [$(date '+%F %T')] BUILD union ($H5) ==="
python scripts/build_h5_centered_crop.py --target union --labeled-only \
  --output "data/raw/video/$H5"

echo "=== [$(date '+%F %T')] TRAIN union ==="
( cd training && python train_centered_vision.py \
    --h5 "$H5" --arm union --num_frames 64 --epochs "$EPOCHS" \
    --batch_size "$BATCH" --no_wandb )

echo "=== [$(date '+%F %T')] EVAL union ==="
( cd training && python evaluation/evaluate_model.py \
    --model-type vision --checkpoint checkpoints/best_vision_union.pth --num-frames 64 )

echo "########## UNION DONE $(date '+%F %T') ##########"
