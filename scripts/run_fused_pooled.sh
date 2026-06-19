#!/usr/bin/env bash
# Pool-then-concat fused model (2026-06-18 feedback) — train + eval on the existing
# union centered-window h5 (no rebuild). Detached so it survives logout.
set -euo pipefail

ROOT=/home/satria/Project/ATLAS
EXP=$ROOT/artifacts/experiments/2026-06-18_centered_crop_vision
mkdir -p "$EXP"
cd "$ROOT/training"

EPOCHS=${EPOCHS:-40}
BATCH=${BATCH:-4}
PATIENCE=${PATIENCE:-5}

echo "########## FUSED_POOLED START $(date '+%F %T')  epochs=$EPOCHS batch=$BATCH patience=$PATIENCE ##########"
echo "=== [$(date '+%F %T')] TRAIN fused_pooled ==="
python train_fused_pooled.py --h5 frames_union_centered.h5 --num_frames 64 \
  --epochs "$EPOCHS" --batch_size "$BATCH" --patience "$PATIENCE" --no_wandb

echo "=== [$(date '+%F %T')] EVAL fused_pooled ==="
python evaluation/evaluate_model.py --model-type fused_pooled \
  --checkpoint checkpoints/best_fused_pooled.pth --num-frames 64

echo "########## FUSED_POOLED DONE $(date '+%F %T') ##########"
