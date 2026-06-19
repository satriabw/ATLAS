#!/usr/bin/env bash
# Centered ±32 fusion (2026-06-18): PooledFusedModel with BOTH branches on the
# centered ±32 / 64-slot window — the true apples-to-apples vs traj-centered (0.683)
# and vision-union-centered (0.558). Unlike run_fused_pooled.sh (whole-track traj +
# centered vision), the trajectory here comes from dataset/centered_window.py, so
# traj, vision AND fusion all share the same centre crop. Detached.
set -euo pipefail

ROOT=/home/satria/Project/ATLAS
EXP=$ROOT/artifacts/experiments/2026-06-18_centered_crop_vision
mkdir -p "$EXP"
cd "$ROOT/training"

EPOCHS=${EPOCHS:-40}
BATCH=${BATCH:-4}
PATIENCE=${PATIENCE:-5}

echo "########## FUSED_CENTERED START $(date '+%F %T')  epochs=$EPOCHS batch=$BATCH patience=$PATIENCE ##########"
echo "=== [$(date '+%F %T')] TRAIN fused_centered ==="
python train_centered_fused.py --h5 frames_union_centered.h5 --window 64 \
  --epochs "$EPOCHS" --batch_size "$BATCH" --patience "$PATIENCE" --no_wandb

echo "=== [$(date '+%F %T')] EVAL fused_centered (centered traj + centered vision) ==="
python evaluation/evaluate_centered.py --model-type fused_pooled \
  --checkpoint checkpoints/best_fused_centered.pth --window 64 --h5 frames_union_centered.h5

echo "########## FUSED_CENTERED DONE $(date '+%F %T') ##########"
