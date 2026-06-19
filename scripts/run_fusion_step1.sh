#!/usr/bin/env bash
# Step 1 — frozen-branch fusion head (2026-06-18). Retrain the standalone centered
# branches (all prior standalone ckpts were deleted in cleanup), then test whether a
# combiner over FROZEN branches beats traj-centered / centered log-odds. Detached.
set -euo pipefail

ROOT=/home/satria/Project/ATLAS
EXP=$ROOT/artifacts/experiments/2026-06-18_fusion_step1_step2
mkdir -p "$EXP"
cd "$ROOT/training"

echo "########## STEP1 START $(date '+%F %T') ##########"

echo "=== [$(date '+%F %T')] (1/3) train traj-centered w64 ==="
python train_centered.py --window 64 --epochs 50 --patience 8 --no_wandb --no_notify

echo "=== [$(date '+%F %T')] (2/3) train vision-centered union (r2plus1d RGB) ==="
python train_centered_vision.py --h5 frames_union_centered.h5 --arm union \
  --num_frames 64 --epochs 30 --batch_size 4 --patience 8 --no_wandb --no_notify

echo "=== [$(date '+%F %T')] (3/3) frozen fusion head ==="
python fuse_frozen_heads.py --window 64 --h5 frames_union_centered.h5 \
  --traj-ckpt checkpoints/best_traj_centered_w64.pth \
  --vis-ckpt checkpoints/best_vision_union.pth

echo "########## STEP1 DONE $(date '+%F %T') ##########"
