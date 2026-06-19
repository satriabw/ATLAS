#!/usr/bin/env bash
# Step 2 — grounded centered vision (2026-06-18). Waits for Step 1 to finish (shared
# single GPU), then trains grounded vs ungrounded centered vision (same resnet18 arch,
# masks the only difference), evals both, and runs complementarity vs traj-centered.
set -euo pipefail

ROOT=/home/satria/Project/ATLAS
EXP=$ROOT/artifacts/experiments/2026-06-18_fusion_step1_step2
STEP1_LOG=$EXP/run_step1.log
cd "$ROOT/training"

echo "########## STEP2 START $(date '+%F %T') — waiting for Step 1 ##########"
while ! grep -q "STEP1 DONE" "$STEP1_LOG" 2>/dev/null; do sleep 30; done
echo "=== [$(date '+%F %T')] Step 1 done — beginning Step 2 ==="

echo "=== [$(date '+%F %T')] (1/5) train GROUNDED vision (resnet18, masks) ==="
python train_centered_vision_grounded.py --arm grounded --window 64 \
  --epochs 30 --batch_size 4 --patience 8 --no_wandb --no_notify

echo "=== [$(date '+%F %T')] (2/5) train UNGROUNDED control (resnet18, zeroed masks) ==="
python train_centered_vision_grounded.py --arm ungrounded --zero_masks --window 64 \
  --epochs 30 --batch_size 4 --patience 8 --no_wandb --no_notify

echo "=== [$(date '+%F %T')] (3/5) eval grounded ==="
python evaluation/evaluate_centered.py --model-type vision --window 64 \
  --h5 frames_union_centered.h5 --ground \
  --checkpoint checkpoints/best_vision_grounded.pth \
  --out-csv checkpoints/best_vision_grounded_predictions.csv

echo "=== [$(date '+%F %T')] (4/5) eval ungrounded control ==="
python evaluation/evaluate_centered.py --model-type vision --window 64 \
  --h5 frames_union_centered.h5 --ground --zero-masks \
  --checkpoint checkpoints/best_vision_ungrounded.pth \
  --out-csv checkpoints/best_vision_ungrounded_predictions.csv

echo "=== [$(date '+%F %T')] (5/5) traj CSV + complementarity (grounded vision vs traj) ==="
python evaluation/evaluate_centered.py --model-type cross_attention --window 64 \
  --checkpoint checkpoints/best_traj_centered_w64.pth \
  --out-csv checkpoints/best_traj_centered_w64_predictions.csv
python evaluation/complementarity.py \
  --traj-csv checkpoints/best_traj_centered_w64_predictions.csv \
  --vision-csv checkpoints/best_vision_grounded_predictions.csv

echo "########## STEP2 DONE $(date '+%F %T') ##########"
