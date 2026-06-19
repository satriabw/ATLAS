#!/usr/bin/env bash
# Centered-window tight-crop vision experiment — full pipeline, both arms.
# Sequential (single GPU, shared spinning-disk master DB): build → train → eval
# for vehicle, then for ped. Designed to run detached (setsid+nohup) so it
# survives logout. See docs/2026-06-18_centered_crop_vision/plan.md.
set -euo pipefail

ROOT=/home/satria/Project/ATLAS
EXP=$ROOT/artifacts/experiments/2026-06-18_centered_crop_vision
mkdir -p "$EXP"
cd "$ROOT"

EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-4}

run_arm () {
  local arm=$1 h5=$2
  echo "=== [$(date '+%F %T')] BUILD $arm ($h5) ==="
  python scripts/build_h5_centered_crop.py --target "$arm" --labeled-only \
    --output "data/raw/video/$h5"

  echo "=== [$(date '+%F %T')] TRAIN $arm ==="
  ( cd training && python train_centered_vision.py \
      --h5 "$h5" --arm "$arm" --num_frames 64 --epochs "$EPOCHS" \
      --batch_size "$BATCH" --no_wandb )

  echo "=== [$(date '+%F %T')] EVAL $arm ==="
  ( cd training && python evaluation/evaluate_model.py \
      --model-type vision --checkpoint "checkpoints/best_vision_${arm}.pth" \
      --num-frames 64 )
  echo "=== [$(date '+%F %T')] DONE $arm ==="
}

echo "########## START $(date '+%F %T')  epochs=$EPOCHS batch=$BATCH ##########"
run_arm vehicle frames_vehicle_centered.h5
run_arm ped     frames_ped_centered.h5
echo "########## ALL DONE $(date '+%F %T') ##########"
