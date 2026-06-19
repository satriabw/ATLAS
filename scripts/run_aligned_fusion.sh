#!/usr/bin/env bash
# S2 training arms for the aligned joint-fusion experiment (2026-06-19).
# Run from anywhere; cds into training/. Sequential (small models, one GPU).
#   full s0, full s1  → stability (>=2 seeds)
#   placebo (shuffle-vision) s0 → confound control
# Ablations (no_vision/no_traj) are EVAL-time on the full model, not trained here.
set -e
cd "$(dirname "$0")/../training"

echo "=== [1/3] full seed 0 ==="
python train_aligned_fusion.py --seed 0 --tag s0

echo "=== [2/3] full seed 1 ==="
python train_aligned_fusion.py --seed 1 --tag s1

echo "=== [3/3] placebo (shuffled vision) seed 0 ==="
python train_aligned_fusion.py --seed 0 --shuffle-vision --tag s0_shuffle

echo "=== ALL DONE ==="
