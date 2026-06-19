#!/usr/bin/env bash
# R1 fallback: max-pool variant of the aligned joint-fusion model (2026-06-19).
# Tests whether dropping the temporal-attention selector (which failed its
# explainability check and may be the lossy head) recovers the operating point
# while KEEPING the vision contribution (vs no_vision and vs placebo).
set -e
cd "$(dirname "$0")/../training"
C=checkpoints

echo "########## TRAIN (max-pool) ##########"
python train_aligned_fusion.py --seed 0 --pool max --tag s0_max
python train_aligned_fusion.py --seed 1 --pool max --tag s1_max
python train_aligned_fusion.py --seed 0 --pool max --shuffle-vision --tag s0_shuffle_max

echo "########## EVAL (max-pool) ##########"
E="python evaluation/evaluate_aligned_fusion.py"
$E --checkpoint $C/best_aligned_fusion_s0_max.pth                 --out-csv $C/af_s0_full_max.csv
$E --checkpoint $C/best_aligned_fusion_s1_max.pth                 --out-csv $C/af_s1_full_max.csv
$E --checkpoint $C/best_aligned_fusion_s0_max.pth --ablate no_vision --out-csv $C/af_s0_no_vision_max.csv
$E --checkpoint $C/best_aligned_fusion_s0_max.pth --ablate no_traj   --out-csv $C/af_s0_no_traj_max.csv
$E --checkpoint $C/best_aligned_fusion_s0_shuffle_max.pth --shuffle-vision --out-csv $C/af_placebo_max.csv

echo "########## BOOTSTRAP CI (max-pool) ##########"
B="python evaluation/bootstrap_ci.py"
echo "--- full s0_max point + CI ---";       $B --csv $C/af_s0_full_max.csv
echo "--- full vs no_vision (paired) ---";   $B --csv $C/af_s0_full_max.csv --vs $C/af_s0_no_vision_max.csv
echo "--- full vs no_traj (paired) ---";     $B --csv $C/af_s0_full_max.csv --vs $C/af_s0_no_traj_max.csv
echo "--- full vs placebo (paired) ---";     $B --csv $C/af_s0_full_max.csv --vs $C/af_placebo_max.csv
echo "########## DONE ##########"
