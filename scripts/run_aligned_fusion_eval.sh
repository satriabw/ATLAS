#!/usr/bin/env bash
# S2 gate eval for the aligned joint-fusion experiment (2026-06-19).
set -e
cd "$(dirname "$0")/../training"
C=checkpoints
E="python evaluation/evaluate_aligned_fusion.py"

echo "########## TEST APv + attention-validity ##########"
$E --checkpoint $C/best_aligned_fusion_s0.pth                 --out-csv $C/af_s0_full.csv
$E --checkpoint $C/best_aligned_fusion_s1.pth                 --out-csv $C/af_s1_full.csv
$E --checkpoint $C/best_aligned_fusion_s0.pth --ablate no_vision --out-csv $C/af_s0_no_vision.csv
$E --checkpoint $C/best_aligned_fusion_s0.pth --ablate no_traj   --out-csv $C/af_s0_no_traj.csv
$E --checkpoint $C/best_aligned_fusion_s0_shuffle.pth --shuffle-vision --out-csv $C/af_placebo.csv

echo "########## video-level bootstrap CI ##########"
B="python evaluation/bootstrap_ci.py"
echo "--- full s0 point + CI ---";        $B --csv $C/af_s0_full.csv
echo "--- full vs no_vision (paired) ---"; $B --csv $C/af_s0_full.csv --vs $C/af_s0_no_vision.csv
echo "--- full vs no_traj (paired) ---";   $B --csv $C/af_s0_full.csv --vs $C/af_s0_no_traj.csv
echo "--- full vs placebo (paired) ---";   $B --csv $C/af_s0_full.csv --vs $C/af_placebo.csv
echo "########## DONE ##########"
