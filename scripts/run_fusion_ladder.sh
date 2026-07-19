#!/usr/bin/env bash
# Fusion bisection ladder (2026-07-08, plan =
# artifacts/docs/2026-07-08_fusion_failure_investigation/plan.md).
# 7 rungs, sequential, then evals + pre-registered paired bootstraps.
# Designed to run detached (setsid nohup). Outputs land in
# artifacts/experiments/2026-07-08_fusion_ladder/.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/artifacts/experiments/2026-07-08_fusion_ladder"
mkdir -p "$OUT"
cd "$ROOT/training"
TRAJ_CKPT=checkpoints/best_traj_centered_w64.pth
GATED_CKPT=checkpoints/best_gated_s0.pth

echo "=== [1/7] R1b: frozen 0.681 core, no vision, head only (protocol sanity) ==="
python train_fusion_ladder.py --seed 0 --tag r1b_s0 --init-traj $TRAJ_CKPT --freeze-traj --no-vision

echo "=== [2/7] R1 s0: frozen 0.681 core + frozen vision, head only (floor test) ==="
python train_fusion_ladder.py --seed 0 --tag r1_s0 --init-traj $TRAJ_CKPT --freeze-traj

echo "=== [3/7] R1 s1 (stability) ==="
python train_fusion_ladder.py --seed 1 --tag r1_s1 --init-traj $TRAJ_CKPT --freeze-traj

echo "=== [4/7] R1p: frozen core + SHUFFLED vision (placebo) ==="
python train_fusion_ladder.py --seed 0 --tag r1p_s0 --init-traj $TRAJ_CKPT --freeze-traj --shuffle-vision

echo "=== [5/7] R1g: frozen core from JOINTLY-TRAINED gated s0, no vision (damage probe) ==="
python train_fusion_ladder.py --seed 0 --tag r1g_s0 --init-traj $GATED_CKPT --freeze-traj --no-vision

echo "=== [6/7] R2 s0: 0.681 core unfrozen at lr 1e-5, vision on ==="
python train_fusion_ladder.py --seed 0 --tag r2_s0 --init-traj $TRAJ_CKPT --traj-lr 1e-5

echo "=== [7/7] R2 s1 (stability) ==="
python train_fusion_ladder.py --seed 1 --tag r2_s1 --init-traj $TRAJ_CKPT --traj-lr 1e-5

echo "=== EVAL: test APv + prediction CSVs ==="
python evaluation/evaluate_gated_fusion.py --checkpoint checkpoints/best_ladder_r1b_s0.pth --ablate no_vision --out-csv "$OUT/r1b_s0.csv"
python evaluation/evaluate_gated_fusion.py --checkpoint checkpoints/best_ladder_r1_s0.pth                     --out-csv "$OUT/r1_s0.csv"
python evaluation/evaluate_gated_fusion.py --checkpoint checkpoints/best_ladder_r1_s1.pth                     --out-csv "$OUT/r1_s1.csv"
python evaluation/evaluate_gated_fusion.py --checkpoint checkpoints/best_ladder_r1p_s0.pth --shuffle-vision   --out-csv "$OUT/r1p_s0.csv"
python evaluation/evaluate_gated_fusion.py --checkpoint checkpoints/best_ladder_r1g_s0.pth --ablate no_vision --out-csv "$OUT/r1g_s0.csv"
python evaluation/evaluate_gated_fusion.py --checkpoint checkpoints/best_ladder_r2_s0.pth                     --out-csv "$OUT/r2_s0.csv"
python evaluation/evaluate_gated_fusion.py --checkpoint checkpoints/best_ladder_r2_s1.pth                     --out-csv "$OUT/r2_s1.csv"

echo "=== BOOTSTRAP: pre-registered comparisons (plan.md readout A-D) ==="
ANCHOR="$ROOT/training/checkpoints/best_traj_centered_w64_predictions.csv"
cd evaluation
echo "--- A. R1b vs traj-only anchor (protocol sanity) ---"; python bootstrap_ci.py --csv "$OUT/r1b_s0.csv" --vs "$ANCHOR"
echo "--- B1. R1 s0 vs R1b (marginal value of vision, frozen cores) ---"; python bootstrap_ci.py --csv "$OUT/r1_s0.csv" --vs "$OUT/r1b_s0.csv"
echo "--- B2. R1 s0 vs placebo ---"; python bootstrap_ci.py --csv "$OUT/r1_s0.csv" --vs "$OUT/r1p_s0.csv"
echo "--- B3. R1 s0 vs traj-only anchor ---"; python bootstrap_ci.py --csv "$OUT/r1_s0.csv" --vs "$ANCHOR"
echo "--- C. R1g vs R1b (joint-training damage probe) ---"; python bootstrap_ci.py --csv "$OUT/r1g_s0.csv" --vs "$OUT/r1b_s0.csv"
echo "--- D1. R2 s0 vs R1 s0 (does gentle end-to-end hold the floor) ---"; python bootstrap_ci.py --csv "$OUT/r2_s0.csv" --vs "$OUT/r1_s0.csv"
echo "--- D2. R2 s0 vs traj-only anchor ---"; python bootstrap_ci.py --csv "$OUT/r2_s0.csv" --vs "$ANCHOR"
echo "--- singles ---"
for t in r1b_s0 r1_s0 r1_s1 r1p_s0 r1g_s0 r2_s0 r2_s1; do
  echo "--- $t ---"; python bootstrap_ci.py --csv "$OUT/$t.csv"
done

echo "=== ALL DONE ==="
