#!/usr/bin/env bash
# Gated fusion experiment (2026-07-08, plan = artifacts/docs/2026-07-08_gated_fusion/plan.md).
# Trains the 5 arms sequentially, then evaluates everything and runs the
# paired bootstrap comparisons. Designed to run detached (setsid nohup) —
# survives the session. All outputs land in artifacts/experiments/2026-07-08_gated_fusion/.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/artifacts/experiments/2026-07-08_gated_fusion"
mkdir -p "$OUT"
cd "$ROOT/training"

echo "=== [1/5] gated seed 0 ==="
python train_gated_fusion.py --seed 0 --tag s0

echo "=== [2/5] gated seed 1 ==="
python train_gated_fusion.py --seed 1 --tag s1

echo "=== [3/5] plain concat seed 0 ==="
python train_gated_fusion.py --seed 0 --no-gate --tag concat_s0

echo "=== [4/5] plain concat seed 1 ==="
python train_gated_fusion.py --seed 1 --no-gate --tag concat_s1

echo "=== [5/5] gated placebo (shuffled vision) seed 0 ==="
python train_gated_fusion.py --seed 0 --shuffle-vision --tag s0_shuffle

echo "=== EVAL: test APv + prediction CSVs ==="
python evaluation/evaluate_gated_fusion.py --checkpoint checkpoints/best_gated_s0.pth        --out-csv "$OUT/gated_s0.csv"
python evaluation/evaluate_gated_fusion.py --checkpoint checkpoints/best_gated_s1.pth        --out-csv "$OUT/gated_s1.csv"
python evaluation/evaluate_gated_fusion.py --checkpoint checkpoints/best_gated_concat_s0.pth --out-csv "$OUT/concat_s0.csv"
python evaluation/evaluate_gated_fusion.py --checkpoint checkpoints/best_gated_concat_s1.pth --out-csv "$OUT/concat_s1.csv"
python evaluation/evaluate_gated_fusion.py --checkpoint checkpoints/best_gated_s0_shuffle.pth --shuffle-vision --out-csv "$OUT/placebo_s0.csv"

echo "=== EVAL: zeroing ablations on gated s0 (secondary evidence) ==="
python evaluation/evaluate_gated_fusion.py --checkpoint checkpoints/best_gated_s0.pth --ablate no_vision --out-csv "$OUT/gated_s0_no_vision.csv"
python evaluation/evaluate_gated_fusion.py --checkpoint checkpoints/best_gated_s0.pth --ablate no_traj   --out-csv "$OUT/gated_s0_no_traj.csv"

echo "=== BOOTSTRAP: video-level CIs (paired deltas) ==="
cd evaluation
echo "--- gated s0 (single) ---";              python bootstrap_ci.py --csv "$OUT/gated_s0.csv"
echo "--- gated s0 vs concat s0 ---";          python bootstrap_ci.py --csv "$OUT/gated_s0.csv" --vs "$OUT/concat_s0.csv"
echo "--- gated s1 vs concat s1 ---";          python bootstrap_ci.py --csv "$OUT/gated_s1.csv" --vs "$OUT/concat_s1.csv"
echo "--- gated s0 vs placebo ---";            python bootstrap_ci.py --csv "$OUT/gated_s0.csv" --vs "$OUT/placebo_s0.csv"
echo "--- gated s0 vs no_vision (secondary) ---"; python bootstrap_ci.py --csv "$OUT/gated_s0.csv" --vs "$OUT/gated_s0_no_vision.csv"

echo "=== ALL DONE ==="
