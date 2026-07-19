#!/usr/bin/env bash
# Gated fusion on FROZEN cores (2026-07-08, plan =
# artifacts/docs/2026-07-08_gated_frozen/plan.md). 3 arms + evals + bootstraps
# against the ladder controls (r1_s0 = same model without gate, r1b_s0 = no
# vision). Designed to run detached (setsid nohup).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/artifacts/experiments/2026-07-08_gated_frozen"
LADDER="$ROOT/artifacts/experiments/2026-07-08_fusion_ladder"
mkdir -p "$OUT"
cd "$ROOT/training"
TRAJ_CKPT=checkpoints/best_traj_centered_w64.pth

echo "=== [1/3] gated-frozen s0 ==="
python train_fusion_ladder.py --seed 0 --tag gf_s0 --init-traj $TRAJ_CKPT --freeze-traj --gate

echo "=== [2/3] gated-frozen s1 ==="
python train_fusion_ladder.py --seed 1 --tag gf_s1 --init-traj $TRAJ_CKPT --freeze-traj --gate

echo "=== [3/3] gated-frozen placebo (shuffled vision) s0 ==="
python train_fusion_ladder.py --seed 0 --tag gf_s0_shuffle --init-traj $TRAJ_CKPT --freeze-traj --gate --shuffle-vision

echo "=== EVAL: test APv + gate readout + CSVs ==="
python evaluation/evaluate_gated_fusion.py --checkpoint checkpoints/best_ladder_gf_s0.pth                          --out-csv "$OUT/gf_s0.csv"
python evaluation/evaluate_gated_fusion.py --checkpoint checkpoints/best_ladder_gf_s1.pth                          --out-csv "$OUT/gf_s1.csv"
python evaluation/evaluate_gated_fusion.py --checkpoint checkpoints/best_ladder_gf_s0_shuffle.pth --shuffle-vision --out-csv "$OUT/gf_placebo.csv"

echo "=== BOOTSTRAP: pre-registered comparisons ==="
ANCHOR="$ROOT/training/checkpoints/best_traj_centered_w64_predictions.csv"
cd evaluation
echo "--- 2. gated-frozen s0 vs R1 s0 (does the gate help over plain concat, frozen) ---"
python bootstrap_ci.py --csv "$OUT/gf_s0.csv" --vs "$LADDER/r1_s0.csv"
echo "--- 3. gated-frozen s0 vs placebo ---"
python bootstrap_ci.py --csv "$OUT/gf_s0.csv" --vs "$OUT/gf_placebo.csv"
echo "--- 5a. gated-frozen s0 vs R1b (no-vision control) ---"
python bootstrap_ci.py --csv "$OUT/gf_s0.csv" --vs "$LADDER/r1b_s0.csv"
echo "--- 5b. gated-frozen s0 vs traj-only anchor ---"
python bootstrap_ci.py --csv "$OUT/gf_s0.csv" --vs "$ANCHOR"
echo "--- singles ---"
for t in gf_s0 gf_s1 gf_placebo; do
  echo "--- $t ---"; python bootstrap_ci.py --csv "$OUT/$t.csv"
done

echo "=== ALL DONE ==="
