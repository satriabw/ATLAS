#!/bin/bash
# Joint gated multimodal training, Stage B (2026-07-10,
# plan = artifacts/docs/2026-07-10_joint_gated/plan.md). Designed to run detached.
set -u
cd /home/satria/Project/ATLAS

OUT=artifacts/experiments/2026-07-10_joint_gated
mkdir -p "$OUT"
echo "[$(date)] joint-gated pipeline start" | tee "$OUT/pipeline.log"

cd training
python train_joint_gated.py --tag s0 --seed 0 \
  > "../$OUT/joint_s0.log" 2>&1 || { echo "JOINT s0 FAILED" | tee -a "../$OUT/pipeline.log"; exit 1; }
grep -E "Ep |DONE tag|test APv|gate readout|CSV" "../$OUT/joint_s0.log" | tee -a "../$OUT/pipeline.log"
cp checkpoints/best_joint_s0_predictions.csv "../$OUT/joint_s0.csv"

# headline bootstrap: joint_s0 vs the whole-track traj anchor (same weights as the init)
cd evaluation
echo "--- joint_s0 vs traj anchor 0.802 ---" | tee -a "../../$OUT/pipeline.log"
python bootstrap_ci.py --csv "../../$OUT/joint_s0.csv" \
  --vs "../../artifacts/experiments/2026-07-09_gated_r2/traj_anchor.csv" 2>&1 | tail -3 | tee -a "../../$OUT/pipeline.log"
echo "--- single joint_s0 ---" | tee -a "../../$OUT/pipeline.log"
python bootstrap_ci.py --csv "../../$OUT/joint_s0.csv" 2>&1 | tail -3 | tee -a "../../$OUT/pipeline.log"

cd ../..
echo "[$(date)] PIPELINE DONE" | tee -a "$OUT/pipeline.log"
touch "$OUT/DONE"
