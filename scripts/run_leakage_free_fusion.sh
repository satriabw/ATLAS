#!/usr/bin/env bash
# Leakage-Free Grounded Fusion — 2026-07-23 plan, step 3 (THE test).
# Treatment = frozen Kinetics-400 r2plus1d feats (r2_whole_feats_kinetics.h5).
# Paired control = fine-tuned r2plus1d feats, already run as gw_* (2026-07-09 gated_r2).
# Only variable changed vs control: --feats. Same gate head, frozen traj core, same crops.
set -euo pipefail
cd "$(dirname "$0")/../training"

OUT=../artifacts/experiments/2026-07-23_leakage_free_fusion
mkdir -p "$OUT"
CORE=checkpoints/best_traj_whole.pth          # traj anchor 0.8019 (frozen)
KFEATS=r2_whole_feats_kinetics.h5             # treatment: frozen Kinetics
ANCHOR=checkpoints/best_traj_whole_predictions.csv
CTRL=../artifacts/experiments/2026-07-09_gated_r2   # surviving fine-tuned gw_* CSVs

echo "===== TRAIN (frozen traj + gate head on precomputed feats — head-only, fast) ====="
for s in 0 1; do
  python train_fusion_ladder.py --gate --freeze-traj --bed whole \
    --init-traj $CORE --feats $KFEATS --seed $s --tag lf_kin_s${s} 2>&1 | tee "$OUT/train_lf_kin_s${s}.log"
done
# placebo: model trained on shuffled vision (seed 0)
python train_fusion_ladder.py --gate --freeze-traj --bed whole \
  --init-traj $CORE --feats $KFEATS --seed 0 --shuffle-vision --tag lf_kin_s0_shuf 2>&1 | tee "$OUT/train_lf_kin_s0_shuf.log"
# no-vision arm is feats-independent (frozen traj, vision zeroed) → reuse gw_s0_novis from gated_r2.

echo "===== EVAL → CSVs ====="
for tag in lf_kin_s0 lf_kin_s1 lf_kin_s0_shuf; do
  python evaluation/evaluate_gated_fusion.py --checkpoint checkpoints/best_ladder_${tag}.pth --out-csv "$OUT/${tag}.csv"
done

echo "===== BOOTSTRAP CI (paired, B=2000) ====="
{
  echo "### THE test: leakage-free treatment vs fine-tuned control (same bed, only --feats differs)"
  echo "--- lf_kin_s0 vs fine-tuned gw_s0 ---"; python evaluation/bootstrap_ci.py --csv "$OUT/lf_kin_s0.csv" --vs "$CTRL/gw_s0.csv"
  echo "--- lf_kin_s1 vs fine-tuned gw_s1 ---"; python evaluation/bootstrap_ci.py --csv "$OUT/lf_kin_s1.csv" --vs "$CTRL/gw_s1.csv"
  echo "### vs traj anchor 0.8019 (does it beat the pass bar?)"
  echo "--- lf_kin_s0 vs anchor ---"; python evaluation/bootstrap_ci.py --csv "$OUT/lf_kin_s0.csv" --vs "$ANCHOR"
  echo "--- lf_kin_s1 vs anchor ---"; python evaluation/bootstrap_ci.py --csv "$OUT/lf_kin_s1.csv" --vs "$ANCHOR"
  echo "### controls: real vision must beat its own placebo, and full must not sag below no-vision"
  echo "--- lf_kin_s0 vs placebo lf_kin_s0_shuf ---"; python evaluation/bootstrap_ci.py --csv "$OUT/lf_kin_s0.csv" --vs "$OUT/lf_kin_s0_shuf.csv"
  echo "--- lf_kin_s0 vs no-vision gw_s0_novis ---"; python evaluation/bootstrap_ci.py --csv "$OUT/lf_kin_s0.csv" --vs "$CTRL/gw_s0_novis.csv"
} 2>&1 | tee "$OUT/bootstrap.log"

touch "$OUT/DONE"
echo "ALL DONE → $OUT"
