#!/bin/bash
# Gated fusion on frozen cores w/ rebuilt r2plus1d feats, both beds (2026-07-09,
# plan = artifacts/docs/2026-07-09_gated_r2/plan.md). Designed to run detached.
set -u
cd /home/satria/Project/ATLAS

OUT=artifacts/experiments/2026-07-09_gated_r2
GF=artifacts/experiments/2026-07-08_gated_frozen
mkdir -p "$OUT"
echo "[$(date)] gated-r2 pipeline start" | tee "$OUT/pipeline.log"

# ---- 1. retrain the whole-track traj core (checkpoint lost in the 06-19
#         cleanup; previous repro = test APv 0.8019). Anchor CSV comes from
#         THIS checkpoint, so anchor and frozen core are the same weights. ----
cd training
python train.py --data_root /home/satria/Project/ATLAS \
  --epochs 60 --patience 10 --seed 42 --no_wandb --no_notify \
  --run_name traj_whole_rebuild \
  > "../$OUT/traj_train.log" 2>&1 || { echo "TRAJ TRAIN FAILED" | tee -a "../$OUT/pipeline.log"; exit 1; }
cp checkpoints/best_model.pth checkpoints/best_traj_whole.pth
python evaluation/evaluate_model.py --checkpoint checkpoints/best_traj_whole.pth \
  > "../$OUT/traj_eval.log" 2>&1 || { echo "TRAJ EVAL FAILED" | tee -a "../$OUT/pipeline.log"; exit 1; }
cp checkpoints/best_traj_whole_predictions.csv "../$OUT/traj_anchor.csv"
echo "[$(date)] traj core rebuilt:" | tee -a "../$OUT/pipeline.log"
grep -E "APv|APn|mAP" "../$OUT/traj_eval.log" | tee -a "../$OUT/pipeline.log"
cd ..

# ---- 2. frozen pooled r2plus1d features, both beds ----
python scripts/precompute_r2plus1d_feats.py --bed whole \
  --out data/raw/video/r2_whole_feats.h5 \
  > "$OUT/feats_whole.log" 2>&1 || { echo "FEATS WHOLE FAILED" | tee -a "$OUT/pipeline.log"; exit 1; }
python scripts/precompute_r2plus1d_feats.py --bed centered \
  --out data/raw/video/r2_centered_feats.h5 \
  > "$OUT/feats_centered.log" 2>&1 || { echo "FEATS CENTERED FAILED" | tee -a "$OUT/pipeline.log"; exit 1; }
echo "[$(date)] feats done" | tee -a "$OUT/pipeline.log"

# ---- 3. head-only gated arms (frozen traj core + frozen vision feats) ----
cd training
run_arm () {  # tag, extra args...
  local tag=$1; shift
  echo "=== train $tag ===" | tee -a "../$OUT/pipeline.log"
  python train_fusion_ladder.py --gate --freeze-traj --tag "$tag" "$@" \
    >> "../$OUT/train_arms.log" 2>&1 || { echo "TRAIN $tag FAILED" | tee -a "../$OUT/pipeline.log"; exit 1; }
}
WCORE=checkpoints/best_traj_whole.pth
CCORE=checkpoints/best_traj_centered_w64.pth

run_arm gw_s0       --bed whole --seed 0 --init-traj $WCORE
run_arm gw_s1       --bed whole --seed 1 --init-traj $WCORE
run_arm gw_s0_shuf  --bed whole --seed 0 --init-traj $WCORE --shuffle-vision
run_arm gw_s0_novis --bed whole --seed 0 --init-traj $WCORE --no-vision

run_arm gc_s0       --bed centered --feats r2_centered_feats.h5 --seed 0 --init-traj $CCORE
run_arm gc_s1       --bed centered --feats r2_centered_feats.h5 --seed 1 --init-traj $CCORE
run_arm gc_s0_shuf  --bed centered --feats r2_centered_feats.h5 --seed 0 --init-traj $CCORE --shuffle-vision
run_arm gc_s0_novis --bed centered --feats r2_centered_feats.h5 --seed 0 --init-traj $CCORE --no-vision
echo "[$(date)] arms trained" | tee -a "../$OUT/pipeline.log"

# ---- 4. test evals (APv + gate readout + CSVs) ----
eval_arm () {  # tag, extra args...
  local tag=$1; shift
  python evaluation/evaluate_gated_fusion.py --checkpoint "checkpoints/best_ladder_$tag.pth" \
    --out-csv "../$OUT/$tag.csv" "$@" \
    >> "../$OUT/eval.log" 2>&1 || { echo "EVAL $tag FAILED" | tee -a "../$OUT/pipeline.log"; exit 1; }
}
eval_arm gw_s0
eval_arm gw_s1
eval_arm gw_s0_shuf  --shuffle-vision
eval_arm gw_s0_novis --ablate no_vision
eval_arm gc_s0
eval_arm gc_s1
eval_arm gc_s0_shuf  --shuffle-vision
eval_arm gc_s0_novis --ablate no_vision
grep -E "test APv|gate readout" "../$OUT/eval.log" | tee -a "../$OUT/pipeline.log"

# ---- 5. pre-registered video-level bootstraps ----
cd evaluation
boot () { echo "--- $1 ---" | tee -a "../../$OUT/pipeline.log"; shift
  python bootstrap_ci.py "$@" 2>&1 | tail -3 | tee -a "../../$OUT/pipeline.log"; }

boot "W1 gw_s0 vs whole traj anchor"    --csv "../../$OUT/gw_s0.csv" --vs "../../$OUT/traj_anchor.csv"
boot "W2 gw_s0 vs placebo"              --csv "../../$OUT/gw_s0.csv" --vs "../../$OUT/gw_s0_shuf.csv"
boot "W3 gw_s0 vs no-vision control"    --csv "../../$OUT/gw_s0.csv" --vs "../../$OUT/gw_s0_novis.csv"
boot "C1 gc_s0 vs gf_s0 (resnet feats)" --csv "../../$OUT/gc_s0.csv" --vs "../../$GF/gf_s0.csv"
boot "C2 gc_s0 vs placebo"              --csv "../../$OUT/gc_s0.csv" --vs "../../$OUT/gc_s0_shuf.csv"
boot "C3 gc_s0 vs no-vision control"    --csv "../../$OUT/gc_s0.csv" --vs "../../$OUT/gc_s0_novis.csv"
boot "C4 gc_s0 vs centered traj anchor" --csv "../../$OUT/gc_s0.csv" --vs "../checkpoints/best_traj_centered_w64_predictions.csv"
for t in gw_s0 gw_s1 gw_s0_shuf gw_s0_novis gc_s0 gc_s1 gc_s0_shuf gc_s0_novis; do
  boot "single $t" --csv "../../$OUT/$t.csv"
done

cd ../..
echo "[$(date)] PIPELINE DONE" | tee -a "$OUT/pipeline.log"
touch "$OUT/DONE"
