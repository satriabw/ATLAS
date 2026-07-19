#!/bin/bash
# Option A step 1 (2026-07-08): rebuild frames_r2.h5 + retrain fine-tuned r2plus1d.
# Target: reproduce the lost best_r2plus1d_r2_phase1 (test APv 0.659).
# Resume-safe: build shards append (existing keys skipped); rerun the script to resume.
set -u
cd /home/satria/Project/ATLAS

LOGDIR=artifacts/experiments/2026-07-08_r2_rebuild
mkdir -p "$LOGDIR"
echo "[$(date)] R2 rebuild pipeline start" | tee "$LOGDIR/pipeline.log"

# ---- 1. three parallel shard builds (one per zip), ~2-2.5h ----
python scripts/build_h5_r2.py --output data/raw/video/frames_r2_p1.h5 \
  --video-start 1 --video-end 40 --jpeg --labeled-only \
  > "$LOGDIR/build_p1.log" 2>&1 &
P1=$!
python scripts/build_h5_r2.py --output data/raw/video/frames_r2_p2.h5 \
  --video-start 41 --video-end 80 --jpeg --labeled-only \
  > "$LOGDIR/build_p2.log" 2>&1 &
P2=$!
python scripts/build_h5_r2.py --output data/raw/video/frames_r2_p3.h5 \
  --video-start 81 --video-end 120 --jpeg --labeled-only \
  > "$LOGDIR/build_p3.log" 2>&1 &
P3=$!

FAIL=0
wait $P1 || { echo "shard p1 FAILED" | tee -a "$LOGDIR/pipeline.log"; FAIL=1; }
wait $P2 || { echo "shard p2 FAILED" | tee -a "$LOGDIR/pipeline.log"; FAIL=1; }
wait $P3 || { echo "shard p3 FAILED" | tee -a "$LOGDIR/pipeline.log"; FAIL=1; }
[ $FAIL -ne 0 ] && { echo "ABORT: shard build failed, see build_p*.log" | tee -a "$LOGDIR/pipeline.log"; exit 1; }
echo "[$(date)] shard builds done" | tee -a "$LOGDIR/pipeline.log"

# ---- 2. merge into external-link master (shards must stay beside it) ----
python scripts/merge_h5.py \
  data/raw/video/frames_r2_p1.h5 data/raw/video/frames_r2_p2.h5 data/raw/video/frames_r2_p3.h5 \
  --output data/raw/video/frames_r2.h5 >> "$LOGDIR/pipeline.log" 2>&1 || exit 1

NKEYS=$(python -c "import h5py; print(len(h5py.File('data/raw/video/frames_r2.h5','r').keys()))")
echo "[$(date)] merged frames_r2.h5: $NKEYS keys (expected 7634)" | tee -a "$LOGDIR/pipeline.log"
[ "$NKEYS" -lt 7500 ] && { echo "ABORT: key count too low" | tee -a "$LOGDIR/pipeline.log"; exit 1; }

# ---- 3. retrain r2plus1d (phase-1 protocol: val-APv selection, unweighted CE,
#         color jitter, freeze early — all built into train.py vision mode) ----
cd training
python train.py --data_root /home/satria/Project/ATLAS \
  --mode vision --backbone r2plus1d --h5 frames_r2.h5 \
  --batch_size 4 --epochs 50 --patience 6 --seed 42 \
  --no_wandb --run_name r2plus1d_r2_rebuild \
  > "../$LOGDIR/train.log" 2>&1 || { echo "TRAIN FAILED" | tee -a "../$LOGDIR/pipeline.log"; exit 1; }
cp checkpoints/best_vision.pth checkpoints/best_r2plus1d_r2_rebuild.pth
echo "[$(date)] training done → best_r2plus1d_r2_rebuild.pth" | tee -a "../$LOGDIR/pipeline.log"

# ---- 4. test-set eval (writes best_r2plus1d_r2_rebuild_predictions.csv) ----
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python evaluation/evaluate_model.py \
  --checkpoint checkpoints/best_r2plus1d_r2_rebuild.pth --batch-size 4 \
  > "../$LOGDIR/eval.log" 2>&1 || { echo "EVAL FAILED" | tee -a "../$LOGDIR/pipeline.log"; exit 1; }
cp checkpoints/best_r2plus1d_r2_rebuild_predictions.csv "../$LOGDIR/" 2>/dev/null
grep -iE "APv|APn|mAP" "../$LOGDIR/eval.log" | tail -5 | tee -a "../$LOGDIR/pipeline.log"

echo "[$(date)] PIPELINE DONE" | tee -a "../$LOGDIR/pipeline.log"
touch "../$LOGDIR/DONE"
