# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All scripts are run from the `training/` directory (it uses relative imports):

```bash
cd training/

# Train trajectory-only model
python train.py --data_root /path/to/ATLAS --epochs 20 --batch_size 2

# Train with vision branch (ResNet18 + trajectory fusion)
python train.py --data_root /path/to/ATLAS --use_vision

# Overfit sanity check on video_001
python train.py --data_root /path/to/ATLAS --overfit

# Evaluate (trajectory-only)
python evaluation/evaluate_model.py \
    --checkpoint checkpoints/best_model.pth \
    --parquet-dir /path/to/ATLAS/data/processed/interactions \
    --labels-pkl  /path/to/ATLAS/data/raw/labels/test_labels.pkl

# Evaluate (fused model with video)
python evaluation/evaluate_model.py \
    --checkpoint checkpoints/best_fused.pth \
    --model-type fused \
    --video-dir  /path/to/ATLAS/data/raw/video \
    --parquet-dir /path/to/ATLAS/data/processed/interactions \
    --labels-pkl  /path/to/ATLAS/data/raw/labels/test_labels.pkl
```

Key training flags: `--top_k` (pedestrians per event, default 5), `--lr`, `--batch_size`. Checkpoints saved to `training/checkpoints/best_model.pth`.

## Architecture

ATLAS detects traffic violations by classifying vehicle–pedestrian interaction events using trajectory and optionally video data.

### Data pipeline

**Input data layout** (expected under `--data_root`):
```
data/raw/labels/train_labels.pkl   # pickled (label_strings, _) list
data/raw/labels/test_labels.pkl
data/processed/interactions/video_NNN_interactions.parquet   # one per video
data/raw/video/video_NNN.avi       # only needed with --use_vision
```

**Label format**: strings like `V001I00002S1D0R0A1` → parsed to `(video_id, tracking_id, roi, annotation)`. Annotation `0`=violation, `1`=compliance.

**Parquet schema**: each row is a vehicle–pedestrian pair with columns `v_track_id`, `roi` (`BOT`/`TOP`), `p_track_id`, `frames`, `v_loc_planar`, `v_speed`, `p_loc_planar`, `p_speed`, `d_min`.

`load_violation_dataset()` in `dataset/violation_dataset.py`:
1. Parses `.pkl` label file → list of `(video_id, tracking_id, roi, annotation)`.
2. Loads parquet files and builds trajectory cache: groups by `(v_track_id, roi)`, selects top-K closest pedestrians by mean `d_min`, resamples trajectories to `num_frames`.
3. Features are relative/centered: vehicle trajectory is centered at its first position; pedestrian features are relative to the vehicle.
4. Trajectories resampled or zero-padded to `num_frames`; padding tracked via boolean masks (True = padded, should be ignored).

Train/val split is **scene-stratified by video**: all events from a given video go to the same split (85/15 by video count, seed 42).

### Models (`training/models/`)

**`TrajectoryEncoder`**: Linear(3→64) + ReLU → bidirectional GRU (hidden 128, split 64×2). Input shape `(B, T, 3)`, output `(B, T, 128)`.

**`CrossAttentionModel`** (trajectory only):
1. Encodes vehicle trajectory → `(B, T_v, H)`.
2. Encodes each pedestrian trajectory independently (reshaped to `(B*K, T_p, 3)` to avoid GRU hidden state bleeding across pedestrians), max-pools to `(B, K, H)`.
3. Cross-attention: vehicle queries pedestrian encodings.
4. Max-pool over vehicle timesteps → MLP classifier.

**`FusedModel`** (trajectory + vision):
- Adds `VisionEncoder` (ResNet18 backbone, AdaptiveAvgPool, Linear projection).
- Step 1: same vehicle-queries-pedestrian cross-attention → `traj_context (B, T_v, H)`.
- Step 2: vision frame features projected to `H`, then vision queries trajectory context via a second cross-attention.
- Step 3: max-pool both `traj_context` and `fused` outputs, sum them, classify.

### Evaluation (`training/evaluation/`)

**`ap_calculator.py`**: Implements `compute_ap` and `compute_map`. APv uses `score=P(violation)` (class 0); APn uses `score_n=P(compliance)` (class 1). Predictions with correct class but `eiou ≤ threshold` have scores zeroed.

**`evaluate_model.py`**: Loads checkpoint, runs batched inference over GT-labeled interactions from parquet files, computes APv / APn / mAP.

### Class imbalance

Training uses weighted CrossEntropyLoss with `weights=[3.5, 1.0]` (violation upweighted). LR scheduler: `ReduceLROnPlateau(factor=0.5, patience=10)`.
