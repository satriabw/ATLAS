# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All training/evaluation scripts must be run from the `training/` directory (they use relative imports):

```bash
# Train trajectory-only model (default mode)
cd training && python train.py --data_root /home/satria/Project/ATLAS --epochs 20 --batch_size 2

# Train vision-only model
cd training && python train.py --data_root /home/satria/Project/ATLAS --mode vision --epochs 20

# Train fused model (trajectory + vision; aux heads fight gradient starvation)
cd training && python train.py --data_root /home/satria/Project/ATLAS --mode fused --epochs 20

# Overfit on video_001 for sanity-checking
cd training && python train.py --data_root /home/satria/Project/ATLAS --overfit

# Evaluate on test set (even-numbered videos 2–120)
cd training && python evaluation/evaluate_model.py

# Evaluate fused model (h5 path auto-resolved from parquet-dir)
cd training && python evaluation/evaluate_model.py --model-type fused

# Evaluate in overfit mode (video_001, train_labels.pkl)
cd training && python evaluation/evaluate_model.py --overfit
```

Checkpoints are saved to `training/checkpoints/best_model.pth` (trajectory-only) and `best_fused.pth` (fused).

## Architecture

This is a pedestrian-vehicle traffic violation detection system. A "violation" means a vehicle failed to yield to a pedestrian at a crosswalk. The pipeline:

**Data flow**: Raw videos + labels → interaction parquet files → `ViolationDataset` → model

**Label encoding**: `annotation=0` = violation, `annotation=1` = compliance in training labels (pkl file). This is inverted from the evaluation event schema where `gt_label=0` = violation.

**Parquet schema** (`data/processed/interactions/*.parquet`): One row per vehicle-pedestrian pair per segment, grouped by `(v_track_id, roi)` for one interaction event. Array columns (`frames`, `v_loc_planar`, `p_loc_planar`, `v_speed`, `p_speed`) store object-dtype numpy arrays. `roi` is `'TOP'` or `'BOT'`.

**Dataset** (`training/dataset/`):
- `labels.py`: Parses label strings like `V001I00002S1D0R0A1` → `(video_id, tracking_id, roi, annotation)`
- `trajectory.py`: Builds per-event trajectory features from parquet groups. Vehicle feature: `(x_centered, y_centered, speed)`. Pedestrian feature: `(rel_x, rel_y, speed)` relative to the vehicle. Top-K pedestrians are selected by minimum average distance.
- `frames.py`: Loads video frames (OpenCV), resamples to `num_frames`, returns ImageNet-normalized tensor `(F, 3, H, W)`.
- `loader.py`: `load_violation_dataset()` is the main entry point — parses pkl, loads parquets, builds `ViolationDataset`.

**Models** (`training/models/`):
- `TrajectoryEncoder`: Bidirectional GRU. Input `(B, T, 3)` → output `(B, T, hidden_dim)`.
- `CrossAttentionModel`: Vehicle trajectory queries over top-K pedestrian encodings via `nn.MultiheadAttention`, then max-pools over time → 2-class classifier.
- `FusedModel`: Trajectory cross-attention plus a `VisionEncoder` (ResNet18 backbone). The classifier concatenates three pooled vectors — `h_traj` (motion), `h_vis` (appearance, a direct vision path), and `h_cross` (trajectory↔frame fusion attention). To counter vision-branch gradient starvation, it also exposes unimodal auxiliary heads (`traj_head`, `vis_head`) when `return_aux=True`; their losses are weighted by `--aux-w-traj` / `--aux-w-vision` in `train.py`. `self.ablate` (`'no_vision'` / `'no_traj'`) zeroes branches for ablation checks.
- `VisionEncoder`: ResNet18 without final layers, applied frame-by-frame `(B, F, C, H, W)` → `(B, F, output_dim)`.

**Training** (`training/train.py`):
- Scene-level train/val split (15% val by video ID, seed=42) to prevent video leakage.
- `CrossEntropyLoss` with class weights `[3.5, 1.0]` (violation vs compliance) to handle imbalance.
- Saves checkpoint on best val loss.

**Evaluation** (`training/evaluation/`):
- `inference.py`: `build_events_with_scores()` — collects labeled events from parquets, runs batched inference, attaches `score` (P(violation)) and `score_n` (P(compliance)).
- `ap_calculator.py`: `compute_map()` — computes APv (violation class) and APn (compliance class) using a sorted PR-curve integral. Predictions where the model is correct but `eiou <= threshold` are penalized (score zeroed).
- `evaluate_model.py`: CLI entry point combining inference + AP reporting.

## Data Quality

**Label counts (raw pkl files):**
- `train_labels.pkl`: 3,776 labels
- `test_labels.pkl`: 3,948 labels
- Total: 7,724 unique `(video_id, v_track_id, roi)` keys

**Why some labels are skipped at load time:**

Parquets only contain rows where a vehicle and pedestrian **co-occurred in the same frame** inside an ROI polygon. If no such co-occurrence happened, the label has no parquet entry and is silently skipped by `_assemble_labels` in `loader.py`.

After regenerating parquets (2026-06-02), ~88 compliance and 2 violation labels remain without parquet entries. The 2 known violation cases are documented in `data/known_label_issues.yaml`:
- `video_106 track_id=1316 TOP` — annotation error: the track is a pedestrian, not a vehicle
- `video_104 track_id=2038 BOT` — ROI annotation error: the vehicle had pedestrian interactions only in TOP, not BOT

**The Track2Data ROI mismatch bug (fixed 2026-06-02):**

The original `extract_pedestrian_vehicle_interactions.py` keyed the `pedestrian_interactions` dict by `p_track_id` alone. Vehicles that crossed both TOP and BOT ROIs were recorded under whichever ROI they first encountered a pedestrian — not the ROI where the labeled interaction occurred. This caused 31 violation labels to have parquets under the wrong ROI.

Fix: key changed to `(p_track_id, v_roi)` so vehicles crossing both ROIs produce separate records per ROI. Parquets were regenerated after this fix.

**H5 frame coverage:**
- `data/raw/video/frames.h5` has 11,908 keys (all parquet interaction groups, labeled + unlabeled)
- `load_violation_dataset(..., use_vision=True)` opens h5 at load time and filters out labels whose key is absent, logging a warning
- After the parquet fix, 30 newly-recovered violation cases have correct parquet entries but no h5 frames yet — they are skipped with a warning until h5 is rebuilt
- To rebuild h5: run `scripts/build_h5.py` with the raw video zip files available at `data/`

## Key Invariants

- Vehicle trajectory is centered at first position (`v_centered = v_loc - v_loc[0:1]`).
- Pedestrian trajectory is expressed relative to the vehicle (`p_rel = p_loc - v_loc`).
- Padding mask convention: `True` = padded (ignore), `False` = valid — matches PyTorch's `key_padding_mask` semantics.
- `score` in evaluation events = `softmax(logits)[:, 0]` = P(violation). `score_n = 1 - score`.
- Test split uses even-numbered videos (2, 4, …, 120); train split uses odd-numbered videos.

## Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```
