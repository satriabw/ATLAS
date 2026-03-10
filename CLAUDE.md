# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ATLAS classifies traffic interaction events (vehicle-pedestrian) as **violation** or **compliance** using trajectory data only. The current branch (`f-trajectory-only`) uses a GRU encoder — video frames have been removed.

## Common Commands

All training/evaluation commands run from `training/` directory:

```bash
# Standard training (odd videos 1,3,...,119 for train; even 2,4,...,120 for val)
python train.py --data_root /home/satria/Project/ATLAS --epochs 20 --batch_size 2 --lr 1e-4

# Overfit sanity check (single video, train==val — model should reach ~100% accuracy)
python train.py --data_root /home/satria/Project/ATLAS --overfit --videos 1 --epochs 20

# Baseline evaluation (no model, score=1.0 — expected mAP ≈ 0.5)
python -m evaluation.run_evaluation --video-ids 2 4 6 8 10 --eiou-threshold 0.5

# Model evaluation (even videos = test split: 2,4,...,120)
python evaluation/evaluate_model.py --checkpoint checkpoints/best_model.pth
```

Default paths for evaluation: `data/processed/interactions/` for parquets, `data/raw/labels/train_labels.pkl` for labels.

## Architecture

### Data Flow

1. **Labels** (`data/raw/labels/train_labels.pkl`) — `(label_strings, annotations)` tuple. Label string `V001I00002S1D0R0A1` parses to `video=001, track=2, roi=BOT (S1), annotation=1 (compliance)`.
2. **Parquet files** (`data/processed/interactions/video_NNN_interactions.parquet`) — one row per vehicle-pedestrian pair per frame. Grouped by `(v_track_id, roi)` to form one interaction event.
3. **`ViolationDataset`** — builds a 7-feature trajectory per event: `[v_loc_x, v_loc_y, v_speed, p_loc_x, p_loc_y, p_speed, v_p_distance]`. All sequences resampled to 32 frames.

### Model (`training/models/`)

- **`TrajectoryEncoder`** — 2-layer GRU (input=7, hidden=128, dropout=0.3) → outputs last hidden state `(B, 128)`
- **`TrajectoryOnlyModel`** — encoder → Linear(128→64)+ReLU+Dropout → Linear(64→2) → logits for `[violation, compliance]`

### Training (`training/train.py`)

- Weighted CrossEntropyLoss (inverse class frequency), Adam optimizer
- Gradient accumulation over 4 steps (effective batch size = 8), AMP mixed precision
- Saves best checkpoint by validation accuracy to `checkpoints/best_model.pth`

### Evaluation (`training/evaluation/`)

**World-EIoU** metric: `tIoU × mean(SPIoU_start, SPIoU_end)` where `SPIoU = max(0, 1 - distance/d_max)` with `d_max=5.0m`.

**AP pipeline** (greedy matching):
- Predictions matched to GT by highest EIoU > 0
- `eiou > threshold (0.5)` → keeps GT label; `0 < eiou ≤ threshold` → label=-1 (FP for both classes); `eiou == 0` → label=0 (non-violation)
- mAP = mean(APv, APn) over violation and non-violation classes

## Critical Label Convention

There is a **label inversion** between training and evaluation:
- `ViolationDataset`: `annotation=0` = violation, `annotation=1` = compliance
- Evaluation event schema: `label=1` = violation, `label=0` = non-violation
- `run_evaluation.py` handles the remap: `label = 1 if annotation == 0 else 0`

## Key Files

| File | Purpose |
|------|---------|
| `training/train.py` | Training entry point and loop |
| `training/dataset/violation_dataset.py` | Label parsing, trajectory building, dataset |
| `training/models/fusion_model.py` | `TrajectoryOnlyModel` (main model) |
| `training/models/trajectory_encoder.py` | GRU encoder |
| `training/evaluation/world_eiou.py` | `calculate_tiou`, `calculate_spiou`, `calculate_world_eiou` |
| `training/evaluation/ap_calculator.py` | `match_predictions_to_gt`, `compute_ap`, `compute_map` |
| `training/evaluation/run_evaluation.py` | Baseline eval CLI |
| `training/evaluation/evaluate_model.py` | Model inference + AP evaluation |
| `training/summarize.md` | Detailed pipeline documentation |

## Data Notes

- Only videos 1–10 have parquet files so far
- `data/` and `checkpoints/` are gitignored (not tracked)
- No `requirements.txt` — dependencies (torch, pandas, numpy, torchvision, tqdm) assumed pre-installed
