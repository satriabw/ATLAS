# ATLAS Training Pipeline — Summary

## Overview

The goal is to classify traffic interaction events as **violation** or **compliance** using only trajectory data (positions + speeds of a vehicle and a pedestrian over time).

---

## 1. Dataset Loading (`dataset/violation_dataset.py`)

### Input Files
| File | Description |
|------|-------------|
| `data/raw/labels/train_labels.pkl` | List of label strings, e.g. `V001I00002S1D0R0A1` |
| `data/processed/interactions/*.parquet` | Per-video interaction data (positions, speeds, frames) |

### Label String Format
```
V001  I00002  S1  D0  R0  A1
 │      │     │           │
video  track  roi         annotation
              S1=BOT      A0=violation
              S0=TOP      A1=compliance
```

### Loading Steps (in `load_violation_dataset`)

```
1. Load label strings from .pkl
        ↓
2. Parse each string → (video_id, tracking_id, roi, annotation)
   Optionally filter to specific videos (video_filter)
        ↓
3. For each unique video, load its .parquet file
   Group rows by (v_track_id, roi)
   For each group → build_group_trajectory()
        ↓
4. Build ViolationLabel list (skip if no matching parquet group)
        ↓
5. Return ViolationDataset
```

### Trajectory Building (`_build_group_trajectory`)

Each `(v_track_id, roi)` group can span multiple parquet rows. The function:
- Picks the **first pedestrian** (earliest-appearing `p_track_id`)
- Concatenates and **sorts by frame index**
- Computes 7 features per timestep:

```
features (T, 7) = [v_loc_x, v_loc_y, v_speed, p_loc_x, p_loc_y, p_speed, v_p_distance]
                    ←── vehicle ──→              ←── pedestrian ──→        └─ Euclidean
```

### Trajectory Sampling (`_sample_trajectory`)

Since sequences have variable lengths, they are resampled to a fixed `num_frames=32`:

| Case | Strategy |
|------|----------|
| T == 32 | Use as-is |
| T > 32 | Uniformly subsample with `np.linspace` |
| T < 32 | Tile and repeat to fill 32 frames |

### Dataset Item (`__getitem__`)
```python
{
    'trajectory': Tensor(32, 7),      # float32
    'label':      Tensor(scalar),     # 0=violation, 1=compliance (long)
    'video_id':   str,
    'tracking_id': int,
    'start_frame': int,
}
```

---

## 2. Model Architecture (`models/`)

### `TrajectoryEncoder` (GRU)
```
Input:  (B, T=32, 7)
          ↓
GRU(input_size=7, hidden_size=128, num_layers=2, dropout=0.3)
          ↓
Take last layer hidden state: hidden[-1]
          ↓
Output: (B, 128)
```

The GRU processes the trajectory as a **sequence**, capturing temporal dynamics. The final hidden state is a compact summary of the entire interaction.

### `TrajectoryOnlyModel` (full model)
```
Input: trajectory (B, 32, 7)
          ↓
TrajectoryEncoder → (B, 128)
          ↓
Linear(128 → 64) → ReLU → Dropout(0.3)
          ↓
Linear(64 → 2)
          ↓
Output: logits (B, 2)   [class 0=violation, class 1=compliance]
```

---

## 3. Training Loop (`train.py`)

### Setup
```
Load full dataset
    ↓
Compute class weights (inverse frequency) → weighted CrossEntropyLoss
    ↓
80/20 train/val split (random, seed=42)
    ↓
Train with Adam optimizer + Mixed Precision (AMP) + Gradient Accumulation (×4)
```

### Class Weights
Because violations and compliance may be imbalanced:
```python
weight[class] = 1.0 / count[class]
weights are then normalized to sum to 1
```
This prevents the model from ignoring the minority class.

### Gradient Accumulation
Effective batch size = `batch_size × accumulation_steps` = `2 × 4 = 8`.
This simulates a larger batch without extra GPU memory.

### Per-Epoch Flow
```
train_epoch():
    for each batch:
        forward pass (AMP autocast)
        loss = CrossEntropy / accumulation_steps
        backward()
        every 4 steps: optimizer.step(), zero_grad()

validate():
    no_grad + AMP
    compute loss + accuracy
```

### Checkpointing
- Best model (by validation accuracy) saved to `checkpoints/best_model.pth`
- Overfit mode saves to `checkpoints/overfit_model.pth` instead

### Overfit Mode (`--overfit`)
A sanity check: train == val on a single video. If the model cannot reach ~100% accuracy here, there is a bug in the data pipeline or model.

---

## 4. Evaluation Pipeline (`evaluation/`)

The evaluation is **independent of training** — it uses the parquet data directly to build "detected events", then scores them against GT labels.

### Label Convention Difference (Important!)

| Context | violation | compliance |
|---------|-----------|------------|
| `ViolationDataset.annotation` | 0 | 1 |
| Evaluation event schema `label` | **1** | **0** |

`run_evaluation.py` remaps: `label = 1 if annotation == 0 else 0`

---

### Event Schema
Both predictions and GT are dicts:
```python
{
    "video_id":    str,        # e.g. "video_001"
    "v_track_id":  int,
    "roi":         str,        # "TOP" or "BOT"
    "label":       int,        # 1=violation, 0=non-violation
    "frame_start": int,
    "frame_end":   int,
    "pos_start":   [x, y],     # vehicle world position (metres)
    "pos_end":     [x, y],
    "score":       float,      # confidence (0–1)
}
```

---

### World-EIoU Metric (`world_eiou.py`)

A combined **temporal + spatial** similarity score between two events.

```
World-EIoU = tIoU × (SPIoU_start + SPIoU_end) / 2
```

**Temporal IoU (tIoU):**
```
tIoU = overlap_frames / union_frames
```
Standard intersection-over-union on frame ranges.

**Spatial Proximity IoU (SPIoU):**
```
SPIoU = max(0, 1 - distance / d_max)
```
Where `distance` is Euclidean distance in metres between vehicle positions, and `d_max=5.0m` is the threshold where score becomes 0.

SPIoU is computed at both the **start** and **end** positions, then averaged.

**Result:** 0 if different video/ROI, or no temporal overlap; otherwise a score in `[0, 1]`.

---

### AP Calculation (`ap_calculator.py`)

**Step 1 — Match predictions to GT (greedy):**
```
For each GT event:
    Find the unmatched prediction with highest EIoU > 0
    Assign GT label to that prediction
All remaining predictions → matched_label = 0 (non-violation)
```

**Step 2 — Apply EIoU threshold:**
```
If 0 < eiou ≤ 0.5 (weak match):
    matched_label = -1  → FP for both classes
If eiou == 0.0 (no match):
    matched_label = 0   → counts as non-violation TP
If eiou > 0.5 (good match):
    keep assigned GT label
```

**Step 3 — Compute AP per class:**
- Sort predictions by `score` descending
- Walk the sorted list, accumulate TP/FP
- Build precision-recall curve
- Area under curve via **trapezoidal rule**

**Step 4 — mAP:**
```
APv = AP for class 1 (violation)
APn = AP for class 0 (non-violation)
mAP = (APv + APn) / 2
```

---

### Localization Rate (`localization.py`)

A label-blind sanity check: _"Are the detected events spatially/temporally covering the GT events?"_

```
For each GT event:
    Find the detected event with best World-EIoU
    If best EIoU >= 0.5 → GT is "localized"

Localization Rate = localized / total_GT
```
Also reported per ROI (TOP / BOT).

---

### Evaluation Entry Point (`run_evaluation.py`)

```
python -m evaluation.run_evaluation \
    --parquet-dir data/processed/interactions \
    --labels-pkl  data/raw/labels/train_labels.pkl \
    --video-ids   1 2 3 4 5
```

Flow:
```
build_detected_events()   ← from parquet (score=1.0, no model yet)
    ↓
build_gt_events()         ← from labels pkl, proxies positions from detected
    ↓
compute_localization_rate()   → prints localization report
    ↓
compute_map()                 → prints APv, APn, mAP
```

> **Baseline expectation:** With `score=1.0` for all detections and no real model, `mAP ≈ 0.5` (random).

---

## End-to-End Data Flow Summary

```
Raw label PKL  ──┐
                 ├──► load_violation_dataset() ──► ViolationDataset
Parquet files  ──┘         (trajectories)               │
                                                         │
                                                   DataLoader
                                                         │
                                              TrajectoryOnlyModel
                                              (GRU → classifier)
                                                         │
                                              CrossEntropyLoss
                                              (class-weighted)
                                                         │
                                              Adam + AMP + accumulation
                                                         │
                                              checkpoints/best_model.pth

─────────────── Separately ─────────────────────────────

Parquet files  ──► build_detected_events()
Label PKL      ──► build_gt_events()
                        │
               compute_localization_rate()   → coverage check
               compute_map()                 → APv, APn, mAP
```
