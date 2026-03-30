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
   For each group → _build_group_trajectory()
        ↓
4. Build ViolationLabel list (skip if no matching parquet group)
        ↓
5. Return ViolationDataset
```

### Trajectory Building (`_build_group_trajectory`)

Each `(v_track_id, roi)` group can span multiple parquet rows. The function:
- Picks the **first pedestrian** (earliest-appearing `p_track_id`)
- Concatenates and **sorts by frame index**
- Returns two separate feature arrays:

```
vehicle_feat (T, 3) = [v_loc_x_centered, v_loc_y_centered, v_speed]
                       └──── origin = vehicle pos at t=0 ────┘

ped_feat (T, 3)     = [p_loc_x_rel, p_loc_y_rel, p_speed]
                       └──── relative to vehicle position ───┘
```

Vehicle trajectory is **ego-centered** (origin at first position). Pedestrian location is expressed relative to the vehicle at each frame.

If no pedestrian is present in an interaction, `has_pedestrian=False` is set and the event is automatically classified as **non-violation** (score zeroed, not passed through the model).

### Normalization (`_zscore_speed`)

Applied per-sample in `__getitem__` after resampling. The speed column (index 2) of both `vehicle_feat` and `ped_feat` is z-score normalized independently:

```python
speed_normalized = (speed - mean) / (std + 1e-6)
```

### Trajectory Resampling (`_resample`)

Sequences are resampled to a fixed `num_frames=32`:

| Case | Strategy |
|------|----------|
| T == 32 | Use as-is |
| T > 32  | Uniformly subsample with `np.linspace` |
| T < 32  | Tile and repeat to fill 32 frames |

### Dataset Item (`__getitem__`)
```python
{
    'vehicle_feat':   Tensor(32, 3),   # float32 — ego-centered vehicle trajectory
    'ped_feat':       Tensor(32, 3),   # float32 — vehicle-relative ped trajectory
    'has_pedestrian': Tensor(bool),    # False if trajectory data was missing
    'label':          Tensor(scalar),  # 0=violation, 1=compliance (long)
    'video_id':       str,
    'tracking_id':    int,
    'start_frame':    int,
}
```

---

## 2. Model Architecture (`models/`)

### `TrajectoryEncoder` (Bidirectional GRU)
```
Input:  (B, T=32, 3)
          ↓
Linear(3 → 64) + ReLU          ← embedding projection
          ↓
BiGRU(input=64, hidden=64, num_layers=1, bidirectional=True)
          ↓
Output: (B, T, 128)             ← full sequence, 64×2 directions
```

One encoder is used for the vehicle stream and a separate independent encoder for the pedestrian stream.

### `CrossAttentionModel` (full model)
```
vehicle_feat (B, 32, 3)          ped_feat (B, 32, 3)
       ↓                                ↓
TrajectoryEncoder            TrajectoryEncoder
       ↓                                ↓
vehicle_enc (B, T, 128)      ped_enc (B, T, 128)
               ↓
MultiheadAttention(embed=128, heads=4)
  query = vehicle_enc
  key   = ped_enc
  value = ped_enc
               ↓
attended (B, T, 128)
               ↓
max-pool over T → (B, 128)
               ↓
Linear(128→64) → ReLU → Dropout(0.3)
               ↓
Linear(64→2)
               ↓
logits (B, 2)   [class 0=violation, class 1=compliance]
```

The cross-attention lets the vehicle encoder **query the pedestrian trajectory** to highlight the frames most relevant for classification.

---

## 3. Training Loop (`train.py`)

### Setup
```
Load full dataset from train_labels.pkl
    ↓
Scene-stratified 85/15 train/val split by video
    ↓
Fixed class weights: violation=3.5, compliance=1.0
    ↓
Train with Adam optimizer + Mixed Precision (AMP) + Gradient Accumulation (×4)
```

### Scene-Stratified Split

All events from the same video are kept in the same partition. Videos are sorted, and the last 15% become the validation set. This prevents data leakage across splits.

```
Example (10 videos sorted): videos 1–8 → train, videos 9–10 → val
```

### Class Weights

Fixed weights (`violation=3.5, compliance=1.0`) passed to `CrossEntropyLoss` to counteract class imbalance.

### Gradient Accumulation

Effective batch size = `batch_size × accumulation_steps` = `2 × 4 = 8`. Simulates a larger batch without extra GPU memory.

### Per-Epoch Flow
```
train_epoch():
    for each batch:
        forward pass: model(vehicle_feat, ped_feat) → logits
        loss = CrossEntropy(logits, labels) / accumulation_steps
        backward()
        every 4 steps: optimizer.step(), zero_grad()

validate():
    no_grad + AMP
    compute loss + accuracy

compute_val_ap():
    run full AP evaluation on val video IDs → APv, APn, mAP
```

### Checkpointing
- Primary criterion: best **APv** (AP for violation class)
- Fallback if no parquet files available for val videos: best **val accuracy**
- Saved to `checkpoints/best_model.pth`
- Overfit mode saves to `checkpoints/overfit_model.pth`

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
    "gt_label":    int,        # 1=violation, 0=non-violation
    "frame_start": int,
    "frame_end":   int,
    "pos_start":   [x, y],     # vehicle world position (metres)
    "pos_end":     [x, y],
    "score":       float,      # P(violation) from model softmax
    "score_n":     float,      # P(non-violation) = 1 - score
    "eiou":        float,      # World-EIoU with matched GT event
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
Standard intersection-over-union on frame ranges (inclusive).

**Spatial Proximity IoU (SPIoU):**
```
SPIoU = max(0, 1 - distance / d_max)
```
Where `distance` is Euclidean distance in metres between vehicle positions, and `d_max=5.0m` is the threshold where score becomes 0.

SPIoU is computed at both the **start** and **end** positions, then averaged.

**Result:** 0 immediately if different `video_id`, different `roi`, or `tIoU == 0`; otherwise a score in `[0, 1]`.

---

### AP Calculation (`ap_calculator.py`)

**Two APs are computed:**
- **APv** — ranks predictions by `score` (P(violation)); TP if `gt_label == 1`
- **APn** — ranks predictions by `score_n` (P(non-violation)); TP if `gt_label == 0`
- **mAP** = (APv + APn) / 2

**Localization penalty (`compute_map`):**
Before computing AP, predictions where `predicted_class == gt_label` but `eiou ≤ 0.5` have both scores zeroed — correct class but poor localization is penalized.

```python
predicted_class = 1 if score >= 0.5 else 0
if predicted_class == gt_label and eiou <= threshold:
    score = 0.0;  score_n = 0.0
```

**Per-class AP (`compute_ap`):**
- Sort predictions by the class-specific score key descending
- Walk the sorted list, accumulate TP/FP at each position
- Build precision-recall curve (prepended with P=1, R=0)
- Area under curve via **trapezoidal rule** (`np.trapz`)

---

### Key Numbers

| Parameter | Value |
|-----------|-------|
| `d_max` for SPIoU | 5.0 m |
| EIoU threshold (TP cutoff) | 0.5 |
| Trajectory length | 32 frames |
| Vehicle/ped feature dims | 3 each |
| GRU encoder output dim | 128 (bidirectional) |
| Cross-attention heads | 4 |
| Baseline expected mAP | ~0.5 |

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

### Evaluation Modes

**Baseline (`run_evaluation.py`)** — no model, `score=1.0`, `eiou=1.0` for all labeled events:
```
python -m evaluation.run_evaluation \
    --parquet-dir data/processed/interactions \
    --labels-pkl  data/raw/labels/train_labels.pkl \
    --video-ids   1 2 3 4 5
```

Flow:
```
build_detected_events()       ← from parquet (score=1.0, no model)
    ↓
build_gt_events()             ← from labels pkl, proxies positions from detected
    ↓
compute_localization_rate()   → localization report
    ↓
build_baseline_events()       ← labeled events only, score=1.0, eiou=1.0
compute_map()                 → APv, APn, mAP
```

> **Baseline expectation:** `mAP ≈ 0.5` — all predictions get the same score so ranking is random.

**Model evaluation (`evaluate_model.py`)** — loads a checkpoint and scores each event:
```
python evaluation/evaluate_model.py \
    --checkpoint checkpoints/best_model.pth \
    --video-ids  2 4 6 8 10
```

Flow:
```
Load checkpoint → CrossAttentionModel
    ↓
For each labeled event in parquet:
    Build vehicle_feat (32, 3) and ped_feat (32, 3)
    Run model(vehicle_feat, ped_feat) → softmax → score = probs[:, 0]  (logit 0 = violation)
    Set eiou = 1.0  (GT proxied from same detection → always matches)
    ↓
compute_map(predictions)  → APv, APn, mAP
```

Since `eiou = 1.0` for all events, the localization penalty never fires — mAP is a **pure classification quality** measure. The test split uses `test_labels.pkl`.

---

## End-to-End Data Flow Summary

```
Raw label PKL  ──┐
                 ├──► load_violation_dataset() ──► ViolationDataset
Parquet files  ──┘     (vehicle_feat, ped_feat)         │
                                                         │
                                              Scene-stratified 85/15 split
                                              (by video, no leakage)
                                                         │
                                                   DataLoader
                                                         │
                                              CrossAttentionModel
                                              (BiGRU × 2 + cross-attn)
                                                         │
                                              CrossEntropyLoss
                                              (fixed weights 3.5/1.0)
                                                         │
                                              Adam + AMP + accumulation×4
                                                         │
                                              checkpoints/best_model.pth
                                              (saved by best APv)

─────────────── Separately ─────────────────────────────

Parquet files  ──► build_detected_events()
Label PKL      ──► build_gt_events()
                        │
               compute_localization_rate()   → coverage check
               compute_map()                 → APv, APn, mAP
```

---

## 5. Comparison: ATLAS vs. `crossswalk-original` + `ap_cal.py`

This section documents how the original crosswalk pipeline (`crossswalk-original/`) and its AP evaluator (`ap_cal.py`) relate to ATLAS.

---

### 5.1 Original Pipeline Overview (`crossswalk-original/`)

The original system is a multi-stage preprocessing pipeline that works in **pixel/image space** from raw tracking output.

**Stage 1 — `filtering_1.py` (vehicle–pedestrian co-occurrence)**
```
tracking_output/*.txt (per-frame bounding boxes: vehicle + pedestrian)
    ↓
For each frame:
    Find vehicles intersecting crosswalk line segments (TOP or BOT region)
    Find pedestrians inside crosswalk polygon
    Only keep vehicles when pedestrians are also present in same region
    ↓
filtering_output/video_NNN_{top,bot}.txt
```

**Stage 2 — `filtering_2.py` (vehicle-only baseline)**
```
filtering_output/  (with-pedestrian vehicle tracks)
    ↓
Collect all vehicle IDs that ever appeared with pedestrians
Re-read tracking_output, keep those vehicles intersecting crosswalk lines
    (regardless of pedestrian presence)
    ↓
filtering_woatt_output/video_NNN_woatt_{top,bot}.txt
```
"woatt" = without attention = vehicle-only temporal extent.

**Stage 3 — `filtering_3.py` (summarize first/last appearance)**
```
filtering_woatt_output/
    ↓
For each vehicle ID: record first and last frame + bbox
Discard interactions shorter than 32 frames
    ↓
filtering_woatt_summary/video_NNN_woatt_{top,bot}.txt
```

**Stage 4a — `preprocessing_vr.py` (video representation)**
```
For each GT label (from label_txt/):
    Spatiotemporally match GT to detected events in filtering_woatt_summary/
    using: EIoU = tIoU × (sIoU_start + sIoU_end)
        where sIoU = pixel bounding-box overlap IoU
    Crop 32 uniformly-sampled video frames from intersection-video/
        using one of 4 fixed crop rectangles (crop_lt, crop_rt, crop_lb, crop_rb)
    ↓
vr/V{vid}I{id}S{roi}D{dir}R0A{label}/  (folder of frame images)
```

**Stage 4b — `preprocessing_rr.py` (trajectory representation)**
```
Same matching logic as preprocessing_vr.py
Crop 32 frames from generated_ped_images/ (rendered trajectory frames)
    ↓
rr/V{vid}I{id}S{roi}D{dir}R0A{label}/  (folder of trajectory frame images)
```

Matched GT events get `A0` (violation); unmatched detected events get `A1` (compliance). Both `vr` and `rr` can be combined as `mr` (merged representation) for multi-modal models.

---

### 5.2 AP Evaluator: `ap_cal.py` vs. `ap_calculator.py`

| Aspect | Original `ap_cal.py` | ATLAS `ap_calculator.py` |
|--------|----------------------|--------------------------|
| **Dependency** | `sklearn.metrics.average_precision_score` | Pure numpy, no sklearn |
| **Score input** | Pre-computed scores loaded from `score_dir/*.pkl` | Computed inline from model softmax |
| **EIoU input** | Loaded from `eiou_values_updated.txt` (pre-computed) | Computed live via `calculate_world_eiou()` |
| **EIoU space** | Pixel bounding-box space (sIoU = box overlap) | World space (SPIoU = proximity in metres) |
| **Localization penalty** | Same logic: if `predicted_class == gt_label AND eiou ≤ threshold` → score zeroed | Same logic |
| **AP curve** | sklearn handles interpolation internally | Explicit trapezoidal rule on precision-recall curve |
| **Classes** | pos_label=0 (violation), pos_label=1 (compliance) | APv (score), APn (score_n) |
| **mAP** | `(AP-V + AP-N) / 2` | `(APv + APn) / 2` |

The penalty logic is **identical** in both systems: a prediction with correct class but insufficient localization (`eiou ≤ threshold`) has its score zeroed, pushing it to the bottom of the ranked list and acting as a false positive.

---

### 5.3 Key Differences: ATLAS vs. Original

| Dimension | Original (`crossswalk-original`) | ATLAS (`f-trajectory-only`) |
|-----------|-----------------------------------|-----------------------------|
| **Input space** | Pixel-space bounding boxes (image coordinates) | World-space coordinates (metres, from parquet) |
| **Spatial EIoU** | `sIoU = bounding-box pixel overlap` | `SPIoU = max(0, 1 − dist/5m)` — distance-based |
| **Feature modality** | Image crops (video frames, trajectory renders) | Numerical trajectory: `[x, y, speed]` per frame |
| **Multi-modal** | Yes: `vr` (video), `rr` (trajectory images), `mr` (both) | No: trajectory numerics only |
| **Event detection** | 3-stage pixel-space filtering pipeline | Parquet groupby `(v_track_id, roi)` |
| **GT matching** | Explicit spatiotemporal matching at preprocessing time | GT proxied directly from same parquet group (eiou=1.0) |
| **No-pedestrian handling** | Not applicable (filtering_1 requires co-occurrence) | Auto non-violation (`has_pedestrian=False`) |
| **Model** | Video backbone (MAN, SlowFast, I3D, ViT, etc.) + trajectory head | BiGRU × 2 + cross-attention |
| **Coordinate system** | Image pixels; direction-aware (downward/upward) | Ego-centered world coords; pedestrian relative to vehicle |
| **Data format** | Folder of frame images per event | Parquet file per video |

---

### 5.4 Design Decisions in ATLAS

1. **World-space over pixel-space**: SPIoU based on metric distance is more physically interpretable and robust to camera perspective distortion.
2. **Numerical trajectory over image crops**: Avoids high-dimensional visual features; faster to train; no dependency on video storage.
3. **Cross-attention**: Allows the model to learn which pedestrian frames are most relevant to the vehicle's decision-making context, replacing the implicit co-occurrence filter of `filtering_1.py`.
4. **Unified data format**: A single parquet per video replaces the multi-stage txt file pipeline, simplifying preprocessing and reducing pipeline fragility.
5. **Live EIoU**: Computed at evaluation time rather than precomputed offline, enabling evaluation on arbitrary video subsets without regenerating files.
