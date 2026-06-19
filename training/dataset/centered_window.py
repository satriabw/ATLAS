"""Centered-window trajectory features — experimental data pipeline (v1, 2026-06-17).

Motivation (see docs/centered_window_experiment.md):
The production pipeline (dataset/trajectory.py) consumes the WHOLE vehicle track
and linspace-resamples it to 32 steps. For long/idle tracks (parked cars, dual-ROI
mega-tracks up to 9000 frames) that aliases the actual interaction down to ~1 of
32 slots. This module instead does an EEG/ECG-style centered crop:

  1. centre t0 = the single frame where ANY of the top-k pedestrians is globally
     closest to the vehicle (min ||p_loc - v_loc||). Robust to discontinuous
     pedestrian tracks (a track id may reappear) because we take the global min.
  2. window = the 32 vehicle steps [t0-16 .. t0+15]; the interaction is ALWAYS at
     index 16. Short sides at the track edges are ZERO-PADDED (mask=True), so the
     temporal phase is identical across every event.
  3. the vehicle is centered on its position at t0, and pedestrians are sampled at
     the SAME 32 frames as the vehicle — so trajectory and (future) vision crops
     share one frame index, which is what makes aligned fusion clean.

This file is intentionally separate from the production path so the two can be
A/B-compared without risk. Production stays the recommendation until this wins.
Padding-mask convention matches the rest of the repo: True = padded (ignore).
"""
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .labels import ViolationLabel
from .loader import _parse_labels, _assemble_labels
from .frames import load_frames_h5
from .tracking import parse_tracking, group_grid_boxes
from .trajectory import (DEFAULT_TOP_K, _to_frames, _to_loc, _to_scalar_seq,
                         _top_k_peds, _extract_row_arrays)

logger = logging.getLogger(__name__)

HALF = 16  # ±16 → 32-frame window, interaction at index HALF
TRACK_W, TRACK_H, PAD_FRAC = 1200, 1100, 0.15  # match scripts/build_h5_r2.union_window


def _union_window(v_boxes, p_boxes):
    """Event-static square window from grid boxes (NaN rows absent). Mirrors
    build_h5_r2.union_window so masks rasterize in the SAME crop the union h5 used."""
    boxes = np.concatenate([v_boxes, p_boxes.reshape(-1, 4)], axis=0)
    boxes = boxes[~np.isnan(boxes[:, 0])]
    if len(boxes) == 0:
        return np.array([0, 0, TRACK_W, TRACK_H], dtype=np.float32)
    x0, y0 = boxes[:, 0].min(), boxes[:, 1].min()
    x1, y1 = boxes[:, 2].max(), boxes[:, 3].max()
    side = max(x1 - x0, y1 - y0) * (1.0 + PAD_FRAC)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    wx0 = np.clip(cx - side / 2, 0, max(TRACK_W - side, 0))
    wy0 = np.clip(cy - side / 2, 0, max(TRACK_H - side, 0))
    return np.array([wx0, wy0, min(wx0 + side, TRACK_W), min(wy0 + side, TRACK_H)], dtype=np.float32)


def _vehicle_steps(group_df):
    """Unique, frame-sorted vehicle (frame, planar, speed)."""
    vf = np.concatenate([_to_frames(r['frames']) for _, r in group_df.iterrows()])
    vl = np.vstack([_to_loc(r['v_loc_planar']) for _, r in group_df.iterrows()])
    vs = np.concatenate([_to_scalar_seq(r['v_speed']) for _, r in group_df.iterrows()])
    order = np.argsort(vf, kind='stable')
    _, keep = np.unique(vf[order], return_index=True)
    idx = order[keep]
    return vf[order][keep], vl[idx], vs[idx]


def _closest_frame(group_df, top_ped_ids):
    """Frame of the global minimum vehicle↔(any top-k pedestrian) distance."""
    best_f, best_d = None, np.inf
    for pid in top_ped_ids:
        p_f, v_loc_k, _, p_loc_k, _ = _extract_row_arrays(group_df[group_df['p_track_id'] == pid])
        d = np.linalg.norm(p_loc_k - v_loc_k, axis=1)
        j = int(np.argmin(d))
        if d[j] < best_d:
            best_d, best_f = d[j], int(p_f[j])
    return best_f


def build_centered_window(group_df, top_k=DEFAULT_TOP_K, half=HALF):
    """Return a centered ±half window. See module docstring for the spec."""
    group_df = group_df.copy()
    group_df['_first_frame'] = group_df['frames'].apply(lambda f: int(_to_frames(f)[0]))
    group_df = group_df.sort_values('_first_frame').reset_index(drop=True)

    vf, vloc, vsp = _vehicle_steps(group_df)
    ped_ids = group_df['p_track_id'].unique()
    top_ped_ids = _top_k_peds(group_df, ped_ids, top_k)

    centre_frame = _closest_frame(group_df, top_ped_ids)
    c_idx = int(np.argmin(np.abs(vf - centre_frame)))
    v_origin = vloc[c_idx]
    n = 2 * half

    # per-pedestrian frame -> (planar, speed) lookup
    ped_maps = {}
    for pid in top_ped_ids:
        p_f, _, _, p_loc_k, p_sp_k = _extract_row_arrays(group_df[group_df['p_track_id'] == pid])
        ped_maps[pid] = {int(f): (p_loc_k[i], float(p_sp_k[i, 0])) for i, f in enumerate(p_f)}

    v_feat = np.zeros((n, 3), np.float32)
    v_mask = np.ones(n, bool)              # True = padded
    frames = np.full(n, -1, np.int64)      # shared frame index for vision alignment
    for k in range(n):
        si = c_idx - half + k
        if 0 <= si < len(vf):
            v_feat[k, :2] = vloc[si] - v_origin
            v_feat[k, 2] = vsp[si]
            v_mask[k] = False
            frames[k] = vf[si]

    ped_feats, ped_masks = [], []
    for pid in top_ped_ids:
        pf_arr = np.zeros((n, 3), np.float32)
        pm = np.ones(n, bool)
        pmap = ped_maps[pid]
        for k in range(n):
            si = c_idx - half + k
            f = int(frames[k])
            if f >= 0 and f in pmap:
                p_loc, p_sp = pmap[f]
                pf_arr[k, :2] = p_loc - vloc[si]
                pf_arr[k, 2] = p_sp
                pm[k] = False
        ped_feats.append(pf_arr)
        ped_masks.append(pm)

    return {
        'centre_frame': int(centre_frame),
        'frames': frames,
        'vehicle_feat': v_feat,
        'vehicle_mask': v_mask,
        'ped_feats': ped_feats,
        'ped_masks': ped_masks,
        'ped_ids': [int(p) for p in top_ped_ids],
    }


class CenteredDataset(Dataset):
    """Yields the same sample keys as ViolationDataset, from centered windows."""

    def __init__(self, labels, window_data, top_k=DEFAULT_TOP_K, half=HALF, h5_path=None,
                 box_data=None, zero_masks=False):
        self.labels = labels
        self.window_data = window_data
        self.top_k = top_k
        self.n = 2 * half
        # Optional centered-crop vision h5 (built from the SAME build_centered_window
        # grid, so frame slots align 1:1 with the trajectory window). Adds 'frames'.
        self._h5_file = h5py.File(Path(h5_path), 'r') if h5_path else None
        # Optional grounding: per-key (v_boxes, p_boxes) on the centered grid → frames
        # become 5-channel (RGB + vehicle/ped box masks). zero_masks keeps 5 channels
        # but zeros the masks (ungrounded control for a same-arch A/B).
        self.box_data = box_data
        self.zero_masks = zero_masks

    def __del__(self):
        if getattr(self, '_h5_file', None) is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        w = self.window_data[(label.video_id, label.tracking_id, label.roi)]

        p_feats = list(w['ped_feats'][:self.top_k])
        p_masks = list(w['ped_masks'][:self.top_k])
        while len(p_feats) < self.top_k:
            p_feats.append(np.zeros((self.n, 3), np.float32))
            p_masks.append(np.ones(self.n, bool))

        sample = {
            'vehicle_feat':   torch.from_numpy(w['vehicle_feat']),
            'ped_feat':       torch.from_numpy(np.concatenate(p_feats, axis=0)),
            'v_padding_mask': torch.from_numpy(w['vehicle_mask']),
            'p_padding_mask': torch.from_numpy(np.concatenate(p_masks, axis=0)),
            'label':          torch.tensor(label.annotation, dtype=torch.long),
            'video_id':       label.video_id,
            'tracking_id':    label.tracking_id,
            'roi':            label.roi,
            'start_frame':    label.start_frame,
        }
        if self._h5_file is not None:
            h5_key = f"V{label.video_id[-3:]}_{label.tracking_id}_{label.roi}"
            boxes, window = None, None
            if self.box_data is not None:
                key = (label.video_id, label.tracking_id, label.roi)
                boxes = self.box_data[key]
                # recompute the SAME union window the h5 build used (first ped,
                # valid frames) so masks align with the stored union crop.
                valid = self.window_data[key]['frames'] >= 0
                v_boxes, p_boxes = boxes
                window = _union_window(v_boxes[valid], p_boxes[:1, valid])
            frames = load_frames_h5(self._h5_file, h5_key, self.n, roi=label.roi,
                                    boxes=boxes, window=window)
            if boxes is not None and self.zero_masks:
                frames[:, 3:] = 0.0
            sample['frames'] = frames
        return sample


def load_centered_dataset(data_root, label_file='train', top_k=DEFAULT_TOP_K,
                          video_filter=None, half=HALF, h5_name=None,
                          ground=False, zero_masks=False):
    data_root = Path(data_root)
    pkl_path = data_root / 'data' / 'raw' / 'labels' / f'{label_file}_labels.pkl'
    tracking_dir = data_root / 'data' / 'raw' / 'tracking' if ground else None

    def _to_vid(v):
        return v if isinstance(v, str) else f"video_{v:03d}"
    allowed = ({_to_vid(video_filter)} if isinstance(video_filter, (str, int))
               else {_to_vid(v) for v in video_filter} if video_filter else None)

    parsed = _parse_labels(pkl_path, allowed)
    video_ids = sorted({p[0] for p in parsed})
    parquet_dir = data_root / 'data' / 'processed' / 'interactions'

    window_data, frame_ranges = {}, {}
    box_data = {} if ground else None
    for vid in video_ids:
        parquet_path = parquet_dir / f'{vid}_interactions.parquet'
        if not parquet_path.exists():
            logger.warning(f"Parquet not found for {vid}: {parquet_path}")
            continue
        df = pd.read_parquet(parquet_path)
        track_frames = None
        if ground:
            tp = tracking_dir / f'{vid}.txt'
            if tp.exists():
                track_frames = parse_tracking(tp)
            else:
                logger.warning(f"No tracking file for {vid}: {tp}")
        for (v_track_id, roi), group in df.groupby(['v_track_id', 'roi']):
            key = (vid, int(v_track_id), str(roi))
            try:
                w = build_centered_window(group, top_k, half=half)
                window_data[key] = w
                frame_ranges[key] = w['centre_frame']
                if ground and track_frames is not None:
                    # boxes on the centered-window frame grid (NOT linspace) so masks
                    # align 1:1 with the centered crops.
                    box_data[key] = group_grid_boxes(
                        track_frames, w['frames'], int(v_track_id), w['ped_ids'])
            except Exception as ex:
                logger.warning(f"Could not build centered window for {key}: {ex}")
    logger.info(f"Built centered-window cache: {len(window_data)} groups")

    labels = _assemble_labels(parsed, frame_ranges)

    h5_path = None
    if h5_name is not None:
        h5_path = data_root / 'data' / 'raw' / 'video' / h5_name
        if not h5_path.exists():
            raise FileNotFoundError(f"{h5_name} not found at {h5_path}")
        # drop labels whose centered-crop clip is absent from the h5 (same policy
        # as ViolationDataset's vision path)
        with h5py.File(h5_path, 'r') as f:
            keys = set(f.keys())
        before = len(labels)
        labels = [l for l in labels
                  if f"V{l.video_id[-3:]}_{l.tracking_id}_{l.roi}" in keys]
        if before != len(labels):
            logger.warning(f"Dropped {before - len(labels)} labels with no h5 frames")

    if ground:
        # also drop labels without computed boxes (missing tracking file)
        before = len(labels)
        labels = [l for l in labels
                  if (l.video_id, l.tracking_id, l.roi) in box_data]
        if before != len(labels):
            logger.warning(f"Dropped {before - len(labels)} labels with no grounding boxes")

    return CenteredDataset(labels, window_data, top_k=top_k, half=half, h5_path=h5_path,
                           box_data=box_data, zero_masks=zero_masks)
