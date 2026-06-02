import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .labels import ViolationLabel, parse_train_label
from .trajectory import DEFAULT_TOP_K, build_group_trajectory, resample_trajectory, padding_mask
import h5py

from .frames import load_frames, load_frames_h5

logger = logging.getLogger(__name__)


class ViolationDataset(Dataset):
    def __init__(self, labels, traj_data, num_frames=32, top_k=DEFAULT_TOP_K, video_dir=None, h5_path=None):
        self.labels     = labels
        self.traj_data  = traj_data
        self.num_frames = num_frames
        self.top_k      = top_k
        self.video_dir  = Path(video_dir) if video_dir is not None else None
        self.h5_path    = Path(h5_path) if h5_path else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        v_feat, p_feat, has_ped, v_mask, p_mask = self._get_trajectories(
            label.video_id, label.tracking_id, label.roi
        )
        sample = {
            'vehicle_feat':   v_feat,
            'ped_feat':       p_feat,
            'v_padding_mask': v_mask,
            'p_padding_mask': p_mask,
            'has_pedestrian': torch.tensor(has_ped, dtype=torch.bool),
            'label':          torch.tensor(label.annotation, dtype=torch.long),
            'video_id':       label.video_id,
            'tracking_id':    label.tracking_id,
            'start_frame':    label.start_frame,
        }
        if self.h5_path is not None:
            h5_key = f"V{label.video_id[-3:]}_{label.tracking_id}_{label.roi}"
            try:
                with h5py.File(self.h5_path, 'r') as hf:
                    sample['frames'] = load_frames_h5(hf, h5_key, self.num_frames)
            except KeyError:
                logger.warning(f"H5 key not found: {h5_key}; falling back to video_dir" if self.video_dir else f"H5 key not found: {h5_key}; returning sample without frames")
                if self.video_dir is not None:
                    video_path = self.video_dir / f'{label.video_id}.avi'
                    sample['frames'] = load_frames(
                        video_path, label.start_frame, label.end_frame, self.num_frames
                    )
        elif self.video_dir is not None:
            video_path = self.video_dir / f'{label.video_id}.avi'
            sample['frames'] = load_frames(
                video_path, label.start_frame, label.end_frame, self.num_frames
            )
        return sample

    def _get_trajectories(self, video_id, tracking_id, roi):
        key   = (video_id, tracking_id, roi)
        entry = self.traj_data.get(key)

        if entry is None:
            raise RuntimeError(
                f"Trajectory entry missing for key {key!r}; this should be unreachable "
                "because _assemble_labels filters to keys present in frame_ranges"
            )

        vehicle_feat_raw, ped_feats_raw = entry
        v_arr, v_len = resample_trajectory(vehicle_feat_raw, self.num_frames)

        p_arrs, p_masks = [], []
        for pf in ped_feats_raw[:self.top_k]:
            pf_arr, p_len = resample_trajectory(pf, self.num_frames)
            p_arrs.append(pf_arr)
            p_masks.append(padding_mask(p_len, self.num_frames))

        while len(p_arrs) < self.top_k:
            p_arrs.append(np.zeros((self.num_frames, 3), dtype=np.float32))
            p_masks.append(np.ones(self.num_frames, dtype=bool))

        return (
            torch.from_numpy(v_arr),
            torch.from_numpy(np.concatenate(p_arrs, axis=0)),
            True,
            torch.from_numpy(padding_mask(v_len, self.num_frames)),
            torch.from_numpy(np.concatenate(p_masks, axis=0)),
        )


def _parse_labels(pkl_path, allowed):
    with open(pkl_path, 'rb') as f:
        label_strings, _ = pickle.load(f)
    logger.info(f"Loaded {len(label_strings)} raw label strings from {pkl_path.name}")

    parsed = []
    for s in label_strings:
        try:
            vid, tid, roi, ann = parse_train_label(s)
        except ValueError as e:
            logger.warning(f"Skipping unparseable label {s!r}: {e}")
            continue
        if allowed and vid not in allowed:
            continue
        parsed.append((vid, tid, roi, ann))

    logger.info(f"Parsed {len(parsed)} labels")
    return parsed


def _load_parquet_trajectories(video_ids, parquet_dir, top_k):
    traj_data, frame_ranges = {}, {}
    for vid in video_ids:
        parquet_path = parquet_dir / f'{vid}_interactions.parquet'
        if not parquet_path.exists():
            logger.warning(f"Parquet not found for {vid}: {parquet_path}")
            continue
        df = pd.read_parquet(parquet_path)
        for (v_track_id, roi), group in df.groupby(['v_track_id', 'roi']):
            key = (vid, int(v_track_id), str(roi))
            try:
                s, e, v_feat, ped_feats = build_group_trajectory(group, top_k)
                traj_data[key]    = (v_feat, ped_feats)
                frame_ranges[key] = (s, e)
            except Exception as ex:
                logger.warning(f"Could not build trajectory for {key}: {ex}")
    logger.info(f"Built trajectory cache: {len(traj_data)} groups")
    return traj_data, frame_ranges


def _assemble_labels(parsed, frame_ranges):
    labels, skipped = [], 0
    for vid, tid, roi, ann in parsed:
        key = (vid, tid, roi)
        if key not in frame_ranges:
            logger.warning(f"No parquet group for {key}, skipping")
            skipped += 1
            continue
        s, e = frame_ranges[key]
        labels.append(ViolationLabel(
            video_id=vid, tracking_id=tid, roi=roi,
            start_frame=s, end_frame=e, annotation=ann,
        ))
    logger.info(f"Final dataset: {len(labels)} samples ({skipped} skipped)")
    return labels


def load_violation_dataset(
    data_root: Path,
    label_file: str = 'train',
    num_frames: int = 32,
    top_k: int = DEFAULT_TOP_K,
    video_filter=None,
    use_vision: bool = False,
    h5_path=None,
) -> ViolationDataset:
    data_root = Path(data_root)
    pkl_path  = data_root / 'data' / 'raw' / 'labels' / f'{label_file}_labels.pkl'

    def _to_vid(v):
        return v if isinstance(v, str) else f"video_{v:03d}"
    allowed = ({_to_vid(video_filter)} if isinstance(video_filter, (str, int))
               else {_to_vid(v) for v in video_filter} if video_filter else None)

    parsed      = _parse_labels(pkl_path, allowed)
    video_ids   = sorted({p[0] for p in parsed})
    parquet_dir = data_root / 'data' / 'processed' / 'interactions'

    traj_data, frame_ranges = _load_parquet_trajectories(video_ids, parquet_dir, top_k)
    labels = _assemble_labels(parsed, frame_ranges)

    video_dir = data_root / 'data' / 'raw' / 'video' if use_vision else None
    return ViolationDataset(
        labels=labels, traj_data=traj_data,
        num_frames=num_frames, top_k=top_k, video_dir=video_dir, h5_path=h5_path,
    )
