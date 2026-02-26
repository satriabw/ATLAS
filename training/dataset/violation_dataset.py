import logging
import re
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ViolationLabel:
    video_id: str
    tracking_id: int
    roi: str          # 'BOT' or 'TOP'
    start_frame: int
    end_frame: int
    annotation: int   # 0=violation, 1=compliance


def parse_train_label(label_str: str) -> Tuple[str, int, str, int]:
    """Parse 'V001I00002S1D0R0A1' -> (video_id, tracking_id, roi, annotation).

    S1 = BOT, S0 = TOP
    A0 = violation, A1 = compliance
    """
    m = re.match(r'V(\d+)I(\d+)S(\d)D\d+R\d+A(\d)', label_str)
    if not m:
        raise ValueError(f"Cannot parse label string: {label_str!r}")
    video_id = f"video_{int(m.group(1)):03d}"
    tracking_id = int(m.group(2))
    roi = 'BOT' if m.group(3) == '1' else 'TOP'
    annotation = int(m.group(4))
    return video_id, tracking_id, roi, annotation


def _to_loc(val) -> np.ndarray:
    """Convert object array of (2,) coord arrays to (T, 2) float32 ndarray."""
    arr = np.asarray(val)
    if arr.dtype == object:
        return np.stack(arr.tolist()).astype(np.float32)
    return arr.astype(np.float32).reshape(-1, 2)


def _to_scalar_seq(val) -> np.ndarray:
    """Convert speed/distance sequence to (T,) float32 ndarray."""
    return np.asarray(val, dtype=np.float32).ravel()


def _to_frames(val) -> np.ndarray:
    """Convert frame index sequence to (T,) int64 ndarray."""
    return np.asarray(val, dtype=np.int64).ravel()


def _build_group_trajectory(
    group_df: pd.DataFrame,
) -> Tuple[int, int, np.ndarray]:
    """Build trajectory for one (v_track_id, roi) group.

    Frame range  = min/max across ALL rows in group.
    Trajectory   = first pedestrian (earliest-appearing p_track_id) only,
                   rows concatenated in frame order.

    Returns (start_frame, end_frame, features) where features is (T, 7):
        [v_loc_x, v_loc_y, v_speed, p_loc_x, p_loc_y, p_speed, v_p_distance]
    """
    group_df = group_df.copy()
    group_df['_first_frame'] = group_df['frames'].apply(lambda f: int(_to_frames(f)[0]))
    group_df = group_df.sort_values('_first_frame').reset_index(drop=True)

    # Frame range from ALL rows
    all_frames_flat = np.concatenate([_to_frames(row['frames']) for _, row in group_df.iterrows()])
    start_frame = int(all_frames_flat.min())
    end_frame = int(all_frames_flat.max())

    # Pick first pedestrian by chronological order
    first_ped = int(group_df.iloc[0]['p_track_id'])
    ped_rows = group_df[group_df['p_track_id'] == first_ped]

    # Concatenate trajectory data across matching rows
    frames_parts, v_loc_parts, v_sp_parts, p_loc_parts, p_sp_parts = [], [], [], [], []
    for _, row in ped_rows.iterrows():
        frames_parts.append(_to_frames(row['frames']))
        v_loc_parts.append(_to_loc(row['v_loc_planar']))    # (N, 2)
        v_sp_parts.append(_to_scalar_seq(row['v_speed']))   # (N,)
        p_loc_parts.append(_to_loc(row['p_loc_planar']))    # (N, 2)
        p_sp_parts.append(_to_scalar_seq(row['p_speed']))   # (N,)

    all_f = np.concatenate(frames_parts)
    order = np.argsort(all_f, kind='stable')

    v_loc_a = np.vstack(v_loc_parts)[order]                    # (T, 2)
    v_sp_a = np.concatenate(v_sp_parts)[order].reshape(-1, 1)  # (T, 1)
    p_loc_a = np.vstack(p_loc_parts)[order]                    # (T, 2)
    p_sp_a = np.concatenate(p_sp_parts)[order].reshape(-1, 1)  # (T, 1)
    dists = np.linalg.norm(v_loc_a - p_loc_a, axis=1, keepdims=True)  # (T, 1)

    features = np.concatenate([v_loc_a, v_sp_a, p_loc_a, p_sp_a, dists], axis=1).astype(np.float32)
    return start_frame, end_frame, features


class ViolationDataset(Dataset):
    def __init__(
        self,
        labels: List[ViolationLabel],
        traj_data: Dict[Tuple[str, int, str], np.ndarray],
        num_frames: int = 32,
    ):
        self.labels = labels
        self.traj_data = traj_data
        self.num_frames = num_frames

        logger.info(f"Initialized dataset with {len(labels)} samples")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        trajectory = self._get_trajectory(label.video_id, label.tracking_id, label.roi)

        return {
            'trajectory': trajectory,
            'label': torch.tensor(label.annotation, dtype=torch.long),
            'video_id': label.video_id,
            'tracking_id': label.tracking_id,
            'start_frame': label.start_frame,
        }

    def _get_trajectory(self, video_id: str, tracking_id: int, roi: str) -> torch.Tensor:
        key = (video_id, tracking_id, roi)
        features = self.traj_data.get(key)
        if features is None:
            logger.warning(f"No trajectory data for {key}, returning zeros")
            return torch.zeros((self.num_frames, 7), dtype=torch.float32)
        return torch.from_numpy(self._sample_trajectory(features))

    def _sample_trajectory(self, features: np.ndarray) -> np.ndarray:
        T = features.shape[0]
        if T == self.num_frames:
            return features
        elif T > self.num_frames:
            idx = np.linspace(0, T - 1, self.num_frames, dtype=int)
        else:
            repeat = self.num_frames // T
            rem = self.num_frames % T
            idx = list(range(T)) * repeat + list(range(rem))
        return features[idx]


def load_violation_dataset(
    data_root: Path,
    label_file: str = 'train',
    num_frames: int = 32,
    video_filter: Optional[str] = None,
) -> 'ViolationDataset':
    """Load ViolationDataset from ATLAS data root.

    Args:
        data_root:     ATLAS project root (contains data/).
        label_file:    'train' or 'test' — selects {label_file}_labels.pkl.
        video_filter:  If set (e.g. 'video_001'), restrict to that video only.
    """
    data_root = Path(data_root)

    # 1. Load label strings
    pkl_path = data_root / 'data' / 'raw' / 'labels' / f'{label_file}_labels.pkl'
    with open(pkl_path, 'rb') as f:
        label_strings, _ = pickle.load(f)
    logger.info(f"Loaded {len(label_strings)} raw label strings from {pkl_path.name}")

    # 2. Parse labels, optionally filter by video
    parsed = []
    for s in label_strings:
        try:
            vid, tid, roi, ann = parse_train_label(s)
            if video_filter and vid != video_filter:
                continue
            parsed.append((vid, tid, roi, ann))
        except Exception as e:
            logger.warning(f"Skipping unparseable label {s!r}: {e}")
    logger.info(f"Parsed {len(parsed)} labels" + (f" (filtered to {video_filter})" if video_filter else ""))

    # 3. Load parquet for each unique video, build trajectory + frame-range cache
    video_ids = sorted({p[0] for p in parsed})
    parquet_dir = data_root / 'data' / 'processed' / 'interactions'

    traj_data: Dict[Tuple[str, int, str], np.ndarray] = {}
    frame_ranges: Dict[Tuple[str, int, str], Tuple[int, int]] = {}

    for vid in video_ids:
        parquet_path = parquet_dir / f'{vid}_interactions.parquet'
        if not parquet_path.exists():
            logger.warning(f"Parquet not found for {vid}: {parquet_path}")
            continue
        df = pd.read_parquet(parquet_path)
        for (v_track_id, roi), group in df.groupby(['v_track_id', 'roi']):
            key = (vid, int(v_track_id), str(roi))
            try:
                s, e, features = _build_group_trajectory(group)
                traj_data[key] = features
                frame_ranges[key] = (s, e)
            except Exception as ex:
                logger.warning(f"Could not build trajectory for {key}: {ex}")

    logger.info(f"Built trajectory cache: {len(traj_data)} (video, track, roi) groups")

    # 4. Build ViolationLabel list (skip labels with no parquet data)
    labels = []
    skipped = 0
    for vid, tid, roi, ann in parsed:
        key = (vid, tid, roi)
        if key not in frame_ranges:
            logger.warning(f"No parquet group for {key}, skipping")
            skipped += 1
            continue
        s, e = frame_ranges[key]
        labels.append(ViolationLabel(
            video_id=vid,
            tracking_id=tid,
            roi=roi,
            start_frame=s,
            end_frame=e,
            annotation=ann,
        ))
    logger.info(f"Final dataset: {len(labels)} samples ({skipped} skipped due to missing parquet)")

    return ViolationDataset(
        labels=labels,
        traj_data=traj_data,
        num_frames=num_frames,
    )
