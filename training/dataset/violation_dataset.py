import logging
import re
import cv2
import numpy as np
import pandas as pd
import pickle
import torch
import torchvision.transforms.functional as TF
from dataclasses import dataclass
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ViolationLabel:
    video_id: str
    tracking_id: int
    roi: str          # 'BOT' or 'TOP'
    start_frame: int
    end_frame: int
    annotation: int   # 0=violation, 1=compliance


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_train_label(label_str: str) -> Tuple[str, int, str, int]:
    m = re.match(r'V(\d+)I(\d+)S(\d)D\d+R\d+A(\d)', label_str)
    if not m:
        raise ValueError(f"Cannot parse label string: {label_str!r}")
    video_id    = f"video_{int(m.group(1)):03d}"
    tracking_id = int(m.group(2))
    roi         = 'BOT' if m.group(3) == '1' else 'TOP'
    annotation  = int(m.group(4))
    return video_id, tracking_id, roi, annotation


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------

def _to_loc(val) -> np.ndarray:
    arr = np.asarray(val)
    if arr.dtype == object:
        return np.stack(arr.tolist()).astype(np.float32)
    return arr.astype(np.float32).reshape(-1, 2)

def _to_scalar_seq(val) -> np.ndarray:
    return np.asarray(val, dtype=np.float32).ravel()

def _to_frames(val) -> np.ndarray:
    return np.asarray(val, dtype=np.int64).ravel()

def _to_dmin(val) -> float:
    return float(np.asarray(val, dtype=np.float64).ravel().min())


# ---------------------------------------------------------------------------
# Trajectory building
# ---------------------------------------------------------------------------

def _extract_row_arrays(
    rows_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames_parts, v_loc_parts, v_sp_parts, p_loc_parts, p_sp_parts = [], [], [], [], []
    for _, row in rows_df.iterrows():
        frames_parts.append(_to_frames(row['frames']))
        v_loc_parts.append(_to_loc(row['v_loc_planar']))
        v_sp_parts.append(_to_scalar_seq(row['v_speed']))
        p_loc_parts.append(_to_loc(row['p_loc_planar']))
        p_sp_parts.append(_to_scalar_seq(row['p_speed']))

    all_f = np.concatenate(frames_parts)
    order = np.argsort(all_f, kind='stable')

    return (
        all_f[order],
        np.vstack(v_loc_parts)[order],
        np.concatenate(v_sp_parts)[order].reshape(-1, 1),
        np.vstack(p_loc_parts)[order],
        np.concatenate(p_sp_parts)[order].reshape(-1, 1),
    )


def _build_vehicle_feat(group_df: pd.DataFrame) -> np.ndarray:
    """Vehicle trajectory from all rows in the group, deduplicated by frame.

    Using all rows (not just the primary pedestrian's) ensures the vehicle
    trajectory covers the full interaction window.
    """
    f_parts, loc_parts, sp_parts = [], [], []
    for _, row in group_df.iterrows():
        f_parts.append(_to_frames(row['frames']))
        loc_parts.append(_to_loc(row['v_loc_planar']))
        sp_parts.append(_to_scalar_seq(row['v_speed']))
    all_f   = np.concatenate(f_parts)
    all_loc = np.vstack(loc_parts)
    all_sp  = np.concatenate(sp_parts)
    order   = np.argsort(all_f, kind='stable')
    _, keep = np.unique(all_f[order], return_index=True)
    idx     = order[keep]
    v_loc   = all_loc[idx]
    v_sp    = all_sp[idx].reshape(-1, 1)
    v_centered = v_loc - v_loc[0:1]
    return np.concatenate([v_centered, v_sp], axis=1).astype(np.float32)


def _build_group_trajectory(
    group_df: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K,
) -> Tuple[int, int, np.ndarray, List[np.ndarray]]:
    """Build vehicle and top-K pedestrian trajectories for one (v_track_id, roi) group.

    Returns:
        start_frame  : int
        end_frame    : int
        vehicle_feat : (T, 3)  [v_loc_x_centered, v_loc_y_centered, v_speed]
        ped_feats    : List of up to top_k arrays each (T_k, 3)
                       [p_loc_x_rel, p_loc_y_rel, p_speed]
    """
    group_df = group_df.copy()
    group_df['_first_frame'] = group_df['frames'].apply(lambda f: int(_to_frames(f)[0]))
    group_df = group_df.sort_values('_first_frame').reset_index(drop=True)

    all_frames = np.concatenate([_to_frames(r['frames']) for _, r in group_df.iterrows()])
    start_frame, end_frame = int(all_frames.min()), int(all_frames.max())

    # Rank pedestrians by mean d_min (closest to vehicle first)
    ped_ids = group_df['p_track_id'].unique()
    if len(ped_ids) > 1 and 'd_min' in group_df.columns:
        ped_dmin = {
            pid: group_df[group_df['p_track_id'] == pid]['d_min'].apply(_to_dmin).mean()
            for pid in ped_ids
        }
        top_ped_ids = sorted(ped_dmin, key=ped_dmin.get)[:top_k]
    else:
        top_ped_ids = list(ped_ids[:top_k])

    vehicle_feat = _build_vehicle_feat(group_df)

    # Pedestrian trajectories for each selected ped
    ped_feats: List[np.ndarray] = []
    for pid in top_ped_ids:
        ped_rows = group_df[group_df['p_track_id'] == pid]
        _, v_loc_k, _, p_loc_k, p_sp_k = _extract_row_arrays(ped_rows)
        p_rel = p_loc_k - v_loc_k
        ped_feats.append(np.concatenate([p_rel, p_sp_k], axis=1).astype(np.float32))

    return start_frame, end_frame, vehicle_feat, ped_feats


# ---------------------------------------------------------------------------
# Resampling / padding
# ---------------------------------------------------------------------------

def _resample_trajectory(features: np.ndarray, num_frames: int) -> Tuple[np.ndarray, int]:
    """Resample or zero-pad a trajectory to num_frames. Returns (features, actual_len)."""
    T = features.shape[0]
    if T == num_frames:
        return features, T
    elif T > num_frames:
        idx = np.linspace(0, T - 1, num_frames, dtype=int)
        return features[idx], num_frames
    else:
        padded = np.zeros((num_frames, features.shape[1]), dtype=np.float32)
        padded[:T] = features
        return padded, T


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

def _parse_labels(pkl_path: Path, allowed: Optional[set]) -> List[Tuple[str, int, str, int]]:
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

    suffix = f" (filtered to {sorted(allowed)})" if allowed else ""
    logger.info(f"Parsed {len(parsed)} labels{suffix}")
    return parsed


def _load_parquet_trajectories(
    video_ids: List[str],
    parquet_dir: Path,
    top_k: int,
) -> Tuple[Dict, Dict]:
    traj_data: Dict[Tuple, Tuple] = {}
    frame_ranges: Dict[Tuple, Tuple] = {}

    for vid in video_ids:
        parquet_path = parquet_dir / f'{vid}_interactions.parquet'
        if not parquet_path.exists():
            logger.warning(f"Parquet not found for {vid}: {parquet_path}")
            continue
        df = pd.read_parquet(parquet_path)
        for (v_track_id, roi), group in df.groupby(['v_track_id', 'roi']):
            key = (vid, int(v_track_id), str(roi))
            try:
                s, e, v_feat, ped_feats = _build_group_trajectory(group, top_k)
                traj_data[key]    = (v_feat, ped_feats)
                frame_ranges[key] = (s, e)
            except Exception as ex:
                logger.warning(f"Could not build trajectory for {key}: {ex}")

    logger.info(f"Built trajectory cache: {len(traj_data)} (video, track, roi) groups")
    return traj_data, frame_ranges


def _assemble_labels(parsed: List[Tuple], frame_ranges: Dict) -> List[ViolationLabel]:
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
    logger.info(f"Final dataset: {len(labels)} samples ({skipped} skipped due to missing parquet)")
    return labels


# ---------------------------------------------------------------------------
# Video frame loading
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _load_frames(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    num_frames: int,
    size: int = 224,
    roi: Optional[str] = None,
) -> torch.Tensor:
    """Extract `num_frames` evenly-spaced frames from a video clip.

    Returns:
        Tensor of shape (num_frames, 3, size, size), ImageNet-normalised.
    """
    cap = cv2.VideoCapture(str(video_path))
    indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)

    out = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (size, size))
        else:
            frame = np.zeros((size, size, 3), dtype=np.uint8)
        out.append(frame)

    cap.release()

    # (num_frames, H, W, 3) → (num_frames, 3, H, W), float [0,1]
    tensor = torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return (tensor - mean) / std


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ViolationDataset(Dataset):
    def __init__(
        self,
        labels: List[ViolationLabel],
        traj_data: Dict[Tuple, Tuple],
        num_frames: int = 32,
        top_k: int = DEFAULT_TOP_K,
        video_dir: Optional[Path] = None,
    ):
        self.labels     = labels
        self.traj_data  = traj_data
        self.num_frames = num_frames
        self.top_k      = top_k
        self.video_dir  = Path(video_dir) if video_dir is not None else None
        logger.info(f"Initialized dataset with {len(labels)} samples "
                    f"(top_k={top_k}, vision={'yes' if video_dir else 'no'})")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        v_feat, p_feat, has_ped, v_mask, p_mask = self._get_modalities(
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

        if self.video_dir is not None:
            video_path = self.video_dir / f'{label.video_id}.avi'
            sample['frames'] = _load_frames(
                video_path, label.start_frame, label.end_frame, self.num_frames
            )

        return sample

    def _get_modalities(
        self, video_id: str, tracking_id: int, roi: str
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, torch.Tensor, torch.Tensor]:
        key   = (video_id, tracking_id, roi)
        entry = self.traj_data.get(key)

        if entry is None:
            logger.warning(f"No trajectory data for {key}, returning zeros")
            v_feat = torch.zeros((self.num_frames, 3))
            p_feat = torch.zeros((self.top_k * self.num_frames, 3))
            v_mask = torch.ones(self.num_frames, dtype=torch.bool)
            p_mask = torch.ones(self.top_k * self.num_frames, dtype=torch.bool)
            return v_feat, p_feat, False, v_mask, p_mask

        vehicle_feat_raw, ped_feats_raw = entry

        # Vehicle
        v_arr, v_len = _resample_trajectory(vehicle_feat_raw, self.num_frames)
        v_mask = _padding_mask(v_len, self.num_frames)

        # Top-K pedestrians: resample and stack; zero-fill if fewer than top_k
        p_arrs, p_masks = [], []
        for pf in ped_feats_raw[:self.top_k]:
            pf_arr, p_len = _resample_trajectory(pf, self.num_frames)
            p_arrs.append(pf_arr)
            p_masks.append(_padding_mask(p_len, self.num_frames))

        while len(p_arrs) < self.top_k:
            p_arrs.append(np.zeros((self.num_frames, 3), dtype=np.float32))
            p_masks.append(np.ones(self.num_frames, dtype=bool))

        p_feat = np.concatenate(p_arrs,  axis=0)
        p_mask = np.concatenate(p_masks, axis=0)

        return (
            torch.from_numpy(v_arr),
            torch.from_numpy(p_feat),
            True,
            torch.from_numpy(v_mask),
            torch.from_numpy(p_mask),
        )


def _padding_mask(valid_len: int, total_len: int) -> np.ndarray:
    """Bool mask of shape (total_len,) where True = padded (should be ignored)."""
    mask = np.zeros(total_len, dtype=bool)
    mask[valid_len:] = True
    return mask


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_violation_dataset(
    data_root: Path,
    label_file: str = 'train',
    num_frames: int = 32,
    top_k: int = DEFAULT_TOP_K,
    video_filter: Optional[Union[str, List[str]]] = None,
    use_vision: bool = False,
) -> ViolationDataset:
    data_root = Path(data_root)
    pkl_path  = data_root / 'data' / 'raw' / 'labels' / f'{label_file}_labels.pkl'

    allowed = ({video_filter} if isinstance(video_filter, str)
               else set(video_filter) if video_filter else None)

    parsed      = _parse_labels(pkl_path, allowed)
    video_ids   = sorted({p[0] for p in parsed})
    parquet_dir = data_root / 'data' / 'processed' / 'interactions'

    traj_data, frame_ranges = _load_parquet_trajectories(video_ids, parquet_dir, top_k)
    labels = _assemble_labels(parsed, frame_ranges)

    video_dir = data_root / 'data' / 'raw' / 'video' if use_vision else None

    return ViolationDataset(
        labels=labels, traj_data=traj_data,
        num_frames=num_frames, top_k=top_k,
        video_dir=video_dir,
    )
