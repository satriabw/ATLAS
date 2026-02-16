import cv2
import logging
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ViolationLabel:
    video_id: str
    tracking_id: int
    start_frame: int
    end_frame: int
    violation_type: int
    

def parse_spatial_label(label_str: str) -> ViolationLabel:
    v_idx = label_str.find('V')
    i_idx = label_str.find('I')
    s_idx = label_str.find('S')
    e_idx = label_str.find('E')
    a_idx = label_str.find('A')
    
    video_num = int(label_str[v_idx+1:i_idx])
    tracking_id = int(label_str[i_idx+1:s_idx])
    start_frame = int(label_str[s_idx+1:e_idx])
    end_frame = int(label_str[e_idx+1:a_idx])
    annotation = int(label_str[a_idx+1:])
    
    return ViolationLabel(
        video_id=f'video_{video_num:03d}',
        tracking_id=tracking_id,
        start_frame=start_frame,
        end_frame=end_frame,
        violation_type=annotation
    )


class ViolationDataset(Dataset):
    def __init__(
        self,
        labels: List[ViolationLabel],
        video_paths: Dict[str, Path],
        parquet_paths: Dict[str, Path],
        context_frames: int = 16,
        num_frames: int = 32,
        img_size: tuple = (224, 224),
        transform=None
    ):
        self.labels = labels
        self.video_paths = video_paths
        self.parquet_paths = parquet_paths
        self.context_frames = context_frames
        self.num_frames = num_frames
        self.img_size = img_size
        
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Store parquet paths only, load on demand to save memory
        self._parquet_cache = {}
        logger.info(f"Initialized dataset with {len(labels)} samples")
    
    def _get_parquet_data(self, video_id: str) -> Optional[pd.DataFrame]:
        """Lazy load parquet data on demand."""
        if video_id in self._parquet_cache:
            return self._parquet_cache[video_id]
        
        if video_id not in self.parquet_paths:
            return None
        
        try:
            df = pd.read_parquet(self.parquet_paths[video_id])
            self._parquet_cache[video_id] = df
            return df
        except Exception as e:
            logger.warning(f"Could not load parquet for {video_id}: {e}")
            return None
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        logger.debug(f"__getitem__ called for idx={idx}")
        label = self.labels[idx]
        logger.debug(f"  video_id={label.video_id}, tracking_id={label.tracking_id}, frames={label.start_frame}-{label.end_frame}")
        
        logger.debug("  Loading video clip...")
        video_clip = self._load_video_clip(
            label.video_id,
            label.start_frame,
            label.end_frame
        )
        logger.debug(f"  Video clip loaded: {video_clip.shape}")
        
        logger.debug("  Loading trajectory...")
        trajectory = self._get_trajectory_clip(
            label.video_id,
            label.tracking_id,
            label.start_frame,
            label.end_frame
        )
        logger.debug(f"  Trajectory loaded: {trajectory.shape}")
        
        result = {
            'frames': video_clip,
            'trajectory': trajectory,
            'label': torch.tensor(label.violation_type, dtype=torch.long),
            'video_id': label.video_id,
            'tracking_id': label.tracking_id,
            'start_frame': label.start_frame
        }
        logger.debug(f"  __getitem__ returning for idx={idx}")
        return result
    
    def _load_video_clip(self, video_id: str, start: int, end: int):
        """Load only the sampled frames from video (not the entire range)."""
        if video_id not in self.video_paths:
            raise ValueError(f"Video {video_id} not found in video_paths")

        video_path = self.video_paths[video_id]
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            zero_frames = torch.zeros((self.num_frames, 3, self.img_size[1], self.img_size[0]))
            return zero_frames

        try:
            clip_start = max(0, start - self.context_frames)
            clip_end = end + self.context_frames
            total_clip_frames = clip_end - clip_start

            if total_clip_frames <= 0:
                return torch.zeros((self.num_frames, 3, self.img_size[1], self.img_size[0]))

            # Compute which frame indices to sample BEFORE reading
            if total_clip_frames >= self.num_frames:
                sample_indices = np.linspace(0, total_clip_frames - 1, self.num_frames, dtype=int)
            else:
                # Repeat frames to fill num_frames
                sample_indices = []
                for _ in range(self.num_frames // total_clip_frames):
                    sample_indices.extend(range(total_clip_frames))
                sample_indices.extend(range(self.num_frames % total_clip_frames))
                sample_indices = np.array(sample_indices, dtype=int)

            # Only read the unique frames we actually need
            unique_indices = sorted(set(sample_indices))

            frames_by_idx = {}
            zero_frame = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)

            for frame_idx in unique_indices:
                abs_frame = clip_start + frame_idx
                cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_by_idx[frame_idx] = self.transform(frame_rgb)
                else:
                    frames_by_idx[frame_idx] = self.transform(zero_frame)

            # Assemble the final sequence in sample order
            frames = [frames_by_idx[idx] for idx in sample_indices]
            return torch.stack(frames, dim=0)
        finally:
            cap.release()
    
    def _get_trajectory_clip(self, video_id: str, tracking_id: int, start: int, end: int):
        """Get trajectory data for frame range."""
        df = self._get_parquet_data(video_id)
        
        if df is None:
            return torch.zeros((self.num_frames, 7), dtype=torch.float32)
        
        vehicle_mask = (df['v_track_id'] == tracking_id)
        clip_df = df[vehicle_mask].copy()
        
        if len(clip_df) == 0:
            return torch.zeros((self.num_frames, 7), dtype=torch.float32)
        
        clip_df = clip_df.iloc[0]
        
        v_loc = np.stack(clip_df['v_loc_planar'])
        v_speed = np.array(clip_df['v_speed']).reshape(-1, 1)
        p_loc = np.stack(clip_df['p_loc_planar'])
        p_speed = np.array(clip_df['p_speed']).reshape(-1, 1)
        
        distances = np.linalg.norm(v_loc - p_loc, axis=1, keepdims=True)
        
        features = np.concatenate([v_loc, v_speed, p_loc, p_speed, distances], axis=1)
        
        sampled_features = self._sample_trajectory(features)
        
        return torch.from_numpy(sampled_features).float()
    
    def _sample_frames(self, frames_tensor):
        """Sample frames to match num_frames."""
        total_frames = frames_tensor.shape[0]
        
        if total_frames == self.num_frames:
            return frames_tensor
        elif total_frames > self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            return frames_tensor[indices]
        else:
            indices = []
            for _ in range(self.num_frames // total_frames):
                indices.extend(range(total_frames))
            indices.extend(range(self.num_frames % total_frames))
            return frames_tensor[indices]
    
    def _sample_trajectory(self, features):
        """Sample trajectory to match num_frames."""
        total_timesteps = features.shape[0]
        
        if total_timesteps == self.num_frames:
            return features
        elif total_timesteps > self.num_frames:
            indices = np.linspace(0, total_timesteps - 1, self.num_frames, dtype=int)
            return features[indices]
        else:
            repeat_factor = self.num_frames // total_timesteps
            remainder = self.num_frames % total_timesteps
            indices = list(range(total_timesteps)) * repeat_factor + list(range(remainder))
            return features[indices]
        

def load_violation_dataset(
    data_root: Path,
    split: str = 'train',
    context_frames: int = 16,
    num_frames: int = 32,
    img_size: tuple = (224, 224),
    transform=None
):
    data_root = Path(data_root)
    
    label_file = data_root / 'data' / 'interim' / 'labels' / f'spatial_labels_{split}.pkl'
    with open(label_file, 'rb') as f:
        label_strings = pickle.load(f)
    
    labels = [parse_spatial_label(s) for s in label_strings]
    logger.info(f"Loaded {len(labels)} {split} labels")
    
    video_dir = data_root / 'data' / 'raw' / 'video'
    parquet_dir = data_root / 'data' / 'processed' / 'interactions'
    
    video_paths = {p.stem: p for p in video_dir.glob('*.avi')}
    parquet_paths = {
        p.stem.replace(f'_interactions_{split}', ''): p 
        for p in parquet_dir.glob(f'*_interactions_{split}.parquet')
    }
    
    logger.info(f"Found {len(video_paths)} videos, {len(parquet_paths)} parquet files")
    
    return ViolationDataset(
        labels=labels,
        video_paths=video_paths,
        parquet_paths=parquet_paths,
        context_frames=context_frames,
        num_frames=num_frames,
        img_size=img_size,
        transform=transform
    )

