import logging

import cv2
import h5py
import numpy as np
import torch
from pathlib import Path


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

logger = logging.getLogger(__name__)


def load_frames(video_path: Path, start_frame: int, end_frame: int, num_frames: int, size: int = 224) -> torch.Tensor:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

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
            logger.warning(f"cap.read() failed at frame {idx} in {video_path.name}; substituting black frame")
            frame = np.zeros((size, size, 3), dtype=np.uint8)
        out.append(frame)
    cap.release()

    tensor = torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (tensor - mean) / std


def load_frames_h5(h5_file: h5py.File, key: str, num_frames: int, size: int = 224) -> torch.Tensor:
    if key not in h5_file:
        raise KeyError(key)
    arr = h5_file[key][:]  # (32, H, W, 3) uint8
    indices = np.linspace(0, arr.shape[0] - 1, num_frames, dtype=int)
    frames = arr[indices]  # (num_frames, H, W, 3)
    if size != arr.shape[1] or size != arr.shape[2]:
        frames = np.stack([cv2.resize(f, (size, size)) for f in frames])
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (tensor - mean) / std
