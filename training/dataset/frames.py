import cv2
import h5py
import numpy as np
import torch


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_frames_h5(h5_file: h5py.File, key: str, num_frames: int, size: int = 224) -> torch.Tensor:
    if key not in h5_file:
        raise KeyError(key)
    
    arr = h5_file[key][:]  # (32, H, W, 3) uint8
    frames = arr[:num_frames]  # (num_frames, H, W, 3) => Since h5 already sampled to 32 frames, this should be a no-op but we slice just in case.
    if size != arr.shape[1] or size != arr.shape[2]:
        frames = np.stack([cv2.resize(f, (size, size)) for f in frames])

    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (tensor - mean) / std
