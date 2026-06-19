"""Dataset for the aligned joint-fusion experiment (2026-06-19).

Wraps a (frames-free, fast) CenteredDataset for the trajectory tensors and
attaches the FROZEN precomputed per-slot appearance features from
centered_vision_feats.h5 (built by scripts/precompute_centered_vision_feats.py).
Keys align 1:1 by construction (same build_centered_window grid).

shuffle_vision=True yields a per-event PERMUTED vision feature (matched shape,
fixed seed) — the placebo arm: if full(shuffled) ≈ full(real), the measured
vision contribution is a capacity/regularization confound, not signal.
"""
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .centered_window import load_centered_dataset


def _key(l):
    return f"V{l.video_id[-3:]}_{l.tracking_id}_{l.roi}"


class AlignedFusionDataset(Dataset):
    def __init__(self, data_root, label_file, feats_name='centered_vision_feats.h5',
                 top_k=5, half=32, shuffle_vision=False, seed=0):
        data_root = Path(data_root)
        base = load_centered_dataset(data_root, label_file=label_file, top_k=top_k, half=half)
        self.base = base
        self._feats_path = data_root / 'data' / 'raw' / 'video' / feats_name
        with h5py.File(self._feats_path, 'r') as f:
            fkeys = {k for k in f.keys() if not k.endswith('__pad')}
        self.valid_idx = [i for i, l in enumerate(base.labels) if _key(l) in fkeys]
        self.labels = [base.labels[i] for i in self.valid_idx]   # for _scene_split
        self.keys = [_key(l) for l in self.labels]
        if shuffle_vision:
            rng = np.random.RandomState(seed)
            self.vis_keys = [self.keys[i] for i in rng.permutation(len(self.keys))]
        else:
            self.vis_keys = list(self.keys)
        self._feats = None  # opened lazily (fork-safe)

    def _feat_file(self):
        if self._feats is None:
            self._feats = h5py.File(self._feats_path, 'r')
        return self._feats

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, j):
        s = self.base[self.valid_idx[j]]
        s['vis_feat'] = torch.from_numpy(self._feat_file()[self.vis_keys[j]][:]).float()
        return s
