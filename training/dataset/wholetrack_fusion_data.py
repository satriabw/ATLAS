"""Whole-track bed for the gated-r2 fusion experiment (2026-07-09, plan =
artifacts/docs/2026-07-09_gated_r2/plan.md).

Wraps a frames-free ViolationDataset (whole-track 32-slot trajectories, the
CrossAttentionModel bed) and attaches the FROZEN pooled r2plus1d feature (512,)
per event from r2_whole_feats.h5 (built by scripts/precompute_r2plus1d_feats.py).

shuffle_vision=True permutes the vision features across events (fixed seed) —
the placebo arm, same convention as AlignedFusionDataset.
"""
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .loader import load_violation_dataset


def _key(l):
    return f"V{l.video_id[-3:]}_{l.tracking_id}_{l.roi}"


class WholeTrackFusionDataset(Dataset):
    def __init__(self, data_root, label_file, feats_name='r2_whole_feats.h5',
                 top_k=5, num_frames=32, shuffle_vision=False, seed=0):
        data_root = Path(data_root)
        base = load_violation_dataset(data_root, label_file, num_frames=num_frames,
                                      top_k=top_k, use_vision=False)
        self.base = base
        self._feats_path = data_root / 'data' / 'raw' / 'video' / feats_name
        with h5py.File(self._feats_path, 'r') as f:
            fkeys = set(f.keys())
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
