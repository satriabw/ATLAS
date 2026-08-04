"""
One labelled interaction event -> aligned BEV window + quadrant crop.

The two branches must see the *same frames*, or the fusion is being asked to
associate a motion map with an image taken at a different moment. That is the
single invariant this module exists to hold, and it is what most of the tests
check.

Frame selection, and why it snaps
---------------------------------
Frames are sampled uniformly over the event span (`event_frame_grid`), then
**snapped to the nearest frame the vehicle actually occupies**. Without
snapping, a sampled frame that falls in a tracking gap scatters nothing and
produces an all-zero BEV slot beside a perfectly real crop frame -- a silent
hole in exactly the alignment we care about.

Measured on 1,802 train events: 98.7% of sampled frames land on a vehicle frame
and 91.1% of events hit on all 32, but the tail is bad (worst event 9.4%, and
14.3% of events have a non-contiguous frame range). Snapping costs slightly
non-uniform spacing and, where gaps are large, repeated frames. That is the same
trade `vision_crop` already documents for short events: a repeated frame is a
truer clip than a black one.

Repeats are handled by rasterising the *unique* snapped frames and then
expanding, because `build_event_bev` resolves each track frame to a single slot
via `searchsorted` and would leave duplicate slots empty.

BEV window
----------
`BEVGrid` is constructed over the quadrant window from `quadrant_geometry`
rather than the default full extent, so the raster covers the region the crop
can actually see. See that module for why 16 m, and for the recorded
label-correlated clipping confound.
"""
import warnings

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .bev import BEVGrid, build_event_bev
from .quadrant_geometry import CELLS, RESOLUTION, WINDOW_M, quadrant_window
from .vision_crop import NUM_FRAMES, crop_clip, event_frame_grid, parse_label, quadrant_rect

QUADRANTS = [('TOP', True), ('TOP', False), ('BOT', True), ('BOT', False)]
QUADRANT_INDEX = {q: i for i, q in enumerate(QUADRANTS)}


def _grid_for(roi, downward):
    x_min, y_min = quadrant_window(roi, downward)
    return BEVGrid(x_min, x_min + WINDOW_M, y_min, y_min + WINDOW_M, RESOLUTION)


def snapped_frames(group_df, num_frames=NUM_FRAMES):
    """Uniform samples over the span, snapped to frames the vehicle occupies.

    Returns ascending frame numbers, length num_frames, possibly with repeats.
    """
    vehicle = np.unique(np.concatenate(
        [np.asarray(r).ravel() for r in group_df['frames']])).astype(np.int64)
    grid = event_frame_grid(group_df, num_frames)
    idx = np.searchsorted(vehicle, grid)
    left = np.clip(idx - 1, 0, len(vehicle) - 1)
    right = np.clip(idx, 0, len(vehicle) - 1)
    take_left = np.abs(vehicle[left] - grid) <= np.abs(vehicle[right] - grid)
    return np.where(take_left, vehicle[left], vehicle[right])


def load_events(labels_pkl, parquet_dir, frame_db=None):
    """[(video_id, v_track_id, roi, downward, annotation)] that can actually be built.

    Drops labels with no parquet group (no vehicle/pedestrian co-occurrence was
    ever recorded) and, when `frame_db` is given, those whose video is absent
    from the frame database. Both are counted and warned about rather than
    silently skipped -- silent skipping is how this project lost track of 88
    labels once before.
    """
    import pickle
    with open(labels_pkl, 'rb') as f:
        strings, annotations = pickle.load(f)

    videos = None
    if frame_db is not None:
        with h5py.File(frame_db, 'r') as h5:
            videos = set(h5.keys())

    by_video = {}
    events, no_parquet, no_frames = [], 0, 0
    for s, a in zip(strings, annotations):
        video_id, track_id, roi, downward, _ = parse_label(s)
        if videos is not None and video_id not in videos:
            no_frames += 1
            continue
        if video_id not in by_video:
            path = f'{parquet_dir}/{video_id}_interactions.parquet'
            try:
                by_video[video_id] = pd.read_parquet(path)
            except (FileNotFoundError, OSError):
                by_video[video_id] = None
        df = by_video[video_id]
        if df is None or not ((df['v_track_id'] == track_id) & (df['roi'] == roi)).any():
            no_parquet += 1
            continue
        events.append((video_id, track_id, roi, downward, int(a)))

    if no_parquet or no_frames:
        warnings.warn(f'dropped {no_parquet} labels with no parquet group and '
                      f'{no_frames} with no frames, kept {len(events)}')
    return events


class FusionDataset(Dataset):
    """(bev, crop, quadrant, label) per event, both streams on identical frames."""

    def __init__(self, events, parquet_dir, frame_db, num_frames=NUM_FRAMES,
                 normalize=True):
        self.events = list(events)
        self.parquet_dir = str(parquet_dir)
        self.frame_db = str(frame_db)
        self.num_frames = num_frames
        self.normalize = normalize
        self._h5 = None          # opened lazily: an h5 handle cannot cross a fork
        self._parquet = {}

    def __len__(self):
        return len(self.events)

    def _group(self, video_id, track_id, roi):
        if video_id not in self._parquet:
            self._parquet[video_id] = pd.read_parquet(
                f'{self.parquet_dir}/{video_id}_interactions.parquet')
        df = self._parquet[video_id]
        return df[(df['v_track_id'] == track_id) & (df['roi'] == roi)]

    def __getitem__(self, i):
        video_id, track_id, roi, downward, label = self.events[i]
        group = self._group(video_id, track_id, roi)
        frames = snapped_frames(group, self.num_frames)

        # rasterise unique frames, then expand -- build_event_bev resolves each
        # track frame to one slot, so duplicates would come back empty
        unique, inverse = np.unique(frames, return_inverse=True)
        _, bev_unique = build_event_bev(group, grid=_grid_for(roi, downward),
                                        frames=unique)
        bev = bev_unique[inverse]

        if self._h5 is None:
            self._h5 = h5py.File(self.frame_db, 'r')
        crop = crop_clip(self._h5[video_id], frames, quadrant_rect(roi, downward),
                         normalize=self.normalize)

        return {
            'bev': torch.from_numpy(np.ascontiguousarray(bev)),
            'crop': torch.from_numpy(crop),
            'quadrant': QUADRANT_INDEX[(roi, downward)],
            'label': label,
            'frames': torch.from_numpy(frames),
        }
