import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def parse_tracking(path: Path):
    """Parse a tracking txt file (block format: frame, count, then count lines of
    `track_id cls x1 y1 x2 y2` in 1200x1100 image space).

    Returns {frame: {track_id: (x1, y1, x2, y2)}}.
    """
    frames = {}
    with open(path) as f:
        tokens = f.read().split('\n')
    i, n = 0, len(tokens)
    while i < n:
        line = tokens[i].strip()
        if not line:
            i += 1
            continue
        frame = int(line)
        count = int(tokens[i + 1])
        objs = {}
        for j in range(count):
            parts = tokens[i + 2 + j].split()
            objs[int(parts[0])] = (
                float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            )
        frames[frame] = objs
        i += 2 + count
    return frames


def group_grid_boxes(track_frames, grid, v_track_id, ped_ids):
    """Look up subject-vehicle and pedestrian boxes at each sampled grid frame.

    Returns (v_boxes (F,4), p_boxes (P,F,4)) float32; NaN rows where the track
    has no box at that frame.
    """
    F = len(grid)
    v_boxes = np.full((F, 4), np.nan, dtype=np.float32)
    p_boxes = np.full((max(len(ped_ids), 1), F, 4), np.nan, dtype=np.float32)
    for i, f in enumerate(grid):
        objs = track_frames.get(int(f), {})
        if v_track_id in objs:
            v_boxes[i] = objs[v_track_id]
        for k, pid in enumerate(ped_ids):
            if pid in objs:
                p_boxes[k, i] = objs[pid]
    return v_boxes, p_boxes
