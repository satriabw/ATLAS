"""
Bird's-eye-view feature maps from interaction-parquet trajectories.

Grid frame = the calibration ground plane (camera_model.yml), which is shared by
all 120 videos, so cell indices are comparable across the dataset. Extent covers
the projected TOP+BOT ROI union (= the full vehicle envelope) plus a 5 m margin.
Note the calibration axes are not road-aligned: the two intersection arms run at
roughly 20 deg and 70 deg, so roads sit diagonally in the grid.

Channels (per timestep):
  0  vehicle count      always 0 or 1 -- one vehicle track per event
  1  pedestrian count   >=2 in ~1% of occupied cells at 0.5 m
  2  vehicle speed      mean over occupants (count is 1, so just the speed)
  3  pedestrian speed   mean over occupants

A cell with a stopped vehicle has ch0=1, ch2=0.0; occupancy is read from the
count channels, never from the speed channels.
"""
import numpy as np

RESOLUTION = 0.5
X_MIN, X_MAX = -18.5, 13.5
Y_MIN, Y_MAX = -16.5, 21.0

CH_VEHICLE_COUNT = 0
CH_PED_COUNT = 1
CH_VEHICLE_SPEED = 2
CH_PED_SPEED = 3
NUM_CHANNELS = 4


class BEVGrid:
    """Maps world XY (metres) to (row, col). Row axis is y, column axis is x."""

    def __init__(self, x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX,
                 resolution=RESOLUTION):
        self.x_min, self.x_max = float(x_min), float(x_max)
        self.y_min, self.y_max = float(y_min), float(y_max)
        self.resolution = float(resolution)
        self.W = int(round((self.x_max - self.x_min) / self.resolution))
        self.H = int(round((self.y_max - self.y_min) / self.resolution))

    def world_to_cell(self, xy):
        """
        xy: (N, 2) world coords -> (rows, cols, valid), each (N,).

        Extent is half-open [min, max): a point on a cell boundary falls in the
        upper cell. `valid` is False for out-of-extent points; callers must mask
        with it -- a negative index would otherwise wrap to the opposite edge.
        """
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        cols = np.floor((xy[:, 0] - self.x_min) / self.resolution).astype(np.int64)
        rows = np.floor((xy[:, 1] - self.y_min) / self.resolution).astype(np.int64)
        valid = (cols >= 0) & (cols < self.W) & (rows >= 0) & (rows < self.H)
        return rows, cols, valid


def _to_loc(val):
    arr = np.asarray(val)
    if arr.dtype == object:
        return np.stack(arr.tolist()).astype(np.float64)
    return arr.astype(np.float64).reshape(-1, 2)


def _to_seq(val):
    return np.asarray(val, dtype=np.float64).ravel()


def _to_frames(val):
    return np.asarray(val, dtype=np.int64).ravel()


def _vehicle_stream(group_df):
    """Concatenate the vehicle track across rows and dedupe by frame.

    Every row of a (v_track_id, roi) group repeats the same vehicle track over
    that row's co-occurrence window, so duplicate frames carry identical
    positions and dropping them is lossless.
    """
    f = np.concatenate([_to_frames(r['frames']) for _, r in group_df.iterrows()])
    xy = np.vstack([_to_loc(r['v_loc_planar']) for _, r in group_df.iterrows()])
    sp = np.concatenate([_to_seq(r['v_speed']) for _, r in group_df.iterrows()])
    order = np.argsort(f, kind='stable')
    _, keep = np.unique(f[order], return_index=True)
    idx = order[keep]
    return f[idx], xy[idx], sp[idx]


def _ped_stream(group_df):
    """Flatten every (pedestrian, frame) observation in the group.

    One row per (vehicle, pedestrian) pair, so no pedestrian is split across
    rows and no deduping is needed.
    """
    f = np.concatenate([_to_frames(r['frames']) for _, r in group_df.iterrows()])
    xy = np.vstack([_to_loc(r['p_loc_planar']) for _, r in group_df.iterrows()])
    sp = np.concatenate([_to_seq(r['p_speed']) for _, r in group_df.iterrows()])
    return f, xy, sp


def _scatter(dst_count, dst_speed, grid, t_idx, xy, speed):
    rows, cols, valid = grid.world_to_cell(xy)
    m = valid & (t_idx >= 0)
    idx = (t_idx[m], rows[m], cols[m])
    np.add.at(dst_count, idx, 1.0)
    np.add.at(dst_speed, idx, speed[m])


def build_event_bev(group_df, grid=None, frames=None):
    """
    Rasterize one interaction event into per-timestep BEV maps.

    group_df: rows of one (v_track_id, roi) group.
    frames:   ascending frames to render; defaults to the event's unique frames.

    Returns (frames, bev) with bev of shape (T, 4, H, W), float32.

    Memory scales with T -- the longest events run to 9,000 frames (~690 MB at
    the default extent), so pass an explicit `frames` subset for those.
    """
    grid = grid or BEVGrid()

    v_f, v_xy, v_sp = _vehicle_stream(group_df)
    p_f, p_xy, p_sp = _ped_stream(group_df)

    if frames is None:
        frames = v_f
    frames = np.asarray(frames, dtype=np.int64).ravel()

    T = len(frames)
    bev = np.zeros((T, NUM_CHANNELS, grid.H, grid.W), dtype=np.float32)

    def time_index(f):
        """Position of each frame in `frames`, or -1 if not requested."""
        pos = np.searchsorted(frames, f)
        pos_c = np.clip(pos, 0, max(T - 1, 0))
        hit = (pos < T) & (frames[pos_c] == f) if T else np.zeros(len(f), bool)
        return np.where(hit, pos, -1)

    _scatter(bev[:, CH_VEHICLE_COUNT], bev[:, CH_VEHICLE_SPEED], grid,
             time_index(v_f), v_xy, v_sp)
    _scatter(bev[:, CH_PED_COUNT], bev[:, CH_PED_SPEED], grid,
             time_index(p_f), p_xy, p_sp)

    for ch_count, ch_speed in ((CH_VEHICLE_COUNT, CH_VEHICLE_SPEED),
                               (CH_PED_COUNT, CH_PED_SPEED)):
        count = bev[:, ch_count]
        occupied = count > 0
        bev[:, ch_speed][occupied] /= count[occupied]

    return frames, bev
