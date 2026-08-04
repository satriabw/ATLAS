"""
The fixed geometry linking a BEV window to its quadrant crop.

The camera is static across all 120 videos and the four crop rectangles are
fixed, so *every quantity in this module is a constant*. It is computed once
from the calibration and cached, never learned. That is the whole argument for
the BEV branch: a rasterised motion map can be put in correspondence with image
features, which a GRU hidden state cannot -- there is nowhere in a GRU state to
attach an image region.

Nothing here assumes the two grids are aligned. The correspondence is built by
pushing each BEV cell through the calibration, so rotation, perspective, scale
and the non-road-aligned calibration axes (the intersection arms run at roughly
20 and 70 degrees) are already inside it. A naive (row, col) <-> (row, col)
scheme would be destroyed by that geometry; this is not that scheme.

Constants below were chosen by measurement on TRAIN labels only (choosing them
on test data would be leakage). Full tables in
artifacts/docs/2026-08-04_bev_vision_fusion/plan.md.

WINDOW_M = 16
    The quadrant crops project to ground footprints of 15.0-17.3 m per side, so
    the window is sized to the region the camera actually sees. Larger windows
    do NOT buy paired tokens -- the count of BEV tokens visible in the crop
    saturates around 10 (at 4 m tokens) whether the window is 16, 20 or 24 m,
    because the extra area lies outside the camera's view of that quadrant.
    What changes is the *fraction* paired: 59% at 16 m, 39% at 20 m, 27% at
    24 m. Cost of 16 m over 20 m: vehicle fully inside on 97.4% of events
    rather than 99.4% (mean containment 0.991 vs 0.997).

    Known confound, recorded rather than solved: clipping is weakly
    label-correlated. Compliant vehicles are clipped slightly more than
    violating ones -- 98.2% vs 97.1% fully inside at 16 m (gap +1.04 pp,
    permutation p = 0.086) and 99.9% vs 99.3% at 20 m (+0.62 pp, p = 0.041).
    The effect is ~1 pp, present at every window size, and therefore inherent
    to windowing rather than to this choice of window. S2 reports APv split by
    clipped/unclipped so we would see the model exploiting it.

RESOLUTION = 0.5
    Inherited from dataset.bev so the two rasterisers agree.

CELLS = 32
    16 m / 0.5 m. Divisible by 4, so two pooling stages land exactly on 8x8 and
    the ragged-edge machinery dataset.bev needs (ceil_mode, count_include_pad)
    has nothing to do here -- two design decisions disappear.

HEIGHT_M = 2.0, Z_UP = -1.0
    The correspondence maps a *column*, not a ground point. A ground-anchored
    mapping points at an object's contact patch while its informative
    appearance sits above: measured, a 1.5 m object appears 70 px higher, which
    is 0.88-1.17 vision tokens. That offset is near-constant across a quadrant
    (mean 70.0, max 70.9 px), so it is a fixed shift -- exactly the kind of
    thing to correct analytically rather than spend learned capacity on. 2.0 m
    covers cars and pedestrians; a bus or truck exceeds it.

    **Up is -Z in this calibration.** The camera centre -R' t sits at world
    Z = -89.5, so the camera occupies the negative half-space and a raised
    point must move toward it. Measured: z = -1.5 moves a pixel up by 70 px,
    z = +1.5 moves it *down* by 70. Getting this backwards is silent -- the
    column is still the right length and M still has the right shape, it just
    reaches into the road surface instead of the object. Guarded by
    test_raising_z_moves_the_pixel_up_not_down and T5b.
"""
import functools
from pathlib import Path

import cv2
import numpy as np

from .vision_crop import quadrant_rect

CAMERA_YML = Path(__file__).with_name('camera_model.yml')

WINDOW_M = 16.0
RESOLUTION = 0.5
CELLS = int(round(WINDOW_M / RESOLUTION))
HEIGHT_M = 2.0
Z_UP = -1.0            # up is -Z here; see the module docstring
Z_SAMPLES = 5          # column samples; 5 puts one every 0.5 m over 2 m
VISION_GRID = 7        # VisionEncoder emits a 7x7 token grid


@functools.lru_cache(maxsize=1)
def _projection():
    """(P, H_ground_inv) from the calibration.

    P is the full 3x4 camera matrix, used for column points off the ground.
    H_ground = K [r1 r2 t] maps (X, Y, 1) on the Z=0 plane to pixels; its
    inverse is the image -> world direction. dist_coeffs are all zero in this
    calibration, so there is no distortion term to carry.
    """
    fs = cv2.FileStorage(str(CAMERA_YML), cv2.FILE_STORAGE_READ)
    K = fs.getNode('camera_matrix').mat().astype(np.float64)
    R = fs.getNode('rot_matrix').mat().astype(np.float64)
    t = fs.getNode('tvec').mat().astype(np.float64).reshape(3, 1)
    fs.release()
    P = K @ np.hstack([R, t])
    H_ground = K @ np.hstack([R[:, 0:1], R[:, 1:2], t])
    return P, np.linalg.inv(H_ground)


def world_to_image(xyz):
    """(N, 3) world metres -> (N, 2) pixels. Z is real, not assumed zero."""
    P, _ = _projection()
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    uv = P @ np.hstack([xyz, np.ones((len(xyz), 1))]).T
    return (uv[:2] / uv[2]).T


def image_to_world(uv):
    """(N, 2) pixels -> (N, 2) world metres on the ground plane (Z = 0)."""
    _, H_inv = _projection()
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    w = H_inv @ np.vstack([uv.T, np.ones(len(uv))])
    return (w[:2] / w[2]).T


@functools.lru_cache(maxsize=8)
def quadrant_window(roi, downward):
    """(x_min, y_min) of the BEV window for this quadrant, in world metres.

    Centred on the ground footprint of the crop rectangle. The footprint is a
    perspective quadrilateral, so it is sampled over the whole rect rather than
    at the four corners, which would understate it.
    """
    x0, y0, x1, y1 = quadrant_rect(roi, downward)
    u, v = np.meshgrid(np.linspace(x0, x1, 40), np.linspace(y0, y1, 40))
    w = image_to_world(np.stack([u.ravel(), v.ravel()], axis=1))
    cx = (w[:, 0].min() + w[:, 0].max()) / 2.0
    cy = (w[:, 1].min() + w[:, 1].max()) / 2.0
    return cx - WINDOW_M / 2.0, cy - WINDOW_M / 2.0


def cell_centres(roi, downward):
    """(CELLS*CELLS, 2) world centres, row-major, row axis = y (matches BEVGrid)."""
    x_min, y_min = quadrant_window(roi, downward)
    xs = x_min + (np.arange(CELLS) + 0.5) * RESOLUTION
    ys = y_min + (np.arange(CELLS) + 0.5) * RESOLUTION
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


@functools.lru_cache(maxsize=8)
def correspondence(roi, downward, bev_grid):
    """Fixed M, shape (bev_grid**2, VISION_GRID**2), rows summing to 1 or 0.

    M[i, j] is the share of BEV token i's column samples that land in vision
    token j. A row is all-zero when the token is not visible in the crop at
    all; that is left explicit rather than smoothed, so a caller can tell
    "no visual partner" from "uniformly attended". Measured at the shipped
    16 m / 8x8 setting: 70.3-73.4% of tokens paired, fan-out 4.81-5.40 of 49
    vision tokens, vertical footprint 2.55-2.81 token rows (the column
    contributes ~1.3 and the token's own 2 m of ground depth the rest).

    Row-normalised so the resampled vision feature is a weighted average, on
    the same scale as the vision tokens themselves whatever the fan-out.
    """
    if CELLS % bev_grid:
        raise ValueError(f'bev_grid {bev_grid} must divide {CELLS} cells')
    x0, y0, x1, y1 = quadrant_rect(roi, downward)
    centres = cell_centres(roi, downward)

    block = CELLS // bev_grid
    rows = np.repeat(np.arange(bev_grid), block)
    bev_tok = (rows[:, None] * bev_grid + rows[None, :]).ravel()

    M = np.zeros((bev_grid * bev_grid, VISION_GRID * VISION_GRID), dtype=np.float32)
    for z in np.linspace(0.0, Z_UP * HEIGHT_M, Z_SAMPLES):
        uv = world_to_image(np.hstack([centres, np.full((len(centres), 1), z)]))
        inside = (uv[:, 0] >= x0) & (uv[:, 0] < x1) & (uv[:, 1] >= y0) & (uv[:, 1] < y1)
        if not inside.any():
            continue
        # crop is resized to a square before tokenising, so normalise by rect size
        tu = ((uv[inside, 0] - x0) / (x1 - x0) * VISION_GRID).astype(int)
        tv = ((uv[inside, 1] - y0) / (y1 - y0) * VISION_GRID).astype(int)
        vis_tok = np.clip(tv, 0, VISION_GRID - 1) * VISION_GRID + np.clip(tu, 0, VISION_GRID - 1)
        np.add.at(M, (bev_tok[inside], vis_tok), 1.0)

    total = M.sum(axis=1, keepdims=True)
    return np.divide(M, total, out=np.zeros_like(M), where=total > 0)
