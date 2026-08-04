"""
Quadrant RGB crops from the raw frame database -- the VR (video representation)
stream of the Crosswalk paper.

The crop is not a bounding box around the event. It is one of four FIXED
rectangles, selected per event by (ROI x direction-of-travel), lifted verbatim
from crosswalk-original/preprocessing_vr.py:43-46 (rects) and :120-127
(selection). Direction comes from the `D` bit of the label string -- the bit
Crosswalk itself wrote -- so no tracking-box reparse is involved.

Why fixed rectangles beat a per-event bbox: the 2026-07-23 event-window audit
found our full-ROI bbox (1.18 M px, both ROIs concatenated) carried no ROI or
direction conditioning, while Crosswalk's quadrant (avg 276 k px) *is* the
conditioning -- picking the rectangle already encodes which crossing and which
approach the event belongs to. Measured: the quadrant bed probes to 0.4912 test
APv against 0.4401 for the union bed, same backbone and pooling.

What this module does NOT provide is grounding *within* the crop: a quadrant may
hold several vehicles and the crop cannot say which one is the key vehicle. That
is the BEV branch's job, and it is why the vision tokens stay spatial.
"""
import re

import cv2
import numpy as np

# crosswalk-original/preprocessing_vr.py:43-46  (x0, y0, x1, y1) in 1200x1100 frame coords
CROP_LB = (200, 380, 790, 950)
CROP_RB = (490, 250, 1040, 750)
CROP_LT = (20, 50, 580, 470)
CROP_RT = (480, 50, 1100, 470)

SIZE = 224
NUM_FRAMES = 32

# torchvision pretrained-weight statistics; the trunk was fitted under these.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_LABEL_RE = re.compile(r'V(\d+)I(\d+)S(\d)D(\d+)R\d+A(\d)')


def parse_label(label_string):
    """'V001I00002S1D0R0A1' -> (video_id, track_id, roi, downward, annotation).

    D0 = downward. annotation 0 = violation, 1 = compliance (inverted from the
    evaluation event schema -- see CLAUDE.md).
    """
    m = _LABEL_RE.match(label_string)
    if m is None:
        raise ValueError(f'unparseable label string: {label_string!r}')
    return (f'video_{int(m.group(1)):03d}',
            int(m.group(2)),
            'BOT' if m.group(3) == '1' else 'TOP',
            m.group(4) == '0',
            int(m.group(5)))


def quadrant_rect(roi, downward):
    """The fixed rectangle Crosswalk assigns to this (ROI, direction) pair."""
    if roi == 'TOP':
        return CROP_RT if downward else CROP_LT
    if roi == 'BOT':
        return CROP_RB if downward else CROP_LB
    raise ValueError(f"roi must be 'TOP' or 'BOT', got {roi!r}")


def event_frame_grid(group_df, num_frames=NUM_FRAMES):
    """Frame numbers to sample for one (v_track_id, roi) group.

    Uniform over the event's full span. Frames repeat when the span is shorter
    than num_frames -- the clip stays fixed-length rather than being padded, so
    a short event is a slow-motion clip, never a clip with black slots.
    """
    all_f = np.concatenate([np.asarray(f).ravel() for f in group_df['frames']])
    return np.linspace(int(all_f.min()), int(all_f.max()), num_frames, dtype=np.int64)


def crop_clip(frames_ds, frame_grid, rect, size=SIZE, normalize=True):
    """
    Decode and crop one clip.

    frames_ds:  h5 dataset of JPEG-encoded full frames for one video, indexed by
                (frame_number - 1); the DB is 1-based, Python is 0-based.
    Returns (T, 3, size, size) float32, RGB, ImageNet-normalized when
    `normalize` (the trunk's input contract) else in [0, 1].

    Out-of-range frame numbers are clamped to the video rather than raising: a
    handful of parquet groups name a frame one past the last decoded one, and a
    clamped duplicate is a truer clip than a black frame.
    """
    x0, y0, x1, y1 = rect
    n = frames_ds.shape[0]
    out = np.empty((len(frame_grid), size, size, 3), dtype=np.float32)
    for i, f in enumerate(frame_grid):
        idx = min(max(int(f) - 1, 0), n - 1)
        img = cv2.imdecode(frames_ds[idx], cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f'frame {idx} failed to decode')
        img = img[y0:y1, x0:x1, ::-1]                       # crop, BGR->RGB
        out[i] = cv2.resize(np.ascontiguousarray(img), (size, size))
    out /= 255.0
    if normalize:
        out = (out - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(out.transpose(0, 3, 1, 2))  # (T, 3, size, size)


def imagenet_denormalize(x):
    """Inverse of the normalization in crop_clip, for visual inspection."""
    arr = np.asarray(x).transpose(0, 2, 3, 1)
    return np.clip(arr * IMAGENET_STD + IMAGENET_MEAN, 0.0, 1.0)
