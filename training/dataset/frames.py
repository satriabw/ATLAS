import cv2
import h5py
import numpy as np
import torch


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Fixed crosswalk ROI polygons in the original 1200x1100 video frame
# (single static-camera scene shared by all videos). Source: crosswalk-original/filtering_2.py.
ORIG_W, ORIG_H = 1200, 1100
ROI_POLYS = {
    'TOP': np.array([[0, 120], [1200, 120], [1200, 480], [0, 480]], dtype=np.float32),
    'BOT': np.array([[120, 480], [1100, 240], [1100, 600], [240, 1000]], dtype=np.float32),
}


def _roi_poly_bbox(roi: str, h: int, w: int):
    poly = ROI_POLYS[roi].copy()
    poly[:, 0] *= w / ORIG_W
    poly[:, 1] *= h / ORIG_H
    poly = poly.astype(np.int32)
    x0, y0 = poly.min(0)
    x1, y1 = poly.max(0)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    return mask, (x0, y0, x1, y1)


def _box_masks(num: int, h: int, w: int, v_boxes: np.ndarray, p_boxes: np.ndarray,
               window=None) -> np.ndarray:
    # Rasterize tracking boxes (1200x1100 image space) onto the frame grid as
    # two binary channels: 0 = subject vehicle, 1 = union of top-K pedestrians.
    # window: (x0, y0, x1, y1) crop in tracking space (R2 entries) — boxes are
    # expressed in crop coordinates; defaults to the full frame.
    masks = np.zeros((num, h, w, 2), dtype=np.uint8)
    wx0, wy0, wx1, wy1 = window if window is not None else (0.0, 0.0, float(ORIG_W), float(ORIG_H))
    sx, sy = w / (wx1 - wx0), h / (wy1 - wy0)

    def fill(f, ch, box):
        if np.isnan(box[0]):
            return
        x0 = max(int(round((box[0] - wx0) * sx)), 0)
        y0 = max(int(round((box[1] - wy0) * sy)), 0)
        x1 = min(int(round((box[2] - wx0) * sx)), w)
        y1 = min(int(round((box[3] - wy0) * sy)), h)
        if x1 > x0 and y1 > y0:
            masks[f, y0:y1, x0:x1, ch] = 1

    for f in range(num):
        fill(f, 0, v_boxes[f])
        for pb in p_boxes:
            fill(f, 1, pb[f])
    return masks


def load_frames_h5(h5_file: h5py.File, key: str, num_frames: int, roi: str = None, size: int = 224,
                   boxes=None) -> torch.Tensor:
    if key not in h5_file:
        raise KeyError(key)

    ds = h5_file[key]
    crop = ds.attrs.get('crop')  # R2 entries: event-static union window in tracking space
    if ds.attrs.get('jpeg', False):
        arr = np.stack([cv2.imdecode(b, cv2.IMREAD_COLOR)[:, :, ::-1] for b in ds[:]])
    else:
        arr = ds[:]  # (32, H, W, 3) uint8
    frames = arr[:num_frames]  # (num_frames, H, W, 3) => Since h5 already sampled to 32 frames, this should be a no-op but we slice just in case.
    masks = None
    if crop is not None:
        # R2: frames are already the per-event crop at build resolution — no ROI
        # polygon, no resize; grounding masks rasterized in crop coordinates.
        if boxes is not None:
            v_boxes, p_boxes = boxes
            masks = _box_masks(len(frames), frames.shape[1], frames.shape[2],
                               v_boxes, p_boxes, window=tuple(crop))
    else:
        if boxes is not None:
            v_boxes, p_boxes = boxes
            masks = _box_masks(len(frames), frames.shape[1], frames.shape[2], v_boxes, p_boxes)
        if roi is not None:
            poly_mask, (x0, y0, x1, y1) = _roi_poly_bbox(roi, frames.shape[1], frames.shape[2])
            frames = (frames * poly_mask[None, :, :, None])[:, y0:y1, x0:x1]
            if masks is not None:
                # Crop only — don't zero boxes outside the polygon, so the subject
                # stays grounded while approaching the crosswalk.
                masks = masks[:, y0:y1, x0:x1]
        if size != frames.shape[1] or size != frames.shape[2]:
            frames = np.stack([cv2.resize(f, (size, size)) for f in frames])
            if masks is not None:
                masks = np.stack([
                    cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST) for m in masks
                ])

    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    if masks is not None:
        tensor = torch.cat([tensor, torch.from_numpy(masks).permute(0, 3, 1, 2).float()], dim=1)
    return tensor
