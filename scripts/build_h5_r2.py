"""Build the R2 representation h5: one event-static union crop per interaction.

R2 (docs/vision_representation_experiment_plan.md): window = union of the
subject-vehicle boxes and the top-1 pedestrian boxes over the 32-frame linspace
grid, padded 15%, squared, clamped to the frame; cropped at native resolution
and resized to --size. The crop window (tracking 1200x1100 space) is stored in
the dataset attrs so the loader can rasterize grounding masks in crop
coordinates. Optionally JPEG-encodes frames (vlen datasets) to fit storage.
"""
import argparse
import logging
import pickle
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from dataset.labels import parse_train_label
from dataset.tracking import group_grid_boxes, parse_tracking
from dataset.trajectory import build_group_trajectory

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ZIP_RANGES = {
    "29983897.zip": range(1, 41),
    "30050131.zip": range(41, 81),
    "30051331.zip": range(81, 121),
}
TRACK_W, TRACK_H = 1200, 1100  # tracking txt coordinate space
PAD_FRAC = 0.15


def union_window(v_boxes, p_boxes):
    """Event-static square-ish window from grid boxes (NaN rows = absent)."""
    boxes = np.concatenate([v_boxes, p_boxes.reshape(-1, 4)], axis=0)
    boxes = boxes[~np.isnan(boxes[:, 0])]
    if len(boxes) == 0:
        return np.array([0, 0, TRACK_W, TRACK_H], dtype=np.float32)
    x0, y0 = boxes[:, 0].min(), boxes[:, 1].min()
    x1, y1 = boxes[:, 2].max(), boxes[:, 3].max()
    side = max(x1 - x0, y1 - y0) * (1.0 + PAD_FRAC)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    # square window centered on the union, shifted (then clamped) into frame
    wx0 = np.clip(cx - side / 2, 0, max(TRACK_W - side, 0))
    wy0 = np.clip(cy - side / 2, 0, max(TRACK_H - side, 0))
    wx1 = min(wx0 + side, TRACK_W)
    wy1 = min(wy0 + side, TRACK_H)
    return np.array([wx0, wy0, wx1, wy1], dtype=np.float32)


def read_crops(cap, frame_indices, window, size, frame_wh):
    """Seek each grid frame, crop the (tracking-space) window, resize to size.

    Grid/tracking/parquet frame numbers are 1-based (1..N); OpenCV's
    CAP_PROP_POS_FRAMES is 0-based, so tracking frame k maps to video frame k-1.
    Converting here keeps the pixels aligned with the trajectory/boxes and avoids
    seeking one past the end when the grid reaches the final frame N.
    """
    fw, fh = frame_wh
    sx, sy = fw / TRACK_W, fh / TRACK_H
    x0, y0 = int(round(window[0] * sx)), int(round(window[1] * sy))
    x1, y1 = int(round(window[2] * sx)), int(round(window[3] * sy))
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
    out = np.zeros((len(frame_indices), size, size, 3), dtype=np.uint8)
    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(idx) - 1, 0))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        crop = frame[y0:y1, x0:x1]
        crop = cv2.resize(crop, (size, size))
        out[i] = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return out


def _labeled_keys():
    keys = set()
    for name in ("train_labels.pkl", "test_labels.pkl"):
        with open(DATA_DIR / "raw" / "labels" / name, "rb") as f:
            label_strings, _ = pickle.load(f)
        for s in label_strings:
            try:
                vid, tid, roi, _ = parse_train_label(s)
            except ValueError:
                continue
            keys.add(f"V{vid[-3:]}_{tid}_{roi}")
    return keys


def build(output, num_frames, size, video_start, video_end, jpeg, jpeg_quality,
          labeled_only, max_groups):
    parquet_dir = DATA_DIR / "processed" / "interactions"
    tracking_dir = DATA_DIR / "raw" / "tracking"
    allowed = _labeled_keys() if labeled_only else None
    if allowed:
        log.info("Restricting to %d labeled keys", len(allowed))
    output.parent.mkdir(parents=True, exist_ok=True)
    built = 0

    with h5py.File(output, "a") as h5f:
        for video_num in tqdm(range(video_start, video_end + 1), desc="Videos"):
            video_name = f"video_{video_num:03d}"
            parquet_path = parquet_dir / f"{video_name}_interactions.parquet"
            tracking_path = tracking_dir / f"{video_name}.txt"
            if not parquet_path.exists() or not tracking_path.exists():
                log.warning("Missing parquet/tracking for %s — skipping", video_name)
                continue
            zip_name = next((z for z, r in ZIP_RANGES.items() if video_num in r), None)
            if zip_name is None:
                log.warning("No zip for %s — skipping", video_name)
                continue

            df = pd.read_parquet(parquet_path)
            pending = []
            for (tid, roi), g in df.groupby(["v_track_id", "roi"]):
                key = f"V{video_num:03d}_{int(tid)}_{roi}"
                if key in h5f or (allowed is not None and key not in allowed):
                    continue
                pending.append((key, int(tid), roi, g))
            if not pending:
                continue

            track_frames = parse_tracking(tracking_path)
            avi_name = f"{video_name}.avi"
            with zipfile.ZipFile(DATA_DIR / zip_name) as zf:
                if avi_name not in zf.namelist():
                    log.warning("%s not in %s — skipping", avi_name, zip_name)
                    continue
                with tempfile.NamedTemporaryFile(suffix=".avi", delete=True) as tmp:
                    tmp.write(zf.read(avi_name))
                    tmp.flush()
                    cap = cv2.VideoCapture(tmp.name)
                    frame_wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    try:
                        for key, tid, roi, g in pending:
                            try:
                                _, _, _, ped_ids = build_group_trajectory(g, top_k=1)
                            except Exception as ex:
                                log.warning("No trajectory for %s: %s", key, ex)
                                continue
                            all_f = np.concatenate([np.asarray(f).ravel() for f in g["frames"]])
                            grid = np.linspace(int(all_f.min()), int(all_f.max()),
                                               num_frames, dtype=int)
                            v_boxes, p_boxes = group_grid_boxes(track_frames, grid, tid, ped_ids)
                            window = union_window(v_boxes, p_boxes)
                            tensor = read_crops(cap, grid, window, size, frame_wh)
                            if jpeg:
                                enc = [cv2.imencode(".jpg", f[:, :, ::-1],
                                                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])[1]
                                       for f in tensor]
                                ds = h5f.create_dataset(
                                    key, shape=(num_frames,),
                                    dtype=h5py.vlen_dtype(np.uint8))
                                for i, e in enumerate(enc):
                                    ds[i] = np.frombuffer(e.tobytes(), dtype=np.uint8)
                                ds.attrs["jpeg"] = True
                                ds.attrs["size"] = size
                            else:
                                ds = h5f.create_dataset(key, data=tensor)
                            ds.attrs["crop"] = window
                            built += 1
                            if max_groups and built >= max_groups:
                                log.info("Hit --max-groups=%d, stopping", max_groups)
                                return
                    finally:
                        cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/video/frames_r2.h5")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--video-start", type=int, default=1)
    parser.add_argument("--video-end", type=int, default=120)
    parser.add_argument("--jpeg", action="store_true",
                        help="store JPEG-encoded frames (vlen) instead of raw uint8")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--labeled-only", action="store_true",
                        help="build only keys present in train/test label pkls")
    parser.add_argument("--max-groups", type=int, default=0,
                        help="stop after N groups (smoke testing)")
    args = parser.parse_args()

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    build(output, args.num_frames, args.size, args.video_start, args.video_end,
          args.jpeg, args.jpeg_quality, args.labeled_only, args.max_groups)
    log.info("Done → %s", output)


if __name__ == "__main__":
    main()
