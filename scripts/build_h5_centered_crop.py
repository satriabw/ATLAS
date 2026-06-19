"""Build a centered-window, per-frame TIGHT-CROP vision h5 for one target type.

Two experiment arms (docs/2026-06-18_centered_crop_vision/plan.md):
  --target vehicle : per-frame tight square crop on the subject-vehicle bbox.
  --target ped     : per-frame tight square crop on the bbox of the single
                     pedestrian that achieves the global-min vehicle↔ped
                     distance (the one that defines the centre frame).

Temporal grid is taken DIRECTLY from dataset.centered_window.build_centered_window
(half=32 → 64 slots), so the clip is 100% temporally aligned with a trajectory
`train_centered.py --window 64` run: same centre frame (global-min ped distance),
same per-slot video frame numbers, same zero-padded edges. A slot is BLACK when
the trajectory slot is padded (frame=-1) OR the target has no tracking box at that
frame ("crop only available target in given frames").

Frames are read from the master frame DB (data/raw/video/frames_db.h5, native
1200x1100 == tracking coordinate space, so boxes map 1:1 with no rescale). Output
is keyed `Vxxx_tid_ROI` with JPEG-encoded vlen frames + a `crop` attr, so the
untouched load_frames_h5 R2 branch serves them as-is (no ROI polygon, no resize).
"""
import argparse
import logging
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from dataset.centered_window import build_centered_window  # noqa: E402
from dataset.tracking import parse_tracking, group_grid_boxes  # noqa: E402
from dataset.trajectory import DEFAULT_TOP_K, _extract_row_arrays  # noqa: E402
from build_h5_r2 import _labeled_keys, union_window  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TRACK_W, TRACK_H = 1200, 1100
HALF = 32  # ±32 → 64-frame window; matches trajectory centered window half=32


def _closest_ped(group_df, top_ped_ids):
    """Pedestrian id achieving the global-min vehicle↔ped distance (defines centre)."""
    best_pid, best_d = None, np.inf
    for pid in top_ped_ids:
        sub = group_df[group_df["p_track_id"] == pid]
        if len(sub) == 0:
            continue
        _, v_loc_k, _, p_loc_k, _ = _extract_row_arrays(sub)
        d = float(np.linalg.norm(p_loc_k - v_loc_k, axis=1).min())
        if d < best_d:
            best_d, best_pid = d, int(pid)
    return best_pid


def _tight_square(box, pad):
    """Square window around a tracking-space box (x0,y0,x1,y1), padded, clamped."""
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) * (1.0 + pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sx0 = np.clip(cx - side / 2, 0, max(TRACK_W - side, 0))
    sy0 = np.clip(cy - side / 2, 0, max(TRACK_H - side, 0))
    sx1 = min(sx0 + side, TRACK_W)
    sy1 = min(sy0 + side, TRACK_H)
    return int(round(sx0)), int(round(sy0)), int(round(sx1)), int(round(sy1))


def _decode(video_ds, frame_1based):
    """Native BGR frame from the master DB (1-based tracking frame → 0-based index)."""
    buf = video_ds[max(int(frame_1based) - 1, 0)]
    return cv2.imdecode(np.frombuffer(buf.tobytes(), np.uint8), cv2.IMREAD_COLOR)


def _build_clip(frames, track_frames, target_id, size, pad, decoded):
    """(64, size, size, 3) RGB uint8; black where padded or target box absent.

    `decoded` maps frame number → native BGR image (decoded once per video frame).
    """
    out = np.zeros((len(frames), size, size, 3), dtype=np.uint8)
    n_valid = 0
    for k, f in enumerate(frames):
        if f < 0:
            continue
        box = track_frames.get(int(f), {}).get(target_id)
        if box is None:
            continue
        x0, y0, x1, y1 = _tight_square(box, pad)
        crop = decoded[int(f)][y0:y1, x0:x1]
        if crop.size == 0:
            continue
        out[k] = cv2.cvtColor(cv2.resize(crop, (size, size)), cv2.COLOR_BGR2RGB)
        n_valid += 1
    return out, n_valid


def _build_union_clip(frames, window, size, decoded):
    """Event-static union window cropped from every non-padded slot (R2 framing)."""
    x0, y0, x1, y1 = (int(round(v)) for v in window)
    out = np.zeros((len(frames), size, size, 3), dtype=np.uint8)
    n_valid = 0
    for k, f in enumerate(frames):
        if f < 0:
            continue
        crop = decoded[int(f)][y0:y1, x0:x1]
        if crop.size == 0:
            continue
        out[k] = cv2.cvtColor(cv2.resize(crop, (size, size)), cv2.COLOR_BGR2RGB)
        n_valid += 1
    return out, n_valid


def build(output, target, video_start, video_end, size, pad, jpeg_quality,
          labeled_only, max_groups):
    parquet_dir = DATA_DIR / "processed" / "interactions"
    tracking_dir = DATA_DIR / "raw" / "tracking"
    master_path = DATA_DIR / "raw" / "video" / "frames_db.h5"
    allowed = _labeled_keys() if labeled_only else None
    if allowed:
        log.info("Restricting to %d labeled keys", len(allowed))
    output.parent.mkdir(parents=True, exist_ok=True)
    built = 0

    with h5py.File(master_path, "r") as master, h5py.File(output, "a") as h5f:
        for vnum in tqdm(range(video_start, video_end + 1), desc="Videos"):
            video_name = f"video_{vnum:03d}"
            parquet_path = parquet_dir / f"{video_name}_interactions.parquet"
            tracking_path = tracking_dir / f"{video_name}.txt"
            master_key = video_name
            if not parquet_path.exists() or not tracking_path.exists():
                log.warning("Missing parquet/tracking for %s — skip", video_name)
                continue
            if master_key not in master:
                log.warning("%s not in master DB — skip", video_name)
                continue

            df = pd.read_parquet(parquet_path)
            pending = []
            for (tid, roi), g in df.groupby(["v_track_id", "roi"]):
                key = f"V{vnum:03d}_{int(tid)}_{roi}"
                if key in h5f or (allowed is not None and key not in allowed):
                    continue
                pending.append((key, int(tid), roi, g))
            if not pending:
                continue

            track_frames = parse_tracking(tracking_path)
            video_ds = master[master_key]

            for key, tid, roi, g in pending:
                try:
                    w = build_centered_window(g, top_k=DEFAULT_TOP_K, half=HALF)
                except Exception as ex:
                    log.warning("No centered window for %s: %s", key, ex)
                    continue
                frames = w["frames"]
                if target == "ped":
                    target_id = _closest_ped(g, w["ped_ids"])
                    if target_id is None:
                        log.warning("No target pedestrian for %s — skip", key)
                        continue
                else:
                    target_id = tid  # vehicle arm + union arm record the vehicle id

                # decode each needed frame once (per-group cache bounds memory to ~64 frames)
                decoded = {int(f): _decode(video_ds, f) for f in set(frames.tolist()) if f >= 0}

                if target == "union":
                    valid = frames[frames >= 0]
                    v_boxes, p_boxes = group_grid_boxes(track_frames, valid, tid, [w["ped_ids"][0]])
                    window = union_window(v_boxes, p_boxes)
                    clip, n_valid = _build_union_clip(frames, window, size, decoded)
                else:
                    clip, n_valid = _build_clip(frames, track_frames, target_id, size, pad, decoded)
                if n_valid == 0:
                    log.warning("%s: 0 valid frames (target %s) — skip", key, target_id)
                    continue

                enc = [cv2.imencode(".jpg", c[:, :, ::-1],
                                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])[1]
                       for c in clip]
                ds = h5f.create_dataset(key, shape=(len(clip),),
                                        dtype=h5py.vlen_dtype(np.uint8))
                for i, e in enumerate(enc):
                    ds[i] = np.frombuffer(e.tobytes(), dtype=np.uint8)
                ds.attrs["jpeg"] = True
                ds.attrs["size"] = size
                ds.attrs["crop"] = np.array([0, 0, size, size], dtype=np.float32)
                ds.attrs["centre_frame"] = int(w["centre_frame"])
                ds.attrs["target_id"] = int(target_id)
                ds.attrs["n_valid"] = int(n_valid)
                built += 1
                if max_groups and built >= max_groups:
                    log.info("Hit --max-groups=%d, stopping", max_groups)
                    return
    log.info("=== built %d groups → %s ===", built, output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["vehicle", "ped", "union"], required=True)
    p.add_argument("--output", default=None,
                   help="default: data/raw/video/frames_{target}_centered.h5")
    p.add_argument("--video-start", type=int, default=1)
    p.add_argument("--video-end", type=int, default=120)
    p.add_argument("--size", type=int, default=112)
    p.add_argument("--pad", type=float, default=0.15)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--labeled-only", action="store_true")
    p.add_argument("--max-groups", type=int, default=0)
    args = p.parse_args()

    output = (Path(args.output) if args.output else
              DATA_DIR / "raw" / "video" / f"frames_{args.target}_centered.h5")
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    build(output, args.target, args.video_start, args.video_end, args.size,
          args.pad, args.jpeg_quality, args.labeled_only, args.max_groups)


if __name__ == "__main__":
    main()
