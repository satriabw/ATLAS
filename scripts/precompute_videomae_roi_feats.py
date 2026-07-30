"""Precompute FROZEN pooled VideoMAE features over the fixed crosswalk ROIs
(Stage A, plan = ~/.claude/plans/dazzling-bouncing-summit.md).

For each labeled interaction event (V{vid}_{tid}_{roi}) this crops BOTH the TOP
and BOT crosswalk ROI regions (fixed polygons in `dataset.frames.ROI_POLYS`) out
of a 16-frame linspace clip, runs each crop through a frozen
`videomae-base-finetuned-kinetics` encoder, mean-pools the token sequence to a
768-d vector, and stores the concatenation [top(768) ; bot(768)] → (1536,) float32
per key.

Frame source is `data/raw/video/frames_db.h5` (full-frame native-res JPEG DB) so
no raw video / zip access is needed. Event frame span comes from the parquet group
`(v_track_id, roi)`, matching how build_h5_r2.py samples the grid.

Output: one (1536,) dataset per key → data/raw/video/videomae_roi_feats.h5
"""
import argparse
import logging
import pickle
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm
from transformers import VideoMAEModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from dataset.labels import parse_train_label          # noqa: E402
from dataset.frames import _roi_poly_bbox, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
NUM_FRAMES = 16
SIZE = 224
ROIS = ("TOP", "BOT")
_MEAN = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)


def _labeled_keys():
    """{key: (video_num, tid, roi)} over both train and test label pkls."""
    keys = {}
    for name in ("train_labels.pkl", "test_labels.pkl"):
        with open(DATA_DIR / "raw" / "labels" / name, "rb") as f:
            label_strings, _ = pickle.load(f)
        for s in label_strings:
            try:
                vid, tid, roi, _ = parse_train_label(s)
            except ValueError:
                continue
            keys[f"V{vid[-3:]}_{tid}_{roi}"] = (int(vid[-3:]), tid, roi)
    return keys


def _clip_from_db(frames_ds, grid):
    """Decode the linspace grid frames (1-based) from the full-frame JPEG DB."""
    out = []
    n = frames_ds.shape[0]
    for k in grid:
        idx = min(max(int(k) - 1, 0), n - 1)  # 1-based frame k → 0-based db index
        img = cv2.imdecode(frames_ds[idx], cv2.IMREAD_COLOR)  # BGR, native res
        out.append(img[:, :, ::-1])  # → RGB
    return out  # list of (H, W, 3) uint8


def _roi_clip(rgb_frames, roi):
    """Bbox-crop one ROI region from each frame, resize to SIZE, ImageNet-normalize.
    Returns (NUM_FRAMES, 3, SIZE, SIZE) float tensor."""
    h, w = rgb_frames[0].shape[:2]
    _, (x0, y0, x1, y1) = _roi_poly_bbox(roi, h, w)
    crops = [cv2.resize(f[y0:y1, x0:x1], (SIZE, SIZE)) for f in rgb_frames]
    t = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float() / 255.0
    return (t - _MEAN) / _STD


@torch.no_grad()
def _pool(model, clips, device):
    """clips: (B, NUM_FRAMES, 3, SIZE, SIZE) → mean-pooled (B, 768)."""
    out = model(pixel_values=clips.to(device)).last_hidden_state  # (B, tokens, 768)
    return out.mean(dim=1).cpu().numpy().astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/raw/video/videomae_roi_feats.h5")
    p.add_argument("--frames-db", default="data/raw/video/frames_db.h5")
    p.add_argument("--batch-clips", type=int, default=16,
                   help="clips per VideoMAE forward (2 clips = 1 event)")
    p.add_argument("--limit", type=int, default=None, help="smoke: stop after N events")
    args = p.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    db_path = Path(args.frames_db)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoMAEModel.from_pretrained(MODEL_ID).eval().to(device)
    for prm in model.parameters():
        prm.requires_grad = False
    log.info("Loaded frozen %s on %s", MODEL_ID, device)

    import pandas as pd
    parquet_dir = DATA_DIR / "processed" / "interactions"
    keys = _labeled_keys()
    log.info("%d labeled event keys", len(keys))

    # group wanted keys by video
    by_video = {}
    for key, (vnum, tid, roi) in keys.items():
        by_video.setdefault(vnum, []).append((key, tid, roi))

    written, done_events = set(), 0
    # batch buffers: parallel lists of (key, roi) and their clip tensors
    buf_meta, buf_clips = [], []
    pending_feats = {}  # key -> {roi: vec}

    with h5py.File(db_path, "r") as db, h5py.File(out_path, "w") as fout:

        def flush():
            if not buf_clips:
                return
            clips = torch.stack(buf_clips)
            vecs = _pool(model, clips, device)
            for (key, roi), vec in zip(buf_meta, vecs):
                pending_feats.setdefault(key, {})[roi] = vec
                if len(pending_feats[key]) == len(ROIS):
                    cat = np.concatenate([pending_feats[key][r] for r in ROIS])
                    fout.create_dataset(key, data=cat)
                    written.add(key)
                    del pending_feats[key]
            buf_meta.clear()
            buf_clips.clear()

        for vnum in tqdm(sorted(by_video), desc="Videos"):
            video_name = f"video_{vnum:03d}"
            parquet_path = parquet_dir / f"{video_name}_interactions.parquet"
            if video_name not in db or not parquet_path.exists():
                log.warning("Missing frames_db/parquet for %s — skipping", video_name)
                continue
            df = pd.read_parquet(parquet_path)
            frames_ds = db[video_name]
            groups = {(int(t), str(r)): g for (t, r), g in df.groupby(["v_track_id", "roi"])}

            for key, tid, roi in by_video[vnum]:
                if key in written:
                    continue
                g = groups.get((tid, roi))
                if g is None:
                    continue  # labeled event has no parquet group
                all_f = np.concatenate([np.asarray(f).ravel() for f in g["frames"]])
                grid = np.linspace(int(all_f.min()), int(all_f.max()), NUM_FRAMES, dtype=int)
                rgb = _clip_from_db(frames_ds, grid)
                for r in ROIS:
                    buf_meta.append((key, r))
                    buf_clips.append(_roi_clip(rgb, r))
                    if len(buf_clips) >= args.batch_clips:
                        flush()
                done_events += 1
                if args.limit and done_events >= args.limit:
                    break
            if args.limit and done_events >= args.limit:
                break
        flush()

    log.info("DONE — %d keys written → %s", len(written), out_path)


if __name__ == "__main__":
    main()
