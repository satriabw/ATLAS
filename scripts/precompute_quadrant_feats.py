"""Precompute FROZEN Kinetics r2plus1d features over Crosswalk's QUADRANT crop.

The event-window audit (2026-07-23) localized our divergence from the Crosswalk paper to
the crop: they select 1-of-4 fixed quadrants per event by (ROI x direction-of-travel),
avg ~276k px; our union-crop bed used the full ROI bbox with no direction conditioning.
This builds the reference-faithful quadrant crop and extracts the SAME frozen-Kinetics
r2plus1d feature (112px, Kinetics renorm, mean-pool) used by r2_whole_feats_kinetics.h5,
so the ONLY change vs that leakage-free bed is the crop region.

Quadrant rectangles + selection are lifted verbatim from Crosswalk/preprocessing_vr.py.
ROI (region_top) and direction (downward) come straight from the label string
`V{vid}I{tid}S{roi}D{dir}...` — the exact bits Crosswalk itself wrote (D0=downward) — so no
tracking-box reparse is needed. Output keyed V{vid}_{tid}_{roi} like the other feature h5s.
"""
import logging
import pickle
import re
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'training'))
sys.path.insert(0, str(ROOT / 'scripts'))
from precompute_r2plus1d_feats import _encoder, _extract   # reuse frozen-Kinetics extraction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Crosswalk/preprocessing_vr.py:43-46  (x0, y0, x1, y1)
CROP_LB, CROP_RB = (200, 380, 790, 950), (490, 250, 1040, 750)
CROP_LT, CROP_RT = (20, 50, 580, 470), (480, 50, 1100, 470)
NUM_FRAMES, SIZE = 32, 112
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _rect(roi, downward):
    # Crosswalk/preprocessing_vr.py:120-127
    if roi == 'TOP':
        return CROP_RT if downward else CROP_LT
    return CROP_RB if downward else CROP_LB


def parse_labels():
    """(vid,tid,roi) -> downward(bool), over train+test label pkls. D0 = downward."""
    out = {}
    for split in ('train', 'test'):
        strs, _ = pickle.load(open(ROOT / f'data/raw/labels/{split}_labels.pkl', 'rb'))
        for s in strs:
            m = re.match(r'V(\d+)I(\d+)S(\d)D(\d+)R\d+A(\d)', s)
            if not m:
                continue
            vid = f"video_{int(m.group(1)):03d}"
            roi = 'BOT' if m.group(3) == '1' else 'TOP'
            out[(vid, int(m.group(2)), roi)] = (m.group(4) == '0')
    return out


def _clip(frames_ds, grid, rect):
    x0, y0, x1, y1 = rect
    n = frames_ds.shape[0]
    crops = []
    for k in grid:
        idx = min(max(int(k) - 1, 0), n - 1)
        img = cv2.imdecode(frames_ds[idx], cv2.IMREAD_COLOR)[:, :, ::-1]   # BGR→RGB
        crops.append(cv2.resize(np.ascontiguousarray(img[y0:y1, x0:x1]), (SIZE, SIZE)))
    t = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float() / 255.0
    return (t - _MEAN) / _STD                                              # (F,3,SIZE,SIZE)


def main():
    out_path = ROOT / 'data/raw/video/r2_quadrant_feats_kinetics.h5'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = _encoder(None, device)   # frozen Kinetics-400 r2plus1d, leakage-free
    log.info("frozen Kinetics r2plus1d on %s", device)

    labels = parse_labels()
    nd = sum(labels.values()); log.info("%d labeled events (downward=%d, up=%d)",
                                        len(labels), nd, len(labels) - nd)
    by_video = {}
    for (vid, tid, roi), down in labels.items():
        by_video.setdefault(vid, []).append((tid, roi, down))

    pq_dir = ROOT / 'data/processed/interactions'
    written, buf_keys, buf_clips = set(), [], []
    with h5py.File(ROOT / 'data/raw/video/frames_db.h5', 'r') as db, \
         h5py.File(out_path, 'w') as fout:

        def flush():
            if not buf_clips:
                return
            vecs = _extract(enc, torch.stack(buf_clips).to(device)).cpu().numpy().astype(np.float32)
            for k, v in zip(buf_keys, vecs):
                fout.create_dataset(k, data=v); written.add(k)
            buf_keys.clear(); buf_clips.clear()

        for vid in sorted(by_video):
            pq = pq_dir / f'{vid}_interactions.parquet'
            if not pq.exists() or vid not in db:
                continue
            df = pd.read_parquet(pq)
            groups = {(int(t), str(r)): g for (t, r), g in df.groupby(['v_track_id', 'roi'])}
            frames_ds = db[vid]
            for tid, roi, down in by_video[vid]:
                key = f"V{vid[-3:]}_{tid}_{roi}"
                if key in written:
                    continue
                g = groups.get((tid, roi))
                if g is None:
                    continue
                all_f = np.concatenate([np.asarray(f).ravel() for f in g['frames']])
                grid = np.linspace(int(all_f.min()), int(all_f.max()), NUM_FRAMES, dtype=int)
                buf_keys.append(key); buf_clips.append(_clip(frames_ds, grid, _rect(roi, down)))
                if len(buf_clips) >= 8:
                    flush()
                    if len(written) % 500 < 8:
                        log.info("  %d keys written", len(written))
        flush()
    log.info("DONE — %d keys → %s", len(written), out_path)


if __name__ == '__main__':
    main()
