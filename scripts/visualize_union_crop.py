"""Visualize the R2 UNION crop (vehicle ∪ top-1 ped, event-static) but computed
over the fixed ±32 centered window — i.e. what the 0.659-anchor representation
would look like if it used the centered window instead of the whole-track linspace.
Lets us eyeball the "union vs tight single-object crop" difference directly.
"""
import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT / "training"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dataset.centered_window import build_centered_window
from dataset.tracking import parse_tracking, group_grid_boxes
from dataset.trajectory import DEFAULT_TOP_K
from build_h5_r2 import union_window
from build_h5_centered_crop import _decode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=int, default=2)
    p.add_argument("--tid", type=int, default=None)
    p.add_argument("--disp", type=int, default=224, help="crop display size")
    p.add_argument("--out", default="artifacts/experiments/2026-06-18_centered_crop_vision/union_crop.png")
    args = p.parse_args()

    vname = f"video_{args.video:03d}"
    df = pd.read_parquet(DATA_DIR / "processed" / "interactions" / f"{vname}_interactions.parquet")
    track_frames = parse_tracking(DATA_DIR / "raw" / "tracking" / f"{vname}.txt")

    groups = list(df.groupby(["v_track_id", "roi"]))
    if args.tid is not None:
        g = next(g for (tid, roi), g in groups if int(tid) == args.tid); tid = args.tid
    else:
        (tid, roi), g = groups[0]; tid = int(tid)

    w = build_centered_window(g, top_k=DEFAULT_TOP_K, half=32)
    frames = w["frames"]
    top1 = w["ped_ids"][0]                      # R2 union uses the top-1 pedestrian
    valid = frames[frames >= 0]
    # event-static union window over the centered-window frames
    v_boxes, p_boxes = group_grid_boxes(track_frames, valid, tid, [top1])
    window = union_window(v_boxes, p_boxes)     # (x0,y0,x1,y1) tracking space
    x0, y0, x1, y1 = map(int, np.round(window))
    print(f"{vname} v={tid} top1_ped={top1} centre={w['centre_frame']} union_window={window}")

    cols = sorted(set(np.linspace(0, len(frames) - 1, 6).astype(int).tolist() + [32]))
    with h5py.File(DATA_DIR / "raw" / "video" / "frames_db.h5", "r") as m:
        vds = m[vname]
        fig, axes = plt.subplots(2, len(cols), figsize=(2.9 * len(cols), 6.2))
        for c, k in enumerate(cols):
            f = int(frames[k]); ax_full, ax_crop = axes[0, c], axes[1, c]
            ttl = f"slot {k} frame {f}" + ("  <<CENTRE" if k == 32 else "")
            if f < 0:
                for ax in (ax_full, ax_crop):
                    ax.imshow(np.zeros((args.disp, args.disp, 3), np.uint8)); ax.axis("off")
                ax_full.set_title(ttl + "\n(padded)", fontsize=8); continue
            img = _decode(vds, f)
            disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
            vb = track_frames.get(f, {}).get(tid)
            pb = track_frames.get(f, {}).get(top1)
            if vb is not None:
                cv2.rectangle(disp, tuple(map(int, vb[:2])), tuple(map(int, vb[2:])), (0, 255, 0), 3)
            if pb is not None:
                cv2.rectangle(disp, tuple(map(int, pb[:2])), tuple(map(int, pb[2:])), (255, 0, 0), 3)
            cv2.rectangle(disp, (x0, y0), (x1, y1), (255, 255, 0), 4)   # static union window
            crop = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, (args.disp, args.disp))
            ax_full.imshow(disp); ax_full.set_title(ttl, fontsize=8); ax_full.axis("off")
            ax_crop.imshow(crop); ax_crop.axis("off")
        axes[1, 0].axis("on"); axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])
        axes[1, 0].set_ylabel("union crop", fontsize=10)
        fig.suptitle(f"{vname} veh={tid} ped={top1} — UNION crop (event-static, yellow) over ±32 window\n"
                     f"green=vehicle  red=top-1 ped", fontsize=11)
        fig.tight_layout()
        out = PROJECT_ROOT / args.out; out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=90, bbox_inches="tight"); print(f"saved → {out}")


if __name__ == "__main__":
    main()
