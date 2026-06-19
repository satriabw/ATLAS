"""Sanity-check temporal+spatial alignment between the trajectory centered window
and the vision crops, for ONE labeled interaction.

For a chosen (video, tid, roi) it builds the centered window (half=32), then for a
handful of slots across the 64-frame window renders: the native frame with the
vehicle bbox (green), target-pedestrian bbox (red) and both tight-crop windows;
plus the vehicle crop and the pedestrian crop the builder would emit. The planar
vehicle↔ped distance from the trajectory features is printed per slot — slot 32
(centre) should be the minimum, proving the vision clip is centred on the same
closest-approach frame the trajectory uses.
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
from dataset.tracking import parse_tracking
from dataset.trajectory import DEFAULT_TOP_K
from build_h5_centered_crop import _closest_ped, _tight_square, _decode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=int, default=2)
    p.add_argument("--tid", type=int, default=None, help="vehicle track id; default = first group")
    p.add_argument("--roi", default=None)
    p.add_argument("--size", type=int, default=112)
    p.add_argument("--pad", type=float, default=0.15)
    p.add_argument("--out", default="artifacts/experiments/2026-06-18_centered_crop_vision/align_check.png")
    args = p.parse_args()

    vname = f"video_{args.video:03d}"
    df = pd.read_parquet(DATA_DIR / "processed" / "interactions" / f"{vname}_interactions.parquet")
    track_frames = parse_tracking(DATA_DIR / "raw" / "tracking" / f"{vname}.txt")

    groups = list(df.groupby(["v_track_id", "roi"]))
    if args.tid is not None:
        g = next(g for (tid, roi), g in groups if int(tid) == args.tid
                 and (args.roi is None or roi == args.roi))
        tid = args.tid
    else:
        (tid, roi), g = groups[0]
        tid = int(tid)
    print(f"Visualizing {vname} v_track_id={tid}")

    w = build_centered_window(g, top_k=DEFAULT_TOP_K, half=32)
    frames = w["frames"]
    centre = int(w["centre_frame"])
    ped_target = _closest_ped(g, w["ped_ids"])
    t_idx = w["ped_ids"].index(ped_target)
    dist = np.linalg.norm(w["ped_feats"][t_idx][:, :2], axis=1)  # planar p_rel magnitude
    print(f"centre_frame={centre}  target_ped={ped_target}  ped_ids={w['ped_ids']}")

    cols = sorted(set(np.linspace(0, len(frames) - 1, 5).astype(int).tolist() + [32]))

    with h5py.File(DATA_DIR / "raw" / "video" / "frames_db.h5", "r") as m:
        vds = m[vname]
        fig, axes = plt.subplots(3, len(cols), figsize=(3.2 * len(cols), 9))
        for c, k in enumerate(cols):
            f = int(frames[k])
            ax_full, ax_veh, ax_ped = axes[0, c], axes[1, c], axes[2, c]
            valid = w["vehicle_mask"][k] == False
            title = f"slot {k}  frame {f}\nd={dist[k]:.1f}m" + ("  <<CENTRE" if k == 32 else "")
            if f < 0:
                for ax in (ax_full, ax_veh, ax_ped):
                    ax.imshow(np.zeros((args.size, args.size, 3), np.uint8)); ax.axis("off")
                ax_full.set_title(title + "\n(padded)", fontsize=8)
                continue

            img = _decode(vds, f)  # native BGR
            disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
            vbox = track_frames.get(f, {}).get(tid)
            pbox = track_frames.get(f, {}).get(ped_target)

            def crop_of(box):
                if box is None:
                    return np.zeros((args.size, args.size, 3), np.uint8)
                x0, y0, x1, y1 = _tight_square(box, args.pad)
                cv2.rectangle(disp, (x0, y0), (x1, y1), (255, 255, 0), 2)
                cr = img[y0:y1, x0:x1]
                if cr.size == 0:
                    return np.zeros((args.size, args.size, 3), np.uint8)
                return cv2.cvtColor(cv2.resize(cr, (args.size, args.size)), cv2.COLOR_BGR2RGB)

            veh_crop = crop_of(vbox)
            ped_crop = crop_of(pbox)
            if vbox is not None:
                cv2.rectangle(disp, tuple(map(int, vbox[:2])), tuple(map(int, vbox[2:])), (0, 255, 0), 3)
            if pbox is not None:
                cv2.rectangle(disp, tuple(map(int, pbox[:2])), tuple(map(int, pbox[2:])), (255, 0, 0), 3)

            ax_full.imshow(disp); ax_full.set_title(title, fontsize=8); ax_full.axis("off")
            ax_veh.imshow(veh_crop); ax_veh.axis("off")
            ax_ped.imshow(ped_crop); ax_ped.axis("off")
        axes[1, 0].set_ylabel("vehicle crop", fontsize=10)
        axes[2, 0].set_ylabel("ped crop", fontsize=10)
        for r, lab in [(1, "vehicle crop"), (2, "ped crop")]:
            axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
            axes[r, 0].set_ylabel(lab, fontsize=10)
        fig.suptitle(f"{vname} veh={tid} ped={ped_target}  (green=vehicle, red=ped, yellow=crop window)",
                     fontsize=11)
        fig.tight_layout()
        out = PROJECT_ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=90, bbox_inches="tight")
        print(f"saved → {out}")


if __name__ == "__main__":
    main()
