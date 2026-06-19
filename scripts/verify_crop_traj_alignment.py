"""Verify crop-sequence ↔ trajectory spatio-temporal alignment for sampled events.

For each chosen interaction event we:
  1. Build the trajectory (frames + v_loc_planar) exactly as the model dataset does.
  2. Build the 32-frame linspace grid + event-static union crop window exactly as
     build_h5_r2.py does (same idx-1 frame->video mapping).
  3. Read the real video frames, crop the window, and overlay — in crop pixel
     coords — the vehicle tracking box, the top-1 pedestrian box, and the
     trajectory's planar position back-projected through the camera model.

If the crop pixels and the trajectory disagree (vehicle not where the projected
dot sits, or off by frames) the overlay shows it immediately.
"""
import os
import pickle
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, "/home/satria/Project/crosswalk-ws")

from modules.calibration.camera_model import CameraModel
from dataset.labels import parse_train_label
from dataset.trajectory import build_group_trajectory
from dataset.tracking import parse_tracking, group_grid_boxes

CAM_YML = "/home/satria/Project/crosswalk-ws/data/calibration/camera_model.yml"
TRACK_W, TRACK_H = 1200, 1100
PAD_FRAC = 0.15
NUM_FRAMES = 32
SIZE = 320
N_SHOW = 8  # frames drawn on the contact sheet
ZIP_RANGES = {"29983897.zip": range(1, 41),
              "30050131.zip": range(41, 81),
              "30051331.zip": range(81, 121)}
OUT = ROOT / "results" / "crop_traj_alignment"


def union_window(v_boxes, p_boxes):
    boxes = np.concatenate([v_boxes, p_boxes.reshape(-1, 4)], axis=0)
    boxes = boxes[~np.isnan(boxes[:, 0])]
    if len(boxes) == 0:
        return np.array([0, 0, TRACK_W, TRACK_H], dtype=np.float32)
    x0, y0 = boxes[:, 0].min(), boxes[:, 1].min()
    x1, y1 = boxes[:, 2].max(), boxes[:, 3].max()
    side = max(x1 - x0, y1 - y0) * (1.0 + PAD_FRAC)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    wx0 = np.clip(cx - side / 2, 0, max(TRACK_W - side, 0))
    wy0 = np.clip(cy - side / 2, 0, max(TRACK_H - side, 0))
    wx1 = min(wx0 + side, TRACK_W)
    wy1 = min(wy0 + side, TRACK_H)
    return np.array([wx0, wy0, wx1, wy1], dtype=np.float32)


def event_trajectory(g):
    """Sorted-unique (frame, v_planar) for the vehicle across the group."""
    fparts = [np.asarray(f).ravel() for f in g["frames"]]
    lparts = [np.stack(np.asarray(l).tolist()).astype(float) for l in g["v_loc_planar"]]
    allf = np.concatenate(fparts)
    alll = np.vstack(lparts)
    o = np.argsort(allf, kind="stable")
    allf, alll = allf[o], alll[o]
    _, keep = np.unique(allf, return_index=True)
    return allf[keep].astype(int), alll[keep]


def tracking_to_crop(px, py, window):
    """Map a tracking-space point into the SIZExSIZE crop pixel frame."""
    wx0, wy0, wx1, wy1 = window
    cx = (px - wx0) / max(wx1 - wx0, 1e-6) * SIZE
    cy = (py - wy0) / max(wy1 - wy0, 1e-6) * SIZE
    return cx, cy


def read_crop(cap, frame_idx, window, frame_wh):
    """Replicate build_h5_r2.read_crops for a single frame (1-based -> idx-1)."""
    fw, fh = frame_wh
    sx, sy = fw / TRACK_W, fh / TRACK_H
    x0, y0 = int(round(window[0] * sx)), int(round(window[1] * sy))
    x1, y1 = int(round(window[2] * sx)), int(round(window[3] * sy))
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(frame_idx) - 1, 0))
    ok, frame = cap.read()
    if not ok or frame is None:
        return np.zeros((SIZE, SIZE, 3), np.uint8)
    crop = cv2.resize(frame[y0:y1, x0:x1], (SIZE, SIZE))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def open_video(video_num):
    zip_name = next(z for z, r in ZIP_RANGES.items() if video_num in r)
    avi = f"video_{video_num:03d}.avi"
    zf = zipfile.ZipFile(ROOT / "data" / zip_name)
    tmp = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
    tmp.write(zf.read(avi))
    tmp.flush()
    zf.close()
    cap = cv2.VideoCapture(tmp.name)
    wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    return cap, wh, tmp.name


def pick_events(n_each=5):
    """Pick n_each violations + n_each compliances, minimising distinct videos."""
    labs, anns = pickle.load(open(ROOT / "data/raw/labels/train_labels.pkl", "rb"))
    by_video = {}
    for s, a in zip(labs, anns):
        vid, tid, roi, ann = parse_train_label(s)
        vn = int(vid[-3:])
        by_video.setdefault(vn, {0: [], 1: []})[ann].append((vn, int(tid), roi, s))
    chosen = {0: [], 1: []}
    # prefer videos that offer both classes, parquet present
    for vn in sorted(by_video):
        pq = ROOT / f"data/processed/interactions/video_{vn:03d}_interactions.parquet"
        tk = ROOT / f"data/raw/tracking/video_{vn:03d}.txt"
        if not pq.exists() or not tk.exists():
            continue
        df = pd.read_parquet(pq)
        groups = {(int(t), r) for t, r in df[["v_track_id", "roi"]].drop_duplicates().values}
        for ann in (0, 1):
            if len(chosen[ann]) >= n_each:
                continue
            for ev in by_video[vn][ann]:
                if len(chosen[ann]) >= n_each:
                    break
                if (ev[1], ev[2]) in groups:
                    chosen[ann].append(ev)
        if len(chosen[0]) >= n_each and len(chosen[1]) >= n_each:
            break
    return chosen[0] + chosen[1]


def process_event(cm, vn, tid, roi, label_str, ann):
    pq = ROOT / f"data/processed/interactions/video_{vn:03d}_interactions.parquet"
    df = pd.read_parquet(pq)
    g = df[(df.v_track_id == tid) & (df.roi == roi)]
    start, vfeat, pfeats, pids = build_group_trajectory(g, top_k=1)
    allf, alll = event_trajectory(g)
    grid = np.linspace(int(allf.min()), int(allf.max()), NUM_FRAMES, dtype=int)

    tr = parse_tracking(ROOT / f"data/raw/tracking/video_{vn:03d}.txt")
    v_boxes, p_boxes = group_grid_boxes(tr, grid, tid, pids)
    window = union_window(v_boxes, p_boxes)

    # per-grid-frame: projected traj point + in-window flag + reproj err vs box
    f2loc = {int(f): l for f, l in zip(allf, alll)}
    cap, frame_wh, tmpname = open_video(vn)
    show_idx = np.linspace(0, NUM_FRAMES - 1, N_SHOW, dtype=int)
    reproj_err, in_window = [], 0
    panels = []
    try:
        for gi in show_idx:
            fr = int(grid[gi])
            crop = read_crop(cap, fr, window, frame_wh)
            vb = v_boxes[gi]
            pb = p_boxes[0, gi] if p_boxes.shape[0] else np.full(4, np.nan)
            loc = f2loc.get(fr)
            proj = None
            if loc is not None:
                u, v = cm.project_point(np.asarray(loc, float))
                proj = (u, v)
                if window[0] <= u <= window[2] and window[1] <= v <= window[3]:
                    in_window += 1
                if not np.isnan(vb[0]):
                    bc = ((vb[0] + vb[2]) / 2, vb[3])
                    reproj_err.append(np.hypot(u - bc[0], v - bc[1]))
            panels.append((fr, crop, vb, pb, proj))
    finally:
        cap.release()
        os.unlink(tmpname)

    # overall in-window over the FULL grid (not just shown frames)
    grid_in = 0
    grid_have = 0
    for fr in grid:
        loc = f2loc.get(int(fr))
        if loc is None:
            continue
        grid_have += 1
        u, v = cm.project_point(np.asarray(loc, float))
        if window[0] <= u <= window[2] and window[1] <= v <= window[3]:
            grid_in += 1

    render(panels, window, allf, alll, cm, vn, tid, roi, ann, label_str,
           reproj_err, grid_in, grid_have)
    return {
        "label": label_str, "video": vn, "tid": tid, "roi": roi,
        "class": "VIOL" if ann == 0 else "COMPLY",
        "traj_frames": len(allf), "grid_have": grid_have, "grid_in_window": grid_in,
        "reproj_px_mean": float(np.mean(reproj_err)) if reproj_err else float("nan"),
        "reproj_px_max": float(np.max(reproj_err)) if reproj_err else float("nan"),
    }


def render(panels, window, allf, alll, cm, vn, tid, roi, ann, label_str,
           reproj_err, grid_in, grid_have):
    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(2, 5)
    # trajectory panel (planar, image-projected)
    axp = fig.add_subplot(gs[:, 0])
    uv = np.array([cm.project_point(np.asarray(l, float)) for l in alll])
    axp.plot(uv[:, 0], uv[:, 1], "-", color="tab:blue", lw=1, label="veh traj (proj)")
    axp.add_patch(Rectangle((window[0], window[1]), window[2] - window[0],
                            window[3] - window[1], fill=False, ec="orange", lw=2,
                            label="crop window"))
    axp.scatter(uv[0, 0], uv[0, 1], c="green", s=30, zorder=5, label="start")
    axp.scatter(uv[-1, 0], uv[-1, 1], c="red", s=30, zorder=5, label="end")
    axp.set_xlim(0, TRACK_W); axp.set_ylim(TRACK_H, 0)
    axp.set_title("image-space (1200x1100)", fontsize=9)
    axp.legend(fontsize=7, loc="upper right")
    axp.set_aspect("equal")

    for k, (fr, crop, vb, pb, proj) in enumerate(panels):
        ax = fig.add_subplot(gs[k // 4, 1 + k % 4])
        ax.imshow(crop)
        if not np.isnan(vb[0]):
            x0, y0 = tracking_to_crop(vb[0], vb[1], window)
            x1, y1 = tracking_to_crop(vb[2], vb[3], window)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   ec="lime", lw=2))
        if not np.isnan(pb[0]):
            x0, y0 = tracking_to_crop(pb[0], pb[1], window)
            x1, y1 = tracking_to_crop(pb[2], pb[3], window)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   ec="cyan", lw=1.5))
        if proj is not None:
            cx, cy = tracking_to_crop(proj[0], proj[1], window)
            ax.scatter([cx], [cy], c="red", s=60, marker="x", lw=2)
        ax.set_title(f"frame {fr}", fontsize=8)
        ax.axis("off")

    cls = "VIOLATION" if ann == 0 else "COMPLIANCE"
    rp = f"{np.mean(reproj_err):.1f}/{np.max(reproj_err):.1f}" if reproj_err else "n/a"
    fig.suptitle(
        f"{label_str}  |  V{vn:03d} tid={tid} {roi}  |  {cls}  |  "
        f"reproj px mean/max={rp}  |  traj-in-window {grid_in}/{grid_have} grid frames"
        f"   (lime=veh box, cyan=top-1 ped, red x=traj proj)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{label_str}_{cls}.png"
    fig.savefig(path, dpi=90)
    plt.close(fig)
    print(f"  wrote {path.name}")


def main():
    cm = CameraModel.load_from_yml(CAM_YML)
    events = pick_events(5)
    print(f"Selected {len(events)} events")
    rows = []
    for vn, tid, roi, s in events:
        ann = parse_train_label(s)[3]
        print(f"Event {s}  V{vn:03d} tid={tid} {roi} {'VIOL' if ann==0 else 'COMPLY'}")
        rows.append(process_event(cm, vn, tid, roi, s, ann))
    summ = pd.DataFrame(rows)
    print("\n==== SUMMARY ====")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summ.to_string(index=False))
    summ.to_csv(OUT / "summary.csv", index=False)
    print(f"\nFigures + summary.csv in {OUT}")


if __name__ == "__main__":
    main()
