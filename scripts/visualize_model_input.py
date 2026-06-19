"""Visualize EXACTLY what the model is fed, per interaction event.

For a handful of labels we render one figure each with three blocks:

  1. Vision input  — the R2 crop frames (what the vision branch reads) with the
     grounding-mask boxes overlaid (lime = subject vehicle, cyan = top-1 ped);
     these boxes are the two extra mask channels the network gets.
  2. Trajectory, camera-oriented — the raw planar path projected through the
     camera model into 1200x1100 image space, so it is oriented to match the
     video (not arbitrary planar axes), with the crop window drawn on top.
  3. Features — both the RAW parquet arrays (planar x/y, speeds) and the DERIVED
     model-input features the GRU actually sees: vehicle (x_centered, y_centered,
     speed) and pedestrian (rel_x, rel_y, speed), built identically to
     dataset/trajectory.py.

Run:  python scripts/visualize_model_input.py
"""
import argparse
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
from dataset.trajectory import (build_group_trajectory, _to_frames, _to_loc,
                                _to_scalar_seq)
from dataset.centered_window import build_centered_window
from dataset.tracking import parse_tracking, group_grid_boxes

CAM_YML = "/home/satria/Project/crosswalk-ws/data/calibration/camera_model.yml"
TRACK_W, TRACK_H = 1200, 1100
PAD_FRAC = 0.15
NUM_FRAMES = 32
SIZE = 320
N_SHOW = 6  # crop frames drawn on the contact sheet
N_VIOL, N_COMPLY = 3, 2
ZIP_RANGES = {"29983897.zip": range(1, 41),
              "30050131.zip": range(41, 81),
              "30051331.zip": range(81, 121)}
OUT = ROOT / "results" / "model_input_inspection"
TRIM = False


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


def tracking_to_crop(px, py, window):
    wx0, wy0, wx1, wy1 = window
    cx = (px - wx0) / max(wx1 - wx0, 1e-6) * SIZE
    cy = (py - wy0) / max(wy1 - wy0, 1e-6) * SIZE
    return cx, cy


def read_crop(cap, frame_idx, window, frame_wh):
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


def vehicle_raw(g):
    """Sorted-unique (frame, planar, speed) for the vehicle — matches trajectory.py."""
    allf = np.concatenate([_to_frames(r["frames"]) for _, r in g.iterrows()])
    allloc = np.vstack([_to_loc(r["v_loc_planar"]) for _, r in g.iterrows()])
    allsp = np.concatenate([_to_scalar_seq(r["v_speed"]) for _, r in g.iterrows()])
    order = np.argsort(allf, kind="stable")
    _, keep = np.unique(allf[order], return_index=True)
    idx = order[keep]
    return allf[order][keep], allloc[idx], allsp[idx]


def ped_raw(g, pid):
    """Sorted (frame, p_planar, p_speed, v_planar@frame) for one pedestrian."""
    rows = g[g["p_track_id"] == pid]
    pf = np.concatenate([_to_frames(r["frames"]) for _, r in rows.iterrows()])
    ploc = np.vstack([_to_loc(r["p_loc_planar"]) for _, r in rows.iterrows()])
    psp = np.concatenate([_to_scalar_seq(r["p_speed"]) for _, r in rows.iterrows()])
    vloc = np.vstack([_to_loc(r["v_loc_planar"]) for _, r in rows.iterrows()])
    o = np.argsort(pf, kind="stable")
    return pf[o], ploc[o], psp[o], vloc[o]


def pick_events():
    labs, anns = pickle.load(open(ROOT / "data/raw/labels/train_labels.pkl", "rb"))
    by_video = {}
    for s, a in zip(labs, anns):
        vid, tid, roi, ann = parse_train_label(s)
        vn = int(vid[-3:])
        by_video.setdefault(vn, {0: [], 1: []})[ann].append((vn, int(tid), roi, s))
    chosen = {0: [], 1: []}
    for vn in sorted(by_video):
        pq = ROOT / f"data/processed/interactions/video_{vn:03d}_interactions.parquet"
        tk = ROOT / f"data/raw/tracking/video_{vn:03d}.txt"
        if not pq.exists() or not tk.exists():
            continue
        df = pd.read_parquet(pq)
        groups = {(int(t), r) for t, r in df[["v_track_id", "roi"]].drop_duplicates().values}
        for ann, need in ((0, N_VIOL), (1, N_COMPLY)):
            for ev in by_video[vn][ann]:
                if len(chosen[ann]) >= need:
                    break
                if (ev[1], ev[2]) in groups:
                    chosen[ann].append(ev)
        if len(chosen[0]) >= N_VIOL and len(chosen[1]) >= N_COMPLY:
            break
    return chosen[0][:N_VIOL] + chosen[1][:N_COMPLY]


def render(cm, vn, tid, roi, label_str, ann):
    pq = ROOT / f"data/processed/interactions/video_{vn:03d}_interactions.parquet"
    g = pd.read_parquet(pq)
    g = g[(g.v_track_id == tid) & (g.roi == roi)]

    # derived model-input features (exactly what the GRU is fed)
    start, vfeat, pfeats, pids = build_group_trajectory(g, top_k=1)
    pid = pids[0]

    # raw parquet arrays
    vf, vloc, vsp = vehicle_raw(g)
    pf, ploc, psp, vloc_at_p = ped_raw(g, pid)

    if TRIM:
        # restrict everything to the EEG-style centered window (32 frames around
        # global closest approach) — exactly what the centered model would see.
        w = build_centered_window(g, top_k=1, half=NUM_FRAMES // 2)
        winf = w['frames']
        anchor = w['centre_frame']
        vfeat, pfeats = w['vehicle_feat'], w['ped_feats']
        f_lo, f_hi = int(winf[winf >= 0].min()), int(winf[winf >= 0].max())
        vm = (vf >= f_lo) & (vf <= f_hi)
        pm = (pf >= f_lo) & (pf <= f_hi)
        vf, vloc, vsp = vf[vm], vloc[vm], vsp[vm]
        pf, ploc, psp, vloc_at_p = pf[pm], ploc[pm], psp[pm], vloc_at_p[pm]
        grid = np.where(winf >= 0, winf, anchor)  # padded slots → centre frame (display only)
    else:
        grid = np.linspace(int(vf.min()), int(vf.max()), NUM_FRAMES, dtype=int)

    v_cent = vloc - vloc[0:1]          # vehicle derived: (x_centered, y_centered)
    p_rel = ploc - vloc_at_p           # pedestrian derived: (rel_x, rel_y)

    # crop window + grounding boxes on the 32-frame grid
    tr = parse_tracking(ROOT / f"data/raw/tracking/video_{vn:03d}.txt")
    v_boxes, p_boxes = group_grid_boxes(tr, grid, tid, pids)
    window = union_window(v_boxes, p_boxes)

    # camera-projected (image-space) paths — oriented to the video
    uv_v = np.array([cm.project_point(np.asarray(l, float)) for l in vloc])
    uv_p = np.array([cm.project_point(np.asarray(l, float)) for l in ploc])

    cap, frame_wh, tmpname = open_video(vn)
    show_idx = np.linspace(0, NUM_FRAMES - 1, N_SHOW, dtype=int)
    panels = []
    try:
        for gi in show_idx:
            fr = int(grid[gi])
            crop = read_crop(cap, fr, window, frame_wh)
            vb = v_boxes[gi]
            pb = p_boxes[0, gi] if p_boxes.shape[0] else np.full(4, np.nan)
            panels.append((fr, crop, vb, pb))
    finally:
        cap.release()
        os.unlink(tmpname)

    cls = "VIOLATION" if ann == 0 else "COMPLIANCE"
    fig = plt.figure(figsize=(22, 11))
    gs = fig.add_gridspec(3, N_SHOW, height_ratios=[1.0, 1.3, 1.3])

    # ---- row 0: vision input (crop frames + grounding boxes) ----
    for k, (fr, crop, vb, pb) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(crop)
        if not np.isnan(vb[0]):
            x0, y0 = tracking_to_crop(vb[0], vb[1], window)
            x1, y1 = tracking_to_crop(vb[2], vb[3], window)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="lime", lw=2))
        if not np.isnan(pb[0]):
            x0, y0 = tracking_to_crop(pb[0], pb[1], window)
            x1, y1 = tracking_to_crop(pb[2], pb[3], window)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="cyan", lw=1.5))
        ax.set_title(f"frame {fr}", fontsize=8)
        ax.axis("off")

    def split2(gs_row):
        # split a gridspec row into 4 equal analysis panels regardless of N_SHOW
        sub = gs_row.subgridspec(1, 4)
        return [fig.add_subplot(sub[0, i]) for i in range(4)]

    # ---- row 1: camera-oriented image-space + raw planar + raw speeds ----
    a_img, a_planar, a_vsp, a_legend = split2(gs[1, :])

    a_img.plot(uv_v[:, 0], uv_v[:, 1], "-", color="tab:blue", lw=1.5, label="vehicle")
    a_img.plot(uv_p[:, 0], uv_p[:, 1], "-", color="magenta", lw=1.5, label="top-1 ped")
    a_img.add_patch(Rectangle((window[0], window[1]), window[2] - window[0],
                              window[3] - window[1], fill=False, ec="orange", lw=2,
                              label="crop window"))
    a_img.scatter(*uv_v[0], c="green", s=30, zorder=5)
    a_img.scatter(*uv_v[-1], c="red", s=30, zorder=5)
    a_img.set_xlim(0, TRACK_W); a_img.set_ylim(TRACK_H, 0)
    a_img.set_aspect("equal"); a_img.set_title("camera-projected (image space, oriented)", fontsize=9)
    a_img.legend(fontsize=7)

    a_planar.plot(vloc[:, 0], vloc[:, 1], "-o", color="tab:blue", ms=2, lw=1, label="vehicle")
    a_planar.plot(ploc[:, 0], ploc[:, 1], "-o", color="magenta", ms=2, lw=1, label="top-1 ped")
    a_planar.scatter(vloc[0, 0], vloc[0, 1], c="green", s=30, zorder=5, label="start")
    a_planar.scatter(vloc[-1, 0], vloc[-1, 1], c="red", s=30, zorder=5, label="end")
    a_planar.set_aspect("equal"); a_planar.set_title("RAW planar (meters)", fontsize=9)
    a_planar.set_xlabel("x [m]"); a_planar.set_ylabel("y [m]"); a_planar.legend(fontsize=7)

    a_vsp.plot(vf, vsp, "-", color="tab:blue", label="v_speed")
    a_vsp.plot(pf, psp, "-", color="magenta", label="p_speed")
    a_vsp.set_title("RAW speed vs frame", fontsize=9)
    a_vsp.set_xlabel("frame"); a_vsp.set_ylabel("speed"); a_vsp.legend(fontsize=7)

    a_legend.axis("off")
    txt = (f"{label_str}\nV{vn:03d}  tid={tid}  {roi}  {cls}\n\n"
           f"DERIVED features fed to GRU:\n"
           f"  vehicle_feat {vfeat.shape}  = (x_centered, y_centered, speed)\n"
           f"  ped_feat[0]  {pfeats[0].shape}  = (rel_x, rel_y, speed)\n\n"
           f"RAW arrays from parquet:\n"
           f"  vehicle: {len(vf)} steps   ped(top-1 id={pid}): {len(pf)} steps\n\n"
           f"VISION input: {NUM_FRAMES} crop frames @ {SIZE}px\n"
           f"  + 2 grounding-mask channels (lime veh / cyan ped boxes)")
    a_legend.text(0.0, 1.0, txt, fontsize=9, va="top", family="monospace")

    # ---- row 2: derived model-input features ----
    a_vpath, a_vcomp, a_ppath, a_pcomp = split2(gs[2, :])

    a_vpath.plot(v_cent[:, 0], v_cent[:, 1], "-o", color="tab:blue", ms=2, lw=1)
    a_vpath.scatter(0, 0, c="green", s=40, zorder=5, label="origin (t0)")
    a_vpath.set_aspect("equal"); a_vpath.set_title("DERIVED veh: centered path", fontsize=9)
    a_vpath.set_xlabel("x_centered"); a_vpath.set_ylabel("y_centered"); a_vpath.legend(fontsize=7)

    a_vcomp.plot(vf, v_cent[:, 0], label="x_centered")
    a_vcomp.plot(vf, v_cent[:, 1], label="y_centered")
    a_vcomp.plot(vf, vsp, label="speed")
    a_vcomp.set_title("DERIVED veh components vs frame", fontsize=9)
    a_vcomp.set_xlabel("frame"); a_vcomp.legend(fontsize=7)

    a_ppath.plot(p_rel[:, 0], p_rel[:, 1], "-o", color="magenta", ms=2, lw=1)
    a_ppath.scatter(0, 0, c="black", s=40, marker="x", zorder=5, label="vehicle (rel origin)")
    a_ppath.set_aspect("equal"); a_ppath.set_title("DERIVED ped: rel-to-vehicle path", fontsize=9)
    a_ppath.set_xlabel("rel_x"); a_ppath.set_ylabel("rel_y"); a_ppath.legend(fontsize=7)

    a_pcomp.plot(pf, p_rel[:, 0], label="rel_x")
    a_pcomp.plot(pf, p_rel[:, 1], label="rel_y")
    a_pcomp.plot(pf, psp, label="speed")
    a_pcomp.set_title("DERIVED ped components vs frame", fontsize=9)
    a_pcomp.set_xlabel("frame"); a_pcomp.legend(fontsize=7)

    fig.suptitle(f"{label_str}  |  V{vn:03d} tid={tid} {roi}  |  {cls}  "
                 f"(lime=veh box, cyan=top-1 ped box)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{label_str}_{cls}{'_TRIM' if TRIM else ''}.png"
    fig.savefig(path, dpi=85)
    plt.close(fig)
    print(f"  wrote {path.name}")


def main():
    global TRIM
    ap = argparse.ArgumentParser()
    ap.add_argument("--trim", action="store_true",
                    help="show the EEG-style centered 32-frame crop instead of full-track linspace")
    TRIM = ap.parse_args().trim
    cm = CameraModel.load_from_yml(CAM_YML)
    events = pick_events()
    print(f"Selected {len(events)} events "
          f"({sum(parse_train_label(s)[3] == 0 for *_, s in events)} viol)")
    for vn, tid, roi, s in events:
        ann = parse_train_label(s)[3]
        print(f"Event {s}  V{vn:03d} tid={tid} {roi} {'VIOL' if ann == 0 else 'COMPLY'}")
        render(cm, vn, tid, roi, s, ann)
    print(f"\nFigures in {OUT}")


if __name__ == "__main__":
    main()
