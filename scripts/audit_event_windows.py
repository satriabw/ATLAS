"""Audit: reproduce Crosswalk's event-window definition and compare it to ours.

Read-only diagnosis. Derives, from raw tracking boxes, the event window Crosswalk
defines (vehicle enters -> leaves the crosswalk area, via line-segment crossing) and
compares it against the window our parquets imply (min->max over all pedestrian-pair
rows of a (v_track_id, roi) group).

Geometry is imported unchanged from the reference implementation so fidelity is not
re-implemented. See artifacts/docs/2026-07-23_event_window_audit/plan.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REF_DIR = "/home/satria/Project/Crosswalk"
sys.path.insert(0, REF_DIR)
from tools import is_in_poly, is_intersection  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
from dataset.labels import parse_train_label  # noqa: E402

# Crosswalk geometry (filtering_1.py / filtering_2.py), 1200x1100 image space.
LINES = {
    "TOP": [((65.0, 153.0), (1028.0, 165.0)), ((57.0, 331.0), (1049.0, 313.0))],
    "BOT": [((338.0, 589.0), (937.0, 377.0)), ((398.0, 676.0), (951.0, 458.0))],
}
POLY = {
    "TOP": [[0.0, 120.0], [1200.0, 120.0], [1200.0, 480.0], [0.0, 480.0]],
    "BOT": [[120.0, 480.0], [1100.0, 240.0], [1100.0, 600.0], [240.0, 1000.0]],
}
MIN_SPAN = 32  # filtering_3.py discards shorter samples
ROIS = ("TOP", "BOT")

# Cheap reject boxes for the line segments (avoids ~40M pure-python geometry calls).
SEG_BBOX = {
    roi: [(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
          for a, b in segs]
    for roi, segs in LINES.items()
}

log = logging.getLogger("audit")


def read_tracking(path: Path) -> dict:
    """Parse `frame_id / count / (track_id cls x1 y1 x2 y2)*` into {frame: [boxes]}."""
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    frames, i = {}, 0
    while i < len(lines):
        fid = int(lines[i]); i += 1
        n = int(lines[i]); i += 1
        boxes = []
        for _ in range(n):
            p = lines[i].split(); i += 1
            boxes.append((int(p[0]), int(p[1]),
                          float(p[2]), float(p[3]), float(p[4]), float(p[5])))
        frames[fid] = boxes
    return frames


def crosswalk_windows(frames: dict):
    """Per (track_id, roi): (first, last) frame the vehicle crosses that ROI's lines.

    Also returns, per roi, the set of frames with a pedestrian centre inside the polygon
    (filtering_1.py's 'vehicle of interest' condition).
    """
    veh_frames = defaultdict(list)
    ped_frames = {roi: set() for roi in ROIS}

    for fid, boxes in frames.items():
        for tid, cls, x1, y1, x2, y2 in boxes:
            if cls == 1:  # pedestrian
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                for roi in ROIS:
                    if is_in_poly([cx, cy], POLY[roi]):
                        ped_frames[roi].add(fid)
            elif cls == 0:  # vehicle
                tl, br = (x1, y1), (x2, y2)
                for roi in ROIS:
                    for (sx0, sy0, sx1, sy1), seg in zip(SEG_BBOX[roi], LINES[roi]):
                        if x2 < sx0 or x1 > sx1 or y2 < sy0 or y1 > sy1:
                            continue  # cheap reject
                        if is_intersection(seg, tl, br):
                            veh_frames[(tid, roi)].append(fid)
                            break

    windows = {}
    for key, fl in veh_frames.items():
        lo, hi = min(fl), max(fl)
        if hi - lo >= MIN_SPAN:
            windows[key] = (lo, hi, len(fl))
    return windows, ped_frames


def parquet_windows(path: Path) -> dict:
    """Per (v_track_id, roi): (min, max) over the group's concatenated frames arrays."""
    df = pd.read_parquet(path)
    out = {}
    for (tid, roi), g in df.groupby(["v_track_id", "roi"]):
        allf = np.concatenate([np.asarray(f).ravel() for f in g["frames"]])
        out[(int(tid), str(roi))] = (int(allf.min()), int(allf.max()), len(g))
    return out


def load_labeled_keys(labels_dir: Path) -> dict:
    """{(video_id, track_id, roi): annotation} across train + test pkl."""
    keys = {}
    for name in ("train_labels.pkl", "test_labels.pkl"):
        p = labels_dir / name
        if not p.exists():
            log.warning("missing %s", p)
            continue
        with open(p, "rb") as f:
            strings, annotations = pickle.load(f)
        for s, a in zip(strings, annotations):
            vid, tid, roi, _ = parse_train_label(s)
            keys[(vid, int(tid), str(roi))] = int(a)
    return keys


def tiou(a, b) -> float:
    lo = max(a[0], b[0]); hi = min(a[1], b[1])
    inter = max(0, hi - lo)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT)
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "artifacts/docs/2026-07-23_event_window_audit")
    ap.add_argument("--videos", type=int, default=120)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tracking_dir = args.data_root / "data/raw/tracking"
    parquet_dir = args.data_root / "data/processed/interactions"

    labeled = load_labeled_keys(args.data_root / "data/raw/labels")
    log.info("labeled events: %d", len(labeled))

    rows = []
    cw_spans, our_spans = [], []
    for vnum in range(1, args.videos + 1):
        vname = f"video_{vnum:03d}"
        tpath = tracking_dir / f"{vname}.txt"
        ppath = parquet_dir / f"{vname}_interactions.parquet"
        if not tpath.exists() or not ppath.exists():
            log.warning("%s: missing tracking or parquet — skipped", vname)
            continue

        cw, ped_frames = crosswalk_windows(read_tracking(tpath))
        ours = parquet_windows(ppath)
        cw_spans += [hi - lo + 1 for lo, hi, _ in cw.values()]
        our_spans += [hi - lo + 1 for lo, hi, _ in ours.values()]

        vkeys = [(v, t, r) for (v, t, r) in labeled if v == vname]
        for (v, tid, roi) in vkeys:
            o = ours.get((tid, roi))
            c = cw.get((tid, roi))
            c_other = cw.get((tid, "BOT" if roi == "TOP" else "TOP"))
            ped_ok = ""
            if c:
                ped_ok = bool(any(f in ped_frames[roi] for f in range(c[0], c[1] + 1)))
            rows.append({
                "video_id": v, "v_track_id": tid, "roi": roi,
                "annotation": labeled[(v, tid, roi)],
                "our_start": o[0] if o else "", "our_end": o[1] if o else "",
                "our_span": (o[1] - o[0] + 1) if o else "",
                "cw_start": c[0] if c else "", "cw_end": c[1] if c else "",
                "cw_span": (c[1] - c[0] + 1) if c else "",
                "tiou": round(tiou(o[:2], c[:2]), 4) if (o and c) else "",
                "start_delta": (o[0] - c[0]) if (o and c) else "",
                "end_delta": (o[1] - c[1]) if (o and c) else "",
                "has_parquet": bool(o), "has_cw_window": bool(c),
                "cw_window_other_roi_only": bool(c_other) and not bool(c),
                "ped_in_roi_during_window": ped_ok,
            })
        log.info("%s: cw=%d ours=%d labeled=%d", vname, len(cw), len(ours), len(vkeys))

    csv_path = args.out_dir / "window_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    log.info("wrote %s (%d rows)", csv_path, len(rows))

    def pct(arr, qs=(50, 75, 90, 95, 99)):
        a = np.asarray(arr)
        return {f"p{q}": float(np.percentile(a, q)) for q in qs} if len(a) else {}

    both = [r for r in rows if r["has_parquet"] and r["has_cw_window"]]
    summary = {
        "labeled_events": len(rows),
        "with_parquet": sum(r["has_parquet"] for r in rows),
        "with_cw_window": sum(r["has_cw_window"] for r in rows),
        "with_both": len(both),
        "no_cw_support_in_labeled_roi": sum(not r["has_cw_window"] for r in rows),
        "cw_window_only_in_OTHER_roi": sum(r["cw_window_other_roi_only"] for r in rows),
        "ped_in_roi_during_window_false": sum(r["ped_in_roi_during_window"] is False
                                              for r in rows),
        "cw_span_all_events": {**pct(cw_spans), "n": len(cw_spans),
                               "max": float(max(cw_spans)) if cw_spans else 0},
        "our_span_all_events": {**pct(our_spans), "n": len(our_spans),
                                "max": float(max(our_spans)) if our_spans else 0},
        "labeled_cw_span": pct([r["cw_span"] for r in both]),
        "labeled_our_span": pct([r["our_span"] for r in both]),
        "tiou": pct([r["tiou"] for r in both], qs=(5, 25, 50, 75, 95)),
        "tiou_mean": float(np.mean([r["tiou"] for r in both])) if both else 0,
        "tiou_below_0.5": sum(r["tiou"] < 0.5 for r in both),
        "tiou_below_0.1": sum(r["tiou"] < 0.1 for r in both),
    }
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("EVENT-WINDOW AUDIT SUMMARY")
    print("=" * 70)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    main()
