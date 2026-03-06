"""Entry point for World-EIoU evaluation.

Usage:
    python -m evaluation.run_evaluation \
        --parquet-dir data/processed/interactions \
        --labels-pkl  data/raw/labels/train_labels.pkl \
        --video-ids   1 2 3 4 5

Or from the training directory:
    python evaluation/run_evaluation.py ...
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a script from training/ without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.localization   import compute_localization_rate
from evaluation.ap_calculator  import compute_map

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label parsing (mirrors violation_dataset.py)
# ---------------------------------------------------------------------------

def parse_label_string(s: str):
    """Parse 'V001I00002S1D0R0A1' -> (video_id, track_id, roi, annotation).

    S1 = BOT, S0 = TOP
    A0 = violation (label=1 in our event schema), A1 = compliance (label=0)

    Note: the task spec uses label=1 for violation, label=0 for non-violation,
    which is the *opposite* of ViolationDataset.annotation (0=violation there).
    We remap here so the evaluation API is self-consistent.
    """
    m = re.match(r'V(\d+)I(\d+)S(\d)D\d+R\d+A(\d)', s)
    if not m:
        raise ValueError(f"Cannot parse label string: {s!r}")
    video_id   = f"video_{int(m.group(1)):03d}"
    track_id   = int(m.group(2))
    roi        = 'BOT' if m.group(3) == '1' else 'TOP'
    annotation = int(m.group(4))          # 0=violation, 1=compliance in pkl
    # Remap: event label=1 means violation, label=0 means non-violation
    label = 1 if annotation == 0 else 0
    return video_id, track_id, roi, label


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------

def _to_loc(val) -> np.ndarray:
    arr = np.asarray(val)
    if arr.dtype == object:
        return np.stack(arr.tolist()).astype(np.float32)
    return arr.astype(np.float32).reshape(-1, 2)


def _to_frames(val) -> np.ndarray:
    return np.asarray(val, dtype=np.int64).ravel()


def build_detected_events(parquet_dir: Path, video_ids: list[str]) -> list[dict]:
    """Build a detected-event list from parquet files.

    Each (video_id, v_track_id, roi) group becomes one event.
    score = 1.0 (no model yet).
    pos_start / pos_end = vehicle planar position at first/last frame.
    label = 0 (unknown at this stage — localization ignores labels anyway).
    """
    events: list[dict] = []

    for vid in video_ids:
        parquet_path = parquet_dir / f"{vid}_interactions.parquet"
        if not parquet_path.exists():
            logger.warning(f"Parquet not found: {parquet_path}")
            continue

        df = pd.read_parquet(parquet_path)

        for (v_track_id, roi), group in df.groupby(["v_track_id", "roi"]):
            try:
                # Collect all frames and matching v_loc_planar across rows
                all_frames: list[np.ndarray] = []
                all_vloc:   list[np.ndarray] = []

                for _, row in group.iterrows():
                    f = _to_frames(row["frames"])
                    v = _to_loc(row["v_loc_planar"])
                    if len(f) != len(v):
                        min_len = min(len(f), len(v))
                        f, v = f[:min_len], v[:min_len]
                    all_frames.append(f)
                    all_vloc.append(v)

                frames_cat = np.concatenate(all_frames)
                vloc_cat   = np.vstack(all_vloc)

                order       = np.argsort(frames_cat, kind="stable")
                frames_cat  = frames_cat[order]
                vloc_cat    = vloc_cat[order]

                events.append({
                    "video_id":    vid,
                    "v_track_id":  int(v_track_id),
                    "roi":         str(roi),
                    "label":       0,                        # unknown
                    "frame_start": int(frames_cat[0]),
                    "frame_end":   int(frames_cat[-1]),
                    "pos_start":   vloc_cat[0].tolist(),
                    "pos_end":     vloc_cat[-1].tolist(),
                    "score":       1.0,
                })
            except Exception as exc:
                logger.warning(f"Skipping group ({vid}, {v_track_id}, {roi}): {exc}")

    logger.info(f"Built {len(events)} detected events from {len(video_ids)} video(s)")
    return events


def build_gt_events(
    labels_pkl: Path,
    video_ids: list[str],
    detected_events: list[dict],
) -> list[dict]:
    """Build GT event list from labels pkl.

    GT has no world positions, so we proxy them from the matching detected
    event (same video_id, v_track_id, roi) — localization check only.
    """
    with open(labels_pkl, "rb") as f:
        label_strings, _ = pickle.load(f)

    # Index detected events for fast lookup
    det_index: dict[tuple, dict] = {}
    for det in detected_events:
        key = (det["video_id"], det["v_track_id"], det["roi"])
        det_index[key] = det

    video_set = set(video_ids)
    gt_events: list[dict] = []
    skipped   = 0

    for s in label_strings:
        try:
            vid, tid, roi, label = parse_label_string(s)
        except ValueError as e:
            logger.warning(e)
            continue

        if vid not in video_set:
            continue

        key = (vid, tid, roi)
        det = det_index.get(key)
        if det is None:
            skipped += 1
            continue

        gt_events.append({
            "video_id":    vid,
            "v_track_id":  tid,
            "roi":         roi,
            "label":       label,
            "frame_start": det["frame_start"],
            "frame_end":   det["frame_end"],
            "pos_start":   det["pos_start"],
            "pos_end":     det["pos_end"],
        })

    logger.info(
        f"Built {len(gt_events)} GT events "
        f"({skipped} skipped — no matching detected event)"
    )
    return gt_events


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="World-EIoU Evaluation")
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=Path("/home/satria/Project/ATLAS/data/processed/interactions"),
        help="Directory containing <video_id>_interactions.parquet files",
    )
    parser.add_argument(
        "--labels-pkl",
        type=Path,
        default=Path("/home/satria/Project/ATLAS/data/raw/labels/test_labels.pkl"),
        help="Path to labels pkl (train_labels.pkl or test_labels.pkl)",
    )
    parser.add_argument(
        "--video-ids",
        nargs="+",
        type=int,
        default=list(range(1, 61, 2)),   # all odd-numbered (train split)
        help="Video numbers to evaluate (e.g. 1 3 5 7 ...)",
    )
    parser.add_argument("--d-max",          type=float, default=5.0,  help="SPIoU distance threshold (m)")
    parser.add_argument("--eiou-threshold", type=float, default=0.5,  help="World-EIoU TP threshold")
    args = parser.parse_args()

    video_ids = [f"video_{n:03d}" for n in args.video_ids]
    logger.info(f"Evaluating {len(video_ids)} video(s): {video_ids[:5]}{'...' if len(video_ids) > 5 else ''}")

    detected = build_detected_events(args.parquet_dir, video_ids)
    gt       = build_gt_events(args.labels_pkl, video_ids, detected)

    # ---- Localization report ----
    loc = compute_localization_rate(
        detected, gt,
        d_max=args.d_max,
        eiou_threshold=args.eiou_threshold,
    )

    print()
    print("=== Localization Report ===")
    print(f"Total GT events : {loc['total_gt']}")
    print(f"Localized       : {loc['localized']}  ({loc['localization_rate']:.1%})")
    for roi in ("TOP", "BOT"):
        rate = loc["per_roi"].get(roi, 0.0)
        print(f"{roi:<8}        : {rate:.1%}")

    # ---- AP report (score=1.0 baseline) ----
    ap_result = compute_map(
        detected, gt,
        d_max=args.d_max,
        eiou_threshold=args.eiou_threshold,
    )

    print()
    print("=== AP Report (score=1.0 baseline) ===")
    print(f"APv  : {ap_result['APv']:.3f}  (no model yet — expected)")
    print(f"APn  : {ap_result['APn']:.3f}  (all unmatched → non-violation)")
    print(f"mAP  : {ap_result['mAP']:.3f}")


if __name__ == "__main__":
    main()
