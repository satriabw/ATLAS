"""Run model inference then compute APv, APn, mAP.

Usage (from training/):
    python evaluation/evaluate_model.py \
        --checkpoint checkpoints/best_model.pth \
        --parquet-dir /home/satria/Project/ATLAS/data/processed/interactions \
        --labels-pkl  /home/satria/Project/ATLAS/data/raw/labels/train_labels.pkl \
        --split       train \
        --video-ids   1 3 5 7 9
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.violation_dataset import (
    _build_group_trajectory,
    _to_frames,
    _to_loc,
    load_violation_dataset,
)
from models import CrossAttentionModel
from evaluation.run_evaluation import build_gt_events, build_detected_events
from evaluation.ap_calculator import compute_map
from evaluation.localization import compute_localization_rate

import pandas as pd
import pickle
import re

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def _sample_trajectory(features: np.ndarray, num_frames: int = 32) -> np.ndarray:
    """Temporal resampling identical to ViolationDataset._sample_trajectory."""
    T = features.shape[0]
    if T == num_frames:
        return features
    elif T > num_frames:
        idx = np.linspace(0, T - 1, num_frames, dtype=int)
    else:
        repeat = num_frames // T
        rem    = num_frames % T
        idx    = list(range(T)) * repeat + list(range(rem))
    return features[idx]


def build_events_with_scores(
    parquet_dir: Path,
    video_ids: list[str],
    model: torch.nn.Module,
    device: torch.device,
    num_frames: int = 32,
    batch_size: int = 64,
) -> list[dict]:
    """Build detected events with model-predicted violation probability as score.

    Model class mapping:   logit index 0 = violation, 1 = compliance/non-violation
    Event label mapping:   label 1 = violation, label 0 = non-violation

    score = softmax(logits)[0]  →  P(violation)
    """
    model.eval()
    events: list[dict] = []

    for vid in video_ids:
        parquet_path = parquet_dir / f"{vid}_interactions.parquet"
        if not parquet_path.exists():
            logger.warning(f"Parquet not found: {parquet_path}")
            continue

        df = pd.read_parquet(parquet_path)

        for (v_track_id, roi), group in df.groupby(["v_track_id", "roi"]):
            try:
                all_frames: list[np.ndarray] = []
                all_vloc:   list[np.ndarray] = []
                for _, row in group.iterrows():
                    f = _to_frames(row["frames"])
                    v = _to_loc(row["v_loc_planar"])
                    n = min(len(f), len(v))
                    all_frames.append(f[:n])
                    all_vloc.append(v[:n])

                frames_cat = np.concatenate(all_frames)
                vloc_cat   = np.vstack(all_vloc)
                order       = np.argsort(frames_cat, kind="stable")
                frames_cat  = frames_cat[order]
                vloc_cat    = vloc_cat[order]

                start_frame = int(frames_cat[0])
                end_frame   = int(frames_cat[-1])
                pos_start   = vloc_cat[0].tolist()
                pos_end     = vloc_cat[-1].tolist()

                # Build (T, 3) vehicle and ped trajectories using the same helper as the dataset
                _, _, vehicle_feat, ped_feat = _build_group_trajectory(group)
                v_traj = _sample_trajectory(vehicle_feat, num_frames)  # (num_frames, 3)
                p_traj = _sample_trajectory(ped_feat, num_frames)      # (num_frames, 3)

                events.append({
                    "video_id":    vid,
                    "v_track_id":  int(v_track_id),
                    "roi":         str(roi),
                    "label":       0,          # unknown until matched
                    "frame_start": start_frame,
                    "frame_end":   end_frame,
                    "pos_start":   pos_start,
                    "pos_end":     pos_end,
                    "_v_traj":     v_traj,     # temporary — removed after inference
                    "_p_traj":     p_traj,     # temporary — removed after inference
                })
            except Exception as exc:
                logger.warning(f"Skipping ({vid}, {v_track_id}, {roi}): {exc}")

    logger.info(f"Running inference on {len(events)} events …")

    # Batch inference
    v_trajs = np.stack([e["_v_traj"] for e in events], axis=0)  # (N, T, 3)
    p_trajs = np.stack([e["_p_traj"] for e in events], axis=0)  # (N, T, 3)
    scores  = []

    for start in range(0, len(v_trajs), batch_size):
        v_batch = torch.from_numpy(v_trajs[start:start + batch_size]).to(device)
        p_batch = torch.from_numpy(p_trajs[start:start + batch_size]).to(device)
        with torch.no_grad():
            logits = model(v_batch, p_batch)             # (B, 2)
            probs  = F.softmax(logits, dim=1)            # (B, 2)
            # index 0 = violation probability
            scores.extend(probs[:, 0].cpu().tolist())

    # Write scores back and drop temporary trajectory arrays
    for ev, sc in zip(events, scores):
        ev["score"] = sc
        del ev["_v_traj"]
        del ev["_p_traj"]

    logger.info(f"Built {len(events)} events with model scores")
    return events


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate model with World-EIoU AP")
    parser.add_argument("--checkpoint",    type=Path, default=Path("checkpoints/best_model.pth"))
    parser.add_argument("--parquet-dir",   type=Path,
                        default=Path("/home/satria/Project/ATLAS/data/processed/interactions"))
    parser.add_argument("--labels-pkl",    type=Path,
                        default=Path("/home/satria/Project/ATLAS/data/raw/labels/test_labels.pkl"))
    parser.add_argument("--video-ids",     nargs="+", type=int,
                        default=list(range(2, 121, 2)),
                        help="Video numbers (even = test split, 2,4,...,120)")
    parser.add_argument("--num-frames",    type=int, default=32)
    parser.add_argument("--d-max",         type=float, default=5.0)
    parser.add_argument("--eiou-threshold",type=float, default=0.5)
    parser.add_argument("--batch-size",    type=int, default=64)
    args = parser.parse_args()

    video_ids = [f"video_{n:03d}" for n in args.video_ids]
    logger.info(f"Videos: {video_ids[:5]}{'...' if len(video_ids) > 5 else ''}")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = CrossAttentionModel(num_classes=2).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded checkpoint: {args.checkpoint.name}  "
                f"(epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.2f}%)")

    # Build events with model scores
    predictions = build_events_with_scores(
        args.parquet_dir, video_ids, model, device,
        num_frames=args.num_frames, batch_size=args.batch_size,
    )

    # Build GT events (positions proxied from detected)
    gt_events = build_gt_events(args.labels_pkl, video_ids, predictions)

    # Localization report
    loc = compute_localization_rate(
        predictions, gt_events,
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

    # AP report
    result = compute_map(
        predictions, gt_events,
        d_max=args.d_max,
        eiou_threshold=args.eiou_threshold,
    )

    print()
    print("=== AP Report ===")
    print(f"APv  : {result['APv']:.3f}")
    print(f"APn  : {result['APn']:.3f}")
    print(f"mAP  : {result['mAP']:.3f}")


if __name__ == "__main__":
    main()
