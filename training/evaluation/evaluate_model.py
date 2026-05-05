"""Run model inference then compute APv, APn, mAP.

Usage (from training/):
    # CrossAttentionModel (trajectory only)
    python evaluation/evaluate_model.py \
        --checkpoint checkpoints/best_model.pth \
        --parquet-dir /home/satria/Project/ATLAS/data/processed/interactions \
        --labels-pkl  /home/satria/Project/ATLAS/data/raw/labels/test_labels.pkl

    # FusedModel (trajectory + video)
    python evaluation/evaluate_model.py \
        --checkpoint checkpoints/best_fused.pth \
        --model-type fused \
        --video-dir  /home/satria/Project/ATLAS/data/raw/video \
        --parquet-dir /home/satria/Project/ATLAS/data/processed/interactions \
        --labels-pkl  /home/satria/Project/ATLAS/data/raw/labels/test_labels.pkl

    # Overfit mode (videos 001-010, uses train labels)
    python evaluation/evaluate_model.py \
        --checkpoint checkpoints/best_model.pth \
        --model-type fused \
        --video-dir  /home/satria/Project/ATLAS/data/raw/video \
        --labels-pkl  /home/satria/Project/ATLAS/data/raw/labels/train_labels.pkl \
        --overfit
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
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.violation_dataset import (
    _build_group_trajectory,
    _load_frames,
    _resample_trajectory,
    _to_frames,
    _to_loc,
    DEFAULT_TOP_K,
)
from models import CrossAttentionModel, FusedModel
from evaluation.ap_calculator import compute_map

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def _parse_label_string(s: str):
    """Parse 'V001I00002S1D0R0A1' -> (video_id, track_id, roi, label).

    Keeps the same convention as the training dataset: label=0 → violation,
    label=1 → compliance.  compute_map uses target_class=0 for APv.
    """
    m = re.match(r'V(\d+)I(\d+)S(\d)D\d+R\d+A(\d)', s)
    if not m:
        raise ValueError(f"Cannot parse label string: {s!r}")
    video_id   = f"video_{int(m.group(1)):03d}"
    track_id   = int(m.group(2))
    roi        = 'BOT' if m.group(3) == '1' else 'TOP'
    label      = int(m.group(4))   # 0=violation, 1=compliance (matches training)
    return video_id, track_id, roi, label


# ---------------------------------------------------------------------------
# Pedestrian stack helper
# ---------------------------------------------------------------------------

def _build_ped_stack(
    ped_feats_raw: list[np.ndarray],
    num_frames: int,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample and stack top-K ped features.

    Returns:
        ped_feat      : (top_k * num_frames, 3)
        p_padding_mask: (top_k * num_frames,) bool — True = padded
    """
    p_arrs, p_masks = [], []

    for pf in ped_feats_raw[:top_k]:
        pf_r, p_len = _resample_trajectory(pf, num_frames)
        mask = np.zeros(num_frames, dtype=bool)
        mask[p_len:] = True
        p_arrs.append(pf_r)
        p_masks.append(mask)

    while len(p_arrs) < top_k:
        p_arrs.append(np.zeros((num_frames, 3), dtype=np.float32))
        p_masks.append(np.ones(num_frames, dtype=bool))

    return np.concatenate(p_arrs, axis=0), np.concatenate(p_masks, axis=0)


# ---------------------------------------------------------------------------
# Model inference — labeled events only
# ---------------------------------------------------------------------------

def build_events_with_scores(
    parquet_dir: Path,
    labels_pkl: Path,
    video_ids: list[str],
    model: torch.nn.Module,
    device: torch.device,
    num_frames: int = 32,
    top_k: int = DEFAULT_TOP_K,
    batch_size: int = 64,
    video_dir: Path | None = None,
) -> list[dict]:
    """Build events for GT-labeled interactions and score them with the model.

    Each returned event carries:
        gt_label  : int   — 1 (violation) or 0 (non-violation)
        score     : float — P(violation)
        score_n   : float — P(non-violation) = 1 - score
        eiou      : float — 1.0  (GT positions are proxied from this detection)
    """
    use_vision = video_dir is not None

    with open(labels_pkl, "rb") as f:
        label_strings, _ = pickle.load(f)

    video_set    = set(video_ids)
    label_index: dict[tuple, int] = {}
    for s in label_strings:
        try:
            vid, tid, roi, lbl = _parse_label_string(s)
        except ValueError as e:
            logger.warning(e)
            continue
        if vid in video_set:
            label_index[(vid, tid, roi)] = lbl

    model.eval()
    events: list[dict] = []

    for vid in video_ids:
        parquet_path = parquet_dir / f"{vid}_interactions.parquet"
        if not parquet_path.exists():
            logger.warning(f"Parquet not found: {parquet_path}")
            continue

        df = pd.read_parquet(parquet_path)

        for (v_track_id, roi), group in df.groupby(["v_track_id", "roi"]):
            key = (vid, int(v_track_id), str(roi))
            if key not in label_index:
                continue

            gt_label = label_index[key]

            try:
                # Collect vehicle world positions for event metadata
                all_frames, all_vloc = [], []
                for _, row in group.iterrows():
                    f = _to_frames(row["frames"])
                    v = _to_loc(row["v_loc_planar"])
                    n = min(len(f), len(v))
                    all_frames.append(f[:n])
                    all_vloc.append(v[:n])
                frames_cat = np.concatenate(all_frames)
                vloc_cat   = np.vstack(all_vloc)
                order      = np.argsort(frames_cat, kind="stable")
                frames_cat = frames_cat[order]
                vloc_cat   = vloc_cat[order]

                start_frame = int(frames_cat[0])
                end_frame   = int(frames_cat[-1])

                _, _, vehicle_feat_raw, ped_feats_raw = _build_group_trajectory(group, top_k)

                # Vehicle: resample and build mask
                v_arr, v_len = _resample_trajectory(vehicle_feat_raw, num_frames)
                v_mask = np.zeros(num_frames, dtype=bool)
                v_mask[v_len:] = True

                # Pedestrians: resample, stack, mask
                p_arr, p_mask = _build_ped_stack(ped_feats_raw, num_frames, top_k)

                events.append({
                    "video_id":    vid,
                    "v_track_id":  int(v_track_id),
                    "roi":         str(roi),
                    "gt_label":    gt_label,
                    "frame_start": start_frame,
                    "frame_end":   end_frame,
                    "pos_start":   vloc_cat[0].tolist(),
                    "pos_end":     vloc_cat[-1].tolist(),
                    "eiou":        1.0,
                    "_v_traj":     v_arr,
                    "_p_traj":     p_arr,
                    "_v_mask":     v_mask,
                    "_p_mask":     p_mask,
                })
            except Exception as exc:
                logger.warning(f"Skipping ({vid}, {v_track_id}, {roi}): {exc}")

    logger.info(f"Running inference on {len(events)} labeled events …")
    if not events:
        return events

    v_trajs = np.stack([e["_v_traj"] for e in events])
    p_trajs = np.stack([e["_p_traj"] for e in events])
    v_masks = np.stack([e["_v_mask"] for e in events])
    p_masks = np.stack([e["_p_mask"] for e in events])
    scores: list[float] = []

    for start in range(0, len(v_trajs), batch_size):
        sl = slice(start, start + batch_size)
        v_batch  = torch.from_numpy(v_trajs[sl]).to(device)
        p_batch  = torch.from_numpy(p_trajs[sl]).to(device)
        vm_batch = torch.from_numpy(v_masks[sl]).to(device)
        pm_batch = torch.from_numpy(p_masks[sl]).to(device)

        with torch.no_grad():
            if use_vision:
                batch_events = events[start : start + batch_size]
                frame_tensors = []
                for ev in batch_events:
                    vid_path = video_dir / f"{ev['video_id']}.avi"
                    frame_tensors.append(
                        _load_frames(vid_path, ev["frame_start"], ev["frame_end"], num_frames)
                    )
                frames_batch = torch.stack(frame_tensors).to(device)
                logits = model(v_batch, p_batch, frames_batch, vm_batch, pm_batch)
            else:
                logits = model(v_batch, p_batch, vm_batch, pm_batch)

            probs = F.softmax(logits, dim=1)
            scores.extend(probs[:, 0].cpu().tolist())  # P(violation) = index 0

    for ev, sc in zip(events, scores):
        ev["score"]   = sc
        ev["score_n"] = 1.0 - sc
        del ev["_v_traj"], ev["_p_traj"], ev["_v_mask"], ev["_p_mask"]

    logger.info(f"Built {len(events)} scored events")
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
    parser.add_argument("--checkpoint",     type=Path, default=Path("checkpoints/best_model.pth"))
    parser.add_argument("--model-type",     choices=["cross_attention", "fused"],
                        default="cross_attention",
                        help="cross_attention = trajectory only; fused = trajectory + video")
    parser.add_argument("--parquet-dir",    type=Path,
                        default=Path("/home/satria/Project/ATLAS/data/processed/interactions"))
    parser.add_argument("--labels-pkl",     type=Path,
                        default=Path("/home/satria/Project/ATLAS/data/raw/labels/test_labels.pkl"))
    parser.add_argument("--video-dir",      type=Path, default=None,
                        help="Directory with <video_id>.avi files (required for --model-type fused)")
    parser.add_argument("--video-ids",      nargs="+", type=int,
                        default=list(range(2, 121, 2)))
    parser.add_argument("--overfit",        action="store_true",
                        help="Restrict to videos 001-010 for overfit sanity check")
    parser.add_argument("--num-frames",     type=int,   default=32)
    parser.add_argument("--top-k",          type=int,   default=DEFAULT_TOP_K)
    parser.add_argument("--eiou-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size",     type=int,   default=16,
                        help="Batch size (keep small when loading video frames)")
    args = parser.parse_args()

    if args.overfit:
        video_ids = [f"video_{n:03d}" for n in range(1, 11)]
        args.labels_pkl = args.labels_pkl.parent / "train_labels.pkl"
        logger.info("Overfit mode: evaluating on videos 001-010, using train_labels.pkl")
    else:
        video_ids = [f"video_{n:03d}" for n in args.video_ids]
    logger.info(f"Videos: {video_ids[:5]}{'...' if len(video_ids) > 5 else ''}")

    if args.model_type == "fused" and args.video_dir is None:
        parser.error("--video-dir is required when --model-type fused")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if args.model_type == "fused":
        model = FusedModel(num_classes=2, top_k=args.top_k, num_frames=args.num_frames).to(device)
    else:
        model = CrossAttentionModel(num_classes=2, top_k=args.top_k, num_frames=args.num_frames).to(device)

    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded checkpoint: {args.checkpoint.name}  "
                f"(epoch {ckpt['epoch']}, val_acc={ckpt.get('val_acc', float('nan')):.2f}%)")

    predictions = build_events_with_scores(
        args.parquet_dir, args.labels_pkl, video_ids, model, device,
        num_frames=args.num_frames, top_k=args.top_k,
        batch_size=args.batch_size,
        video_dir=args.video_dir if args.model_type == "fused" else None,
    )

    result = compute_map(predictions, eiou_threshold=args.eiou_threshold)

    print()
    print("=== AP Report ===")
    print(f"APv  : {result['APv']:.3f}")
    print(f"APn  : {result['APn']:.3f}")
    print(f"mAP  : {result['mAP']:.3f}")


if __name__ == "__main__":
    main()
