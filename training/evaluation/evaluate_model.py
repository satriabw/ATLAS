from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.trajectory import DEFAULT_TOP_K
from models import CrossAttentionModel, FusedModel
from evaluation.ap_calculator import compute_map
from evaluation.inference import build_events_with_scores

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate model with World-EIoU AP")
    parser.add_argument("--checkpoint",     type=Path, default=None)
    parser.add_argument("--model-type",     choices=["cross_attention", "fused"], default="cross_attention")
    parser.add_argument("--parquet-dir",    type=Path, default=Path("/home/satria/Project/ATLAS/data/processed/interactions"))
    parser.add_argument("--labels-pkl",     type=Path, default=Path("/home/satria/Project/ATLAS/data/raw/labels/test_labels.pkl"))
    parser.add_argument("--video-dir",      type=Path, default=None)
    parser.add_argument("--video-ids",      nargs="+", type=int, default=list(range(2, 121, 2)))
    parser.add_argument("--overfit",        action="store_true")
    parser.add_argument("--num-frames",     type=int,   default=32)
    parser.add_argument("--top-k",          type=int,   default=DEFAULT_TOP_K)
    parser.add_argument("--eiou-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size",     type=int,   default=16)
    args = parser.parse_args()

    if args.checkpoint is None:
        ckpt_name = "best_fused.pth" if args.model_type == "fused" else "best_model.pth"
        args.checkpoint = Path("checkpoints") / ckpt_name

    if args.overfit:
        video_ids = ["video_001"]
        args.labels_pkl = args.labels_pkl.parent / "train_labels.pkl"
        logger.info("Overfit mode: evaluating on video_001 only, using train_labels.pkl")
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

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded checkpoint: {args.checkpoint.name} (epoch {ckpt['epoch']}, val_acc={ckpt.get('val_acc', float('nan')):.2f}%)")

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
