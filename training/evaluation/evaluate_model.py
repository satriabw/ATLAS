from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.trajectory import DEFAULT_TOP_K
from models import CrossAttentionModel, FusedModel, VisionOnlyModel, PooledFusedModel
from evaluation.ap_calculator import compute_map, compute_pr_curve
from evaluation.inference import build_events_with_scores

logger = logging.getLogger(__name__)


def _save_outputs(predictions, result, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{stem}_predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["video_id", "v_track_id", "roi", "gt_label",
                           "score", "score_n", "predicted_label"]
        )
        writer.writeheader()
        for ev in predictions:
            writer.writerow({
                "video_id":        ev["video_id"],
                "v_track_id":      ev["v_track_id"],
                "roi":             ev["roi"],
                "gt_label":        ev["gt_label"],
                "score":           round(ev["score"],   6),
                "score_n":         round(ev["score_n"], 6),
                "predicted_label": 0 if ev["score"] >= 0.5 else 1,
            })
    print(f"Predictions  → {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, (cls, score_key, label, ap_key) in zip(axes, [
            (0, "score",   "Violation",  "APv"),
            (1, "score_n", "Compliance", "APn"),
        ]):
            r, p = compute_pr_curve(predictions, cls, score_key=score_key)
            ax.plot(r, p)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"{label}  (AP={result[ap_key]:.3f})")
            ax.grid(True, alpha=0.3)

        fig.suptitle(stem)
        fig.tight_layout()
        pr_path = output_dir / f"{stem}_pr.png"
        fig.savefig(pr_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"PR curve     → {pr_path}")
    except ImportError:
        logger.warning("matplotlib not installed — skipping PR curve plot")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate model with World-EIoU AP")
    parser.add_argument("--checkpoint",     type=Path, default=None)
    parser.add_argument("--model-type",     choices=["cross_attention", "fused", "vision", "fused_pooled"], default="cross_attention")
    parser.add_argument("--parquet-dir",    type=Path, default=Path("/home/satria/Project/ATLAS/data/processed/interactions"))
    parser.add_argument("--labels-pkl",     type=Path, default=Path("/home/satria/Project/ATLAS/data/raw/labels/test_labels.pkl"))
    parser.add_argument("--video-ids",      nargs="+", type=int, default=list(range(2, 121, 2)))
    parser.add_argument("--overfit",        action="store_true")
    parser.add_argument("--num-frames",     type=int,   default=32)
    parser.add_argument("--top-k",          type=int,   default=DEFAULT_TOP_K)
    parser.add_argument("--eiou-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size",     type=int,   default=16)
    parser.add_argument("--output-dir",     type=Path, default=Path("checkpoints"))
    parser.add_argument("--no-save",        action="store_true", help="Skip CSV and PR curve output")
    args = parser.parse_args()

    if args.overfit:
        video_ids = ["video_001"]
        args.labels_pkl = args.labels_pkl.parent / "train_labels.pkl"
        logger.info("Overfit mode: evaluating on video_001 only, using train_labels.pkl")
    else:
        video_ids = [f"video_{n:03d}" for n in args.video_ids]
    logger.info(f"Videos: {video_ids[:5]}{'...' if len(video_ids) > 5 else ''}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if args.checkpoint is None:
        ckpt_name = {"fused": "best_fused.pth", "vision": "best_vision.pth"}.get(args.model_type, "best_model.pth")
        args.checkpoint = Path("checkpoints") / ckpt_name

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # auto-detect model type from checkpoint, fall back to --model-type
    model_type = ckpt.get("model_type", args.model_type)
    if model_type != args.model_type:
        logger.info(f"Checkpoint model_type='{model_type}' overrides --model-type='{args.model_type}'")

    if model_type == "vision":
        model = VisionOnlyModel(num_classes=2, num_frames=args.num_frames,
                                backbone=ckpt.get("backbone", "resnet18")).to(device)
    elif model_type == "fused":
        model = FusedModel(num_classes=2, top_k=args.top_k, num_frames=args.num_frames).to(device)
    elif model_type == "fused_pooled":
        model = PooledFusedModel(num_classes=2, top_k=args.top_k, num_frames=args.num_frames).to(device)
    else:
        model = CrossAttentionModel(num_classes=2, top_k=args.top_k, num_frames=args.num_frames).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded checkpoint: {args.checkpoint.name} (epoch {ckpt['epoch']}, val_acc={ckpt.get('val_acc', float('nan')):.2f}%)")

    # h5 file name comes from the checkpoint so evaluation matches the
    # representation the model was trained on (R0 frames.h5 vs R2 frames_r2.h5).
    h5_name = ckpt.get("h5", "frames.h5")
    h5_path = args.parquet_dir.parent.parent / "raw" / "video" / h5_name if model_type in ("fused", "vision", "fused_pooled") else None
    predictions = build_events_with_scores(
        args.parquet_dir, args.labels_pkl, video_ids, model, device,
        num_frames=args.num_frames, top_k=args.top_k,
        batch_size=args.batch_size,
        h5_path=h5_path,
        vision_only=(model_type == "vision"),
    )

    result = compute_map(predictions, eiou_threshold=args.eiou_threshold)
    print()
    print("=== AP Report ===")
    print(f"APv  : {result['APv']:.3f}")
    print(f"APn  : {result['APn']:.3f}")
    print(f"mAP  : {result['mAP']:.3f}")

    if not args.no_save:
        print()
        _save_outputs(predictions, result, args.output_dir, args.checkpoint.stem)


if __name__ == "__main__":
    main()
