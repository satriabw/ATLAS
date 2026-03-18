"""Plot APv (and optionally APn) precision-recall curves.

Usage (from training/):
    python evaluation/plot_ap_curve.py \\
        --checkpoint checkpoints/best_model.pth \\
        --video-ids 1 3 5 7 9 \\
        --output ap_curve.png

Requires: matplotlib
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.ap_calculator import compute_ap, compute_pr_curve

logger = logging.getLogger(__name__)


def plot_curves(
    predictions: list[dict],
    eiou_threshold: float = 0.5,
    output_path: Path | None = None,
    show: bool = True,
) -> None:
    """Plot APv and APn PR curves from scored, labeled predictions.

    Each prediction must have: gt_label, score, score_n, eiou.
    """
    # Apply EIoU threshold: zero scores for correct-class but poorly-localized events
    thresholded = []
    for p in predictions:
        p2 = dict(p)
        predicted_class = 1 if p2["score"] >= 0.5 else 0
        if predicted_class == p2["gt_label"] and p2["eiou"] <= eiou_threshold:
            p2["score"]   = 0.0
            p2["score_n"] = 0.0
        thresholded.append(p2)

    recalls_v, precisions_v = compute_pr_curve(thresholded, target_class=1, score_key="score")
    recalls_n, precisions_n = compute_pr_curve(thresholded, target_class=0, score_key="score_n")

    apv = compute_ap(thresholded, target_class=1, score_key="score")
    apn = compute_ap(thresholded, target_class=0, score_key="score_n")
    map_ = (apv + apn) / 2.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Precision-Recall Curves  (EIoU threshold={eiou_threshold}, mAP={map_:.3f})")

    for ax, recalls, precisions, ap, title, color in [
        (axes[0], recalls_v, precisions_v, apv, "Violation (APv)", "tab:red"),
        (axes[1], recalls_n, precisions_n, apn, "Non-Violation (APn)", "tab:blue"),
    ]:
        ax.step(recalls, precisions, where="post", color=color, linewidth=2)
        ax.fill_between(recalls, precisions, alpha=0.15, color=color, step="post")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{title}  AP={ap:.3f}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")

    if show:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Plot APv / APn curves")
    parser.add_argument("--checkpoint",     type=Path, default=Path("checkpoints/best_model.pth"))
    parser.add_argument("--parquet-dir",    type=Path,
                        default=Path("/home/satria/Project/ATLAS/data/processed/interactions"))
    parser.add_argument("--labels-pkl",     type=Path,
                        default=Path("/home/satria/Project/ATLAS/data/raw/labels/test_labels.pkl"))
    parser.add_argument("--video-ids",      nargs="+", type=int,
                        default=list(range(2, 121, 2)))
    parser.add_argument("--num-frames",     type=int,   default=32)
    parser.add_argument("--eiou-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size",     type=int,   default=64)
    parser.add_argument("--output",         type=Path,  default=Path("ap_curve.png"))
    parser.add_argument("--no-show",        action="store_true",
                        help="Skip interactive display (save only)")
    args = parser.parse_args()

    video_ids = [f"video_{n:03d}" for n in args.video_ids]

    from evaluation.evaluate_model import build_events_with_scores
    from models import CrossAttentionModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossAttentionModel(num_classes=2).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded {args.checkpoint.name}  epoch={ckpt['epoch']}  val_acc={ckpt['val_acc']:.2f}%")

    predictions = build_events_with_scores(
        args.parquet_dir, args.labels_pkl, video_ids, model, device,
        num_frames=args.num_frames, batch_size=args.batch_size,
    )

    plot_curves(
        predictions,
        eiou_threshold=args.eiou_threshold,
        output_path=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
