"""Test-set APv for the centered-window trajectory model (2026-06-17).

Mirrors evaluate_model.py's metric (compute_map; eiou is 1.0 for trajectory-only
so APv == plain PR-AP) but feeds events through the centered-window pipeline
(dataset/centered_window.py) instead of the production whole-track resample, so
the centered checkpoint is scored on the data it was trained on. Test set =
even videos 2..120, test_labels.pkl. See docs/centered_window_experiment.md.
"""
import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.centered_window import load_centered_dataset
from dataset.trajectory import DEFAULT_TOP_K
from models import CrossAttentionModel, PooledFusedModel, VisionOnlyModel
from evaluation.ap_calculator import compute_map

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window",     type=int, default=32, help="centered window length (even); half=window//2")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="defaults to checkpoints/best_traj_centered_w{window}.pth")
    ap.add_argument("--data-root",  default="/home/satria/Project/ATLAS")
    ap.add_argument("--label-file", default="test")
    ap.add_argument("--video-ids",  nargs="+", type=int, default=list(range(2, 121, 2)))
    ap.add_argument("--top-k",      type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--model-type", choices=["cross_attention", "fused_pooled", "vision"], default="cross_attention")
    ap.add_argument("--h5",         default=None, help="centered-crop vision h5 (fused/vision)")
    ap.add_argument("--ground",     action="store_true", help="grounding mask channels (vision)")
    ap.add_argument("--zero-masks", action="store_true", help="zero the mask channels (ungrounded control)")
    ap.add_argument("--out-csv",    type=Path, default=None, help="write per-event predictions CSV")
    args = ap.parse_args()

    assert args.window % 2 == 0, "--window must be even"
    half = args.window // 2
    if args.checkpoint is None:
        args.checkpoint = Path(f"checkpoints/best_traj_centered_w{args.window}.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fused  = args.model_type == "fused_pooled"
    vision = args.model_type == "vision"
    ds = load_centered_dataset(Path(args.data_root), args.label_file,
                               top_k=args.top_k, video_filter=args.video_ids, half=half,
                               h5_name=args.h5 if (fused or vision) else None,
                               ground=(vision and args.ground), zero_masks=args.zero_masks)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if fused:
        model = PooledFusedModel(num_classes=2, top_k=args.top_k, num_frames=args.window).to(device)
    elif vision:
        model = VisionOnlyModel(num_classes=2, num_frames=args.window, backbone="resnet18").to(device)
    else:
        model = CrossAttentionModel(num_classes=2, top_k=args.top_k, num_frames=args.window).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for b in loader:
            if fused:
                logits = model(b["vehicle_feat"].to(device), b["ped_feat"].to(device),
                               b["frames"].to(device),
                               b["v_padding_mask"].to(device), b["p_padding_mask"].to(device))
            elif vision:
                logits = model(b["frames"].to(device))
            else:
                logits = model(b["vehicle_feat"].to(device), b["ped_feat"].to(device),
                               b["v_padding_mask"].to(device), b["p_padding_mask"].to(device))
            scores = torch.softmax(logits, dim=1)[:, 0].cpu().tolist()
            for i, sc in enumerate(scores):
                preds.append({
                    "video_id": b["video_id"][i], "v_track_id": int(b["tracking_id"][i]),
                    "roi": b["roi"][i], "gt_label": int(b["label"][i]),
                    "score": sc, "score_n": 1.0 - sc, "eiou": 1.0,
                    "predicted_label": 0 if sc >= 0.5 else 1,
                })

    res = compute_map(preds, eiou_threshold=0.5)
    print(f"\n=== CENTERED test (n={len(preds)}) ===")
    print(f"APv : {res['APv']:.4f}")
    print(f"APn : {res['APn']:.4f}")
    print(f"mAP : {res['mAP']:.4f}")

    if args.out_csv is not None:
        import csv
        with open(args.out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(preds[0].keys()))
            w.writeheader(); w.writerows(preds)
        print(f"CSV → {args.out_csv}")


if __name__ == "__main__":
    main()
