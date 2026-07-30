"""Stage A — vision-only gate: train a shallow head on frozen VideoMAE ROI
features and report test APv (plan = ~/.claude/plans/dazzling-bouncing-summit.md).

Features come from scripts/precompute_videomae_roi_feats.py
(data/raw/video/videomae_roi_feats.h5, one (1536,) vector per event key). Labels
and the scene-level train/val split are reused from the standard loader so this
arm is directly comparable to prior vision arms (r2plus1d whole 0.659) and the
traj anchor (best_traj_whole_predictions.csv, 0.802).

Run from training/:
    python train_videomae_roi.py
"""
import argparse
import csv
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import load_violation_dataset
from train import _scene_split
from models.videomae_head import VideoMAEROIHead
from evaluation.ap_calculator import compute_map

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def _key(lbl):
    return f"V{lbl.video_id[-3:]}_{lbl.tracking_id}_{lbl.roi}"


def _collect(labels, feat_h5):
    """Return (X, meta) for labels present in the feature h5 (skip missing)."""
    X, meta, missing = [], [], 0
    for lbl in labels:
        k = _key(lbl)
        if k not in feat_h5:
            missing += 1
            continue
        X.append(feat_h5[k][:])
        meta.append((lbl.video_id, lbl.tracking_id, lbl.roi, int(lbl.annotation)))
    if missing:
        log.warning("Skipped %d labels with no VideoMAE feature", missing)
    return np.stack(X).astype(np.float32), meta


def _preds(scores, meta):
    return [{"video_id": m[0], "v_track_id": m[1], "roi": m[2], "gt_label": m[3],
             "score": float(s), "score_n": 1.0 - float(s), "eiou": 1.0}
            for s, m in zip(scores, meta)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="/home/satria/Project/ATLAS")
    ap.add_argument("--feats", default="data/raw/video/videomae_roi_feats.h5")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", default="checkpoints/best_videomae_roi_predictions.csv")
    args = ap.parse_args()

    root = Path(args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    feat_path = root / args.feats if not Path(args.feats).is_absolute() else Path(args.feats)
    feat_h5 = h5py.File(feat_path, "r")

    # labels + scene split reused from the standard loader (trajectory not needed)
    full = load_violation_dataset(root, "train", use_vision=False)
    tr_sub, va_sub, _ = _scene_split(full, args.seed)
    tr_labels = [full.labels[i] for i in tr_sub.indices]
    va_labels = [full.labels[i] for i in va_sub.indices]
    te = load_violation_dataset(root, "test", use_vision=False,
                                video_filter=list(range(2, 121, 2)))

    Xtr, mtr = _collect(tr_labels, feat_h5)
    Xva, mva = _collect(va_labels, feat_h5)
    Xte, mte = _collect(te.labels, feat_h5)
    log.info("train %d / val %d / test %d events", len(mtr), len(mva), len(mte))

    Ytr = torch.tensor([m[3] for m in mtr], dtype=torch.long)
    Xtr_t = torch.from_numpy(Xtr)
    Xva_t, Xte_t = torch.from_numpy(Xva).to(device), torch.from_numpy(Xte).to(device)

    loader = DataLoader(TensorDataset(Xtr_t, Ytr), batch_size=args.batch_size, shuffle=True)
    head = VideoMAEROIHead(in_dim=Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Vision-only convention (train.py): unweighted CE; val-APv selection handles imbalance.
    crit = nn.CrossEntropyLoss()

    best_state, best_va, patience = None, -1.0, 0
    for ep in range(args.epochs):
        head.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(head(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            sv = torch.softmax(head(Xva_t), dim=1)[:, 0].cpu().numpy()
        va_apv = compute_map(_preds(sv, mva))["APv"]
        if va_apv > best_va:
            best_va, patience = va_apv, 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            patience += 1
        if patience >= args.patience:
            log.info("Early stop at epoch %d", ep + 1)
            break

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        st = torch.softmax(head(Xte_t), dim=1)[:, 0].cpu().numpy()
    te_metrics = compute_map(_preds(st, mte))
    log.info("BEST val APv %.4f | TEST APv %.4f  APn %.4f", best_va,
             te_metrics["APv"], te_metrics["APn"])

    # save checkpoint + predictions CSV (columns match anchor CSVs for bootstrap_ci)
    ckpt_dir = Path(__file__).parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save({"model_state_dict": best_state, "val_apv": best_va,
                "test_apv": te_metrics["APv"]}, ckpt_dir / "best_videomae_roi.pth")

    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = Path(__file__).parent / out_csv
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "v_track_id", "roi", "gt_label", "score", "score_n", "predicted_label"])
        for s, m in zip(st, mte):
            w.writerow([m[0], m[1], m[2], m[3], f"{s:.6f}", f"{1 - s:.6f}", 0 if s >= 0.5 else 1])
    log.info("Wrote predictions → %s", out_csv)
    feat_h5.close()


if __name__ == "__main__":
    main()
