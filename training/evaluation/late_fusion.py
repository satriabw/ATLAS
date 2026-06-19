from __future__ import annotations

# ============================================================================
# TEMPORARY STOPGAP — late fusion of independently-trained traj + vision models.
#
# The end-to-end FusedModel currently scores *below* trajectory-only (APv 0.79 <
# 0.81): the dominant trajectory gradient starves the vision branch and the
# attention path dilutes vision's max-pool signal. Until FusedModel is fixed,
# we recover vision's complementary signal by averaging the two models' output
# probabilities with a hardcoded weight (APv ~0.82, above both single models).
#
# This reads the prediction CSVs emitted by evaluate_model.py — it does NOT
# re-run inference. Regenerate those CSVs first if the checkpoints changed.
# The proper fix is to repair FusedModel; delete this file once that lands.
# ============================================================================

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.ap_calculator import compute_map

W_TRAJ = 0.8  # weight on trajectory P(violation); vision gets (1 - W_TRAJ)
_EPS = 1e-6   # clamp probs off {0,1} so log-odds stays finite


def _load(path: Path) -> dict:
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r["video_id"], r["v_track_id"], r["roi"])
            rows[key] = (float(r["score"]), int(r["gt_label"]))
    return rows


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _fuse(traj: dict, vis: dict, w_traj: float, logodds: bool = False) -> list[dict]:
    preds = []
    for key in traj.keys() & vis.keys():
        s_traj, gt = traj[key]
        s_vis, _ = vis[key]
        if logodds:
            z = w_traj * _logit(s_traj) + (1.0 - w_traj) * _logit(s_vis)
            score = 1.0 / (1.0 + math.exp(-z))
        else:
            score = w_traj * s_traj + (1.0 - w_traj) * s_vis
        preds.append({"gt_label": gt, "score": score, "score_n": 1.0 - score, "eiou": 1.0})
    return preds


def main() -> None:
    ckpt = Path("checkpoints")
    parser = argparse.ArgumentParser(description="TEMPORARY late-fusion eval (traj + vision)")
    parser.add_argument("--traj-csv",   type=Path, default=ckpt / "best_model_predictions.csv")
    parser.add_argument("--vision-csv", type=Path, default=ckpt / "best_vision_predictions.csv")
    parser.add_argument("--w-traj",     type=float, default=W_TRAJ)
    parser.add_argument("--sweep",      action="store_true", help="print APv across a weight sweep")
    parser.add_argument("--logodds",    action="store_true", help="fuse in log-odds space instead of probability space")
    args = parser.parse_args()

    traj = _load(args.traj_csv)
    vis = _load(args.vision_csv)
    matched = traj.keys() & vis.keys()
    print(f"matched events: {len(matched)} (traj={len(traj)}, vision={len(vis)})")

    if args.sweep:
        print(f"\n=== weight sweep (APv, {'log-odds' if args.logodds else 'linear'}) ===")
        for w in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0]:
            apv = compute_map(_fuse(traj, vis, w, args.logodds))["APv"]
            print(f"  w_traj={w:.1f}: APv={apv:.4f}")

    result = compute_map(_fuse(traj, vis, args.w_traj, args.logodds))
    print(f"\n=== Late fusion (w_traj={args.w_traj}, {'log-odds' if args.logodds else 'linear'}) ===")
    print(f"APv  : {result['APv']:.3f}")
    print(f"APn  : {result['APn']:.3f}")
    print(f"mAP  : {result['mAP']:.3f}")


if __name__ == "__main__":
    main()
