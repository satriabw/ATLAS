"""Complementarity count between two models' prediction CSVs.

Answers: of the events model A gets wrong, how many does model B get right?
That count is the ceiling on what *any* fusion of the two can add over A alone.
Reads CSVs emitted by evaluate_model.py (does not re-run inference).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _load(path: Path) -> dict:
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r["video_id"], r["v_track_id"], r["roi"])
            rows[key] = (float(r["score"]), int(r["gt_label"]), int(r["predicted_label"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Complementarity count between two prediction CSVs")
    parser.add_argument("--traj-csv",   type=Path, default=Path("checkpoints/best_model_predictions.csv"))
    parser.add_argument("--vision-csv", type=Path, default=Path("checkpoints/best_vision_predictions.csv"))
    args = parser.parse_args()

    traj, vis = _load(args.traj_csv), _load(args.vision_csv)
    keys = sorted(traj.keys() & vis.keys())
    print(f"Joint events: {len(keys)} (traj: {len(traj)}, vision: {len(vis)})")

    counts = {"both_right": 0, "both_wrong": 0, "traj_only_right": 0, "vis_only_right": 0}
    vis_rescues = []  # the money list: traj wrong, vision right
    for key in keys:
        t_score, gt, t_pred = traj[key]
        v_score, _, v_pred = vis[key]
        t_ok, v_ok = t_pred == gt, v_pred == gt
        if t_ok and v_ok:
            counts["both_right"] += 1
        elif t_ok:
            counts["traj_only_right"] += 1
        elif v_ok:
            counts["vis_only_right"] += 1
            vis_rescues.append((key, gt, t_score, v_score))
        else:
            counts["both_wrong"] += 1

    n = len(keys)
    print(f"\ntraj acc: {(counts['both_right'] + counts['traj_only_right']) / n:.4f}, "
          f"vision acc: {(counts['both_right'] + counts['vis_only_right']) / n:.4f}, "
          f"oracle acc: {(n - counts['both_wrong']) / n:.4f}")
    for k, v in counts.items():
        print(f"{k:18s} {v:5d}  ({v / n:.2%})")

    by_class = {0: 0, 1: 0}
    for _, gt, _, _ in vis_rescues:
        by_class[gt] += 1
    print(f"\nVision rescues traj on {len(vis_rescues)} events "
          f"({by_class[0]} violations, {by_class[1]} compliance):")
    for (vid, tid, roi), gt, t_score, v_score in vis_rescues[:30]:
        cls = "violation" if gt == 0 else "compliance"
        print(f"  {vid} track={tid:>5s} {roi:3s} {cls:10s}  P_viol traj={t_score:.3f} vis={v_score:.3f}")
    if len(vis_rescues) > 30:
        print(f"  ... and {len(vis_rescues) - 30} more")


if __name__ == "__main__":
    main()
