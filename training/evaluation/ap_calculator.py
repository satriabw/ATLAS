from __future__ import annotations

import numpy as np

def _pr_points(predictions, target_class, n_pos, score_key):
    sorted_preds = sorted(predictions, key=lambda p: p[score_key], reverse=True)
    tp, fp = 0, 0
    precisions, recalls = [], []
    for pred in sorted_preds:
        if pred["gt_label"] == target_class:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_pos)
    return precisions, recalls


def _resolve_n_pos(predictions, target_class, n_gt, score_key):
    return n_gt if n_gt is not None else sum(
        1 for p in predictions if p[score_key] and p["gt_label"] == target_class
        # Use gt_label directly — score_key not needed for counting
    )


def compute_ap(predictions, target_class, n_gt=None, score_key="score"):
    if not predictions:
        return 0.0

    n_pos = n_gt if n_gt is not None else sum(
        1 for p in predictions if p["gt_label"] == target_class
    )
    if n_pos == 0:
        return 0.0

    precisions, recalls = _pr_points(predictions, target_class, n_pos, score_key)
    recalls_arr = np.concatenate([[0.0], recalls])
    ap = float(np.sum(np.diff(recalls_arr) * np.array(precisions)))
    return max(0.0, min(1.0, ap))


def compute_pr_curve(predictions, target_class, n_gt=None, score_key="score"):
    if not predictions:
        return np.array([0.0, 1.0]), np.array([1.0, 1.0])

    n_pos = n_gt if n_gt is not None else sum(
        1 for p in predictions if p["gt_label"] == target_class
    )
    if n_pos == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 1.0])

    precisions, recalls = _pr_points(predictions, target_class, n_pos, score_key)
    return np.concatenate([[0.0], recalls]), np.concatenate([[1.0], precisions])


def compute_map(predictions, eiou_threshold=0.5):
    thresholded = []
    for p in predictions:
        p2 = dict(p)
        predicted_class = 0 if p2["score"] >= 0.5 else 1
        if predicted_class == p2["gt_label"] and p2["eiou"] <= eiou_threshold:
            p2["score"]   = 0.0
            p2["score_n"] = 0.0
        thresholded.append(p2)

    apv = compute_ap(thresholded, target_class=0, score_key="score")
    apn = compute_ap(thresholded, target_class=1, score_key="score_n")
    return {"APv": apv, "APn": apn, "mAP": (apv + apn) / 2.0}
