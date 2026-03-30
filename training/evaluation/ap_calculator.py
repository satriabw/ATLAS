from __future__ import annotations

import numpy as np


def compute_ap(
    predictions: list[dict],
    target_class: int,
    n_gt: int | None = None,
    score_key: str = "score",
) -> float:
    if not predictions:
        return 0.0

    sorted_preds = sorted(predictions, key=lambda p: p[score_key], reverse=True)
    n_pos = n_gt if n_gt is not None else sum(
        1 for p in sorted_preds if p["gt_label"] == target_class
    )
    if n_pos == 0:
        return 0.0

    tp_cumsum = 0
    fp_cumsum = 0
    precisions: list[float] = []
    recalls:    list[float] = []

    for pred in sorted_preds:
        if pred["gt_label"] == target_class:
            tp_cumsum += 1
        else:
            fp_cumsum += 1
        precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
        recalls.append(tp_cumsum / n_pos)

    precisions_arr = np.array(precisions)
    recalls_arr    = np.concatenate([[0.0], recalls])
    ap = float(np.sum(np.diff(recalls_arr) * precisions_arr))
    return max(0.0, min(1.0, ap))


def compute_pr_curve(
    predictions: list[dict],
    target_class: int,
    n_gt: int | None = None,
    score_key: str = "score",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (recalls, precisions) arrays for one class."""
    if not predictions:
        return np.array([0.0, 1.0]), np.array([1.0, 1.0])

    sorted_preds = sorted(predictions, key=lambda p: p[score_key], reverse=True)
    n_pos = n_gt if n_gt is not None else sum(
        1 for p in sorted_preds if p["gt_label"] == target_class
    )
    if n_pos == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 1.0])

    tp_cumsum = 0
    fp_cumsum = 0
    precisions: list[float] = []
    recalls:    list[float] = []

    for pred in sorted_preds:
        if pred["gt_label"] == target_class:
            tp_cumsum += 1
        else:
            fp_cumsum += 1
        precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
        recalls.append(tp_cumsum / n_pos)

    return np.concatenate([[0.0], recalls]), np.concatenate([[1.0], precisions])


def compute_map(
    predictions: list[dict],
    eiou_threshold: float = 0.5,
) -> dict:
    """Compute APv, APn, mAP.

    APv uses score=P(violation), APn uses score_n=P(non-violation).
    Predictions with correct class but eiou <= threshold have scores zeroed (poor localization).
    """
    thresholded = []
    for p in predictions:
        p2 = dict(p)
        predicted_class = 1 if p2["score"] >= 0.5 else 0
        if predicted_class == p2["gt_label"] and p2["eiou"] <= eiou_threshold:
            p2["score"]   = 0.0
            p2["score_n"] = 0.0
        thresholded.append(p2)

    apv  = compute_ap(thresholded, target_class=1, score_key="score")
    apn  = compute_ap(thresholded, target_class=0, score_key="score_n")
    return {"APv": apv, "APn": apn, "mAP": (apv + apn) / 2.0}
