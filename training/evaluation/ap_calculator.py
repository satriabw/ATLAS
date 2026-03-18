"""APv, APn, mAP computation aligned with the original Crosswalk ap_cal.py.

Each prediction must carry:
    gt_label  : int   — 1 (violation) or 0 (non-violation)
    score     : float — P(violation)  from model softmax
    score_n   : float — P(non-violation) = 1 - score
    eiou      : float — World-EIoU with the corresponding GT event

AP is computed per class with the class-specific confidence score:
    APv  →  sorted by score   (P(violation))
    APn  →  sorted by score_n (P(non-violation))

If eiou <= eiou_threshold AND the predicted class matches gt_label,
both scores are zeroed (correct class, poor localization — same as
the original ap_cal.py zeroing logic).

AP is computed as a step-function sum (Σ ΔR × P), matching sklearn's
average_precision_score and the original ap_cal.py behaviour.
"""

from __future__ import annotations

import numpy as np


def compute_ap(
    predictions: list[dict],
    target_class: int,
    n_gt: int | None = None,
    score_key: str = "score",
) -> float:
    """Compute AP for one class using a precision-recall curve.

    A prediction is a TP if gt_label == target_class.
    Predictions are sorted by score_key descending.

    Args:
        predictions:  list of event dicts with gt_label and score fields.
        target_class: class index to compute AP for (1=violation, 0=non-viol).
        n_gt:         total GT count for this class (recall denominator).
                      Defaults to TP count found in predictions.
        score_key:    dict key for the ranking score.

    Returns AP in [0, 1].
    """
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
    """Return (recalls, precisions) arrays for one class.

    Useful for plotting PR curves. Uses the same logic as compute_ap.
    """
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

    precisions_arr = np.concatenate([[1.0], precisions])
    recalls_arr    = np.concatenate([[0.0], recalls])
    return recalls_arr, precisions_arr


def compute_map(
    predictions: list[dict],
    eiou_threshold: float = 0.5,
) -> dict:
    """Compute APv, APn, and mAP.

    Aligned with the original ap_cal.py:
      - APv uses score   = P(violation)
      - APn uses score_n = P(non-violation) = 1 - score
      - If eiou <= eiou_threshold AND predicted class == gt_label:
        zero both scores (correct class but poor localization → penalized).

    Each prediction must have: gt_label, score, score_n, eiou.

    Returns {"APv": float, "APn": float, "mAP": float}.
    """
    thresholded = []
    for p in predictions:
        p2 = dict(p)
        predicted_class = 1 if p2["score"] >= 0.5 else 0
        if predicted_class == p2["gt_label"] and p2["eiou"] <= eiou_threshold:
            p2["score"]   = 0.0
            p2["score_n"] = 0.0
        thresholded.append(p2)

    apv = compute_ap(thresholded, target_class=1, score_key="score")
    apn = compute_ap(thresholded, target_class=0, score_key="score_n")
    map_ = (apv + apn) / 2.0

    return {"APv": apv, "APn": apn, "mAP": map_}
