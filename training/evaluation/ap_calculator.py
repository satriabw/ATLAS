"""APv, APn, mAP computation using World-EIoU matching.

Matching is greedy (same logic as Crosswalk ap_cal.py):
  for each GT event → find the unmatched prediction with highest EIoU > 0
  assign GT label to that prediction; all remaining predictions → label 0.

AP is computed manually via precision-recall curve (no sklearn).
"""

from __future__ import annotations

import numpy as np
from .world_eiou import calculate_world_eiou


def match_predictions_to_gt(
    predictions: list[dict],
    ground_truth: list[dict],
    d_max: float = 5.0,
) -> list[dict]:
    """Match predictions to GT events and annotate each prediction.

    Adds two fields to every prediction dict (copies are returned):
        matched_label : int   — GT label if matched, else 0 (non-violation)
        eiou          : float — best World-EIoU found (0.0 if unmatched)

    Args:
        predictions:  list of predicted event dicts (must have 'score' field).
        ground_truth: list of GT event dicts (must have 'label' field).
        d_max:        spatial proximity threshold in metres.

    Returns:
        List of annotated prediction dicts (same order as input).
    """
    # Work on shallow copies so callers' dicts are not mutated.
    annotated = [dict(p) for p in predictions]
    for p in annotated:
        p["matched_label"] = 0
        p["eiou"] = 0.0

    matched_pred_indices: set[int] = set()

    for gt in ground_truth:
        best_eiou = 0.0
        best_idx  = -1

        for i, pred in enumerate(annotated):
            if i in matched_pred_indices:
                continue
            eiou = calculate_world_eiou(pred, gt, d_max=d_max)
            if eiou > best_eiou:
                best_eiou = eiou
                best_idx  = i

        if best_idx >= 0 and best_eiou > 0.0:
            annotated[best_idx]["matched_label"] = gt["label"]
            annotated[best_idx]["eiou"]          = best_eiou
            matched_pred_indices.add(best_idx)

    return annotated


def compute_ap(matched_predictions: list[dict], target_class: int) -> float:
    """Compute AP for one class using a precision-recall curve.

    A prediction is a TP if:
        matched_label == target_class  AND  eiou > 0  (already encoded by matching)

    Predictions are sorted by 'score' descending.

    Returns AP in [0, 1].
    """
    if not matched_predictions:
        return 0.0

    # Sort by score descending
    sorted_preds = sorted(matched_predictions, key=lambda p: p["score"], reverse=True)

    n_pos = sum(1 for p in sorted_preds if p["matched_label"] == target_class)
    if n_pos == 0:
        return 0.0

    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls    = []

    for pred in sorted_preds:
        if pred["matched_label"] == target_class:
            tp_cumsum += 1
        else:
            fp_cumsum += 1
        precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
        recalls.append(tp_cumsum / n_pos)

    # Area under PR curve via trapezoidal rule (matches sklearn's method)
    precisions = np.array(precisions, dtype=np.float64)
    recalls    = np.array(recalls,    dtype=np.float64)

    # Prepend (0, 1) sentinel so the curve starts correctly
    precisions = np.concatenate([[1.0], precisions])
    recalls    = np.concatenate([[0.0], recalls])

    ap = float(np.trapz(precisions, recalls))
    return max(0.0, min(1.0, ap))


def compute_map(
    predictions: list[dict],
    ground_truth: list[dict],
    d_max: float = 5.0,
    eiou_threshold: float = 0.5,
) -> dict:
    """Compute APv, APn, and mAP.

    A prediction is TP only if eiou > eiou_threshold AND label matches.

    Args:
        predictions:    predicted event dicts (each must have 'score' field).
        ground_truth:   GT event dicts (each must have 'label' field).
        d_max:          spatial proximity threshold (metres).
        eiou_threshold: minimum EIoU to count as a true positive.

    Returns:
        {"APv": float, "APn": float, "mAP": float}
    """
    matched = match_predictions_to_gt(predictions, ground_truth, d_max=d_max)

    # Apply eiou_threshold only to *matched* predictions (eiou > 0) whose
    # match quality is too low.  Truly unmatched predictions (eiou == 0.0)
    # keep matched_label=0 (non-violation) so they count as TP for APn.
    thresholded = []
    for p in matched:
        p2 = dict(p)
        if 0.0 < p2["eiou"] <= eiou_threshold:
            p2["matched_label"] = -1   # weak match → FP for both classes
        thresholded.append(p2)

    apv = compute_ap(thresholded, target_class=1)   # violation = 1
    apn = compute_ap(thresholded, target_class=0)   # non-violation = 0
    map_ = (apv + apn) / 2.0

    return {"APv": apv, "APn": apn, "mAP": map_}
