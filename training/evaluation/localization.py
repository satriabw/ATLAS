"""Localization sanity check.

Checks whether detected events (from the interaction parquet) spatially and
temporally cover the GT events — independently of the label classifier.
"""

from __future__ import annotations

from .world_eiou import calculate_world_eiou


def compute_localization_rate(
    detected_events: list[dict],
    ground_truth: list[dict],
    d_max: float = 5.0,
    eiou_threshold: float = 0.5,
) -> dict:
    """Check what fraction of GT events are covered by any detected event.

    Labels are ignored — only temporal and spatial coverage matters.

    Args:
        detected_events: list of detected event dicts.
        ground_truth:    list of GT event dicts.
        d_max:           spatial proximity threshold (metres).
        eiou_threshold:  minimum World-EIoU to consider an event "localized".

    Returns:
        {
            "total_gt":          int,
            "localized":         int,
            "localization_rate": float,
            "per_roi":           {"TOP": float, "BOT": float},
        }
    """
    total_gt   = len(ground_truth)
    localized  = 0
    per_roi: dict[str, list[int]] = {"TOP": [0, 0], "BOT": [0, 0]}  # [localized, total]

    for gt in ground_truth:
        roi = gt.get("roi", "TOP")
        if roi not in per_roi:
            per_roi[roi] = [0, 0]
        per_roi[roi][1] += 1

        best = 0.0
        for det in detected_events:
            eiou = calculate_world_eiou(det, gt, d_max=d_max)
            if eiou > best:
                best = eiou
            if best >= eiou_threshold:
                break  # no need to check further

        if best >= eiou_threshold:
            localized += 1
            per_roi[roi][0] += 1

    loc_rate = localized / total_gt if total_gt > 0 else 0.0

    per_roi_rates: dict[str, float] = {}
    for roi, (loc, tot) in per_roi.items():
        per_roi_rates[roi] = loc / tot if tot > 0 else 0.0

    return {
        "total_gt":          total_gt,
        "localized":         localized,
        "localization_rate": loc_rate,
        "per_roi":           per_roi_rates,
    }
