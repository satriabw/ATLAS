"""World-coordinate EIoU metric for traffic violation detection.

World-EIoU = tIoU × (SPIoU_start + SPIoU_end) / 2
"""

import numpy as np


def calculate_tiou(pred_start: int, pred_end: int, gt_start: int, gt_end: int) -> float:
    """Temporal IoU between two frame ranges [start, end] (inclusive)."""
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    inter = max(0, inter_end - inter_start + 1)
    if inter == 0:
        return 0.0
    union = (pred_end - pred_start + 1) + (gt_end - gt_start + 1) - inter
    return float(inter) / float(union)


def calculate_spiou(pred_pos: list, gt_pos: list, d_max: float = 5.0) -> float:
    """Spatial Proximity IoU.

    Args:
        pred_pos: [x, y] in meters
        gt_pos:   [x, y] in meters
        d_max:    distance (m) at which score becomes 0

    Returns:
        Score in [0, 1].
    """
    dx = float(pred_pos[0]) - float(gt_pos[0])
    dy = float(pred_pos[1]) - float(gt_pos[1])
    dist = np.sqrt(dx * dx + dy * dy)
    return float(max(0.0, 1.0 - dist / d_max))


def calculate_world_eiou(pred_event: dict, gt_event: dict, d_max: float = 5.0) -> float:
    """World-coordinate EIoU between a predicted and a ground-truth event.

    Returns 0.0 immediately if video_id or roi don't match.

    Formula:
        World-EIoU = tIoU × (SPIoU_start + SPIoU_end) / 2
    """
    if pred_event["video_id"] != gt_event["video_id"]:
        return 0.0
    if pred_event["roi"] != gt_event["roi"]:
        return 0.0

    tiou = calculate_tiou(
        pred_event["frame_start"], pred_event["frame_end"],
        gt_event["frame_start"],  gt_event["frame_end"],
    )
    if tiou == 0.0:
        return 0.0

    sp_start = calculate_spiou(pred_event["pos_start"], gt_event["pos_start"], d_max)
    sp_end   = calculate_spiou(pred_event["pos_end"],   gt_event["pos_end"],   d_max)
    spatial  = (sp_start + sp_end) / 2.0

    return float(tiou * spatial)
