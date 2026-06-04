from __future__ import annotations

import logging
import pickle

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dataset.labels import parse_train_label
from dataset.trajectory import DEFAULT_TOP_K, build_group_trajectory, resample_trajectory, padding_mask

logger = logging.getLogger(__name__)


# mirrors dataset/loader.py _get_trajectories pedestrian concatenation — keep in sync
def _build_ped_stack(ped_feats_raw, num_frames, top_k):
    p_arrs, p_masks = [], []
    for pf in ped_feats_raw[:top_k]:
        pf_r, p_len = resample_trajectory(pf, num_frames)
        p_arrs.append(pf_r)
        p_masks.append(padding_mask(p_len, num_frames))
    while len(p_arrs) < top_k:
        p_arrs.append(np.zeros((num_frames, 3), dtype=np.float32))
        p_masks.append(np.ones(num_frames, dtype=bool))
    return np.concatenate(p_arrs, axis=0), np.concatenate(p_masks, axis=0)



def _build_event(vid, v_track_id, roi, group, gt_label, num_frames, top_k):
    _, vehicle_feat_raw, ped_feats_raw = build_group_trajectory(group, top_k)
    v_arr, v_len = resample_trajectory(vehicle_feat_raw, num_frames)
    p_arr, p_mask = _build_ped_stack(ped_feats_raw, num_frames, top_k)

    return {
        "video_id":    vid,
        "v_track_id":  int(v_track_id),
        "roi":         str(roi),
        "gt_label":    gt_label,
        "eiou":        1.0,
        "_v_traj":     v_arr,
        "_p_traj":     p_arr,
        "_v_mask":     padding_mask(v_len, num_frames),
        "_p_mask":     p_mask,
    }


def _collect_events(parquet_dir, labels_pkl, video_ids, num_frames, top_k):
    with open(labels_pkl, "rb") as f:
        label_strings, _ = pickle.load(f)

    video_set   = set(video_ids)
    label_index = {}
    for s in label_strings:
        try:
            vid, tid, roi, lbl = parse_train_label(s)
        except ValueError as e:
            logger.warning(e)
            continue
        if vid in video_set:
            label_index[(vid, tid, roi)] = lbl

    events = []
    for vid in video_ids:
        parquet_path = parquet_dir / f"{vid}_interactions.parquet"
        if not parquet_path.exists():
            logger.warning(f"Parquet not found: {parquet_path}")
            continue
        df = pd.read_parquet(parquet_path)
        for (v_track_id, roi), group in df.groupby(["v_track_id", "roi"]):
            key = (vid, int(v_track_id), str(roi))
            if key not in label_index:
                continue
            try:
                events.append(_build_event(vid, v_track_id, roi, group, label_index[key], num_frames, top_k))
            except Exception as exc:
                logger.warning(f"Skipping ({vid}, {v_track_id}, {roi}): {exc}")
    return events


def _preload_h5_frames(events, h5_path, num_frames):
    from dataset.frames import load_frames_h5
    frame_tensors = {}
    with h5py.File(h5_path, 'r') as hf:
        for i, ev in enumerate(events):
            key = f"V{ev['video_id'][-3:]}_{ev['v_track_id']}_{ev['roi']}"
            frame_tensors[i] = load_frames_h5(hf, key, num_frames)
    return frame_tensors


def _run_inference(model, events, device, num_frames, batch_size, h5_path=None):
    v_trajs = np.stack([e["_v_traj"] for e in events])
    p_trajs = np.stack([e["_p_traj"] for e in events])
    v_masks = np.stack([e["_v_mask"] for e in events])
    p_masks = np.stack([e["_p_mask"] for e in events])

    if h5_path is not None:
        logger.info("Pre-loading frames from H5 …")
        preloaded_frames = _preload_h5_frames(events, h5_path, num_frames)

    scores = []
    for start in range(0, len(v_trajs), batch_size):
        sl       = slice(start, start + batch_size)
        v_batch  = torch.from_numpy(v_trajs[sl]).to(device)
        p_batch  = torch.from_numpy(p_trajs[sl]).to(device)
        vm_batch = torch.from_numpy(v_masks[sl]).to(device)
        pm_batch = torch.from_numpy(p_masks[sl]).to(device)

        with torch.no_grad():
            if h5_path is not None:
                end = min(start + batch_size, len(v_trajs))
                frames_batch = torch.stack(
                    [preloaded_frames[i] for i in range(start, end)]
                ).to(device)
                logits = model(v_batch, p_batch, frames_batch, vm_batch, pm_batch)
            else:
                logits = model(v_batch, p_batch, vm_batch, pm_batch)
            probs = F.softmax(logits, dim=1)
            scores.extend(probs[:, 0].cpu().tolist())

    return scores


def build_events_with_scores(
    parquet_dir, labels_pkl, video_ids, model, device,
    num_frames=32, top_k=DEFAULT_TOP_K, batch_size=64, h5_path=None,
):
    events = _collect_events(parquet_dir, labels_pkl, video_ids, num_frames, top_k)
    if h5_path is not None:
        with h5py.File(h5_path, 'r') as hf:
            h5_keys = set(hf.keys())
        before = len(events)
        events = [e for e in events if f"V{e['video_id'][-3:]}_{e['v_track_id']}_{e['roi']}" in h5_keys]
        skipped = before - len(events)
        if skipped:
            logger.warning(f"Skipped {skipped} events with no h5 entry ({len(events)} remain)")
    logger.info(f"Running inference on {len(events)} labeled events …")
    if not events:
        return events

    model.eval()
    scores = _run_inference(model, events, device, num_frames, batch_size, h5_path)

    for ev, sc in zip(events, scores):
        ev["score"]   = sc
        ev["score_n"] = 1.0 - sc
        del ev["_v_traj"], ev["_p_traj"], ev["_v_mask"], ev["_p_mask"]

    logger.info(f"Built {len(events)} scored events")
    return events
