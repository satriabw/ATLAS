from __future__ import annotations

import logging
import pickle

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dataset.labels import parse_train_label
from dataset.tracking import group_grid_boxes, parse_tracking
from dataset.trajectory import DEFAULT_TOP_K, build_group_trajectory, resample_trajectory, padding_mask

logger = logging.getLogger(__name__)


# mirrors dataset/loader.py _get_trajectories pedestrian concatenation — keep in sync
def _build_ped_stack(ped_feats_raw, num_frames, top_k, stretch, feat_dim):
    p_arrs, p_masks = [], []
    for pf in ped_feats_raw[:top_k]:
        pf_r, p_len = resample_trajectory(pf, num_frames, stretch=stretch)
        p_arrs.append(pf_r)
        p_masks.append(padding_mask(p_len, num_frames))
    while len(p_arrs) < top_k:
        p_arrs.append(np.zeros((num_frames, feat_dim), dtype=np.float32))
        p_masks.append(np.ones(num_frames, dtype=bool))
    return np.concatenate(p_arrs, axis=0), np.concatenate(p_masks, axis=0)



def _build_event(vid, v_track_id, roi, group, gt_label, num_frames, top_k, stretch, track_frames):
    _, vehicle_feat_raw, ped_feats_raw, ped_ids = build_group_trajectory(
        group, top_k, with_time=track_frames is not None)
    v_arr, v_len = resample_trajectory(vehicle_feat_raw, num_frames, stretch=stretch)
    p_arr, p_mask = _build_ped_stack(ped_feats_raw, num_frames, top_k, stretch,
                                     feat_dim=vehicle_feat_raw.shape[1])

    boxes = None
    if track_frames is not None:
        # Same grid build_h5 used to sample the frames for this group — keep in
        # sync with dataset/loader.py.
        all_f = np.concatenate([np.asarray(f).ravel() for f in group['frames']])
        grid = np.linspace(int(all_f.min()), int(all_f.max()), num_frames, dtype=int)
        boxes = group_grid_boxes(track_frames, grid, int(v_track_id), ped_ids)

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
        "_boxes":      boxes,
    }


def _collect_events(parquet_dir, labels_pkl, video_ids, num_frames, top_k, stretch, tracking_dir):
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
        track_frames = None
        if tracking_dir is not None:
            tracking_path = tracking_dir / f"{vid}.txt"
            if tracking_path.exists():
                track_frames = parse_tracking(tracking_path)
            else:
                logger.warning(f"Tracking not found: {tracking_path}")
        df = pd.read_parquet(parquet_path)
        for (v_track_id, roi), group in df.groupby(["v_track_id", "roi"]):
            key = (vid, int(v_track_id), str(roi))
            if key not in label_index:
                continue
            try:
                events.append(_build_event(vid, v_track_id, roi, group, label_index[key], num_frames, top_k, stretch, track_frames))
            except Exception as exc:
                logger.warning(f"Skipping ({vid}, {v_track_id}, {roi}): {exc}")
    return events


def _load_h5_batch(hf, events, start, end, num_frames):
    from dataset.frames import load_frames_h5
    tensors = []
    for ev in events[start:end]:
        key = f"V{ev['video_id'][-3:]}_{ev['v_track_id']}_{ev['roi']}"
        tensors.append(load_frames_h5(hf, key, num_frames, roi=ev['roi'], boxes=ev['_boxes']))
    return torch.stack(tensors)


def _run_inference(model, events, device, num_frames, batch_size, h5_path=None, vision_only=False):
    v_trajs = np.stack([e["_v_traj"] for e in events])
    p_trajs = np.stack([e["_p_traj"] for e in events])
    v_masks = np.stack([e["_v_mask"] for e in events])
    p_masks = np.stack([e["_p_mask"] for e in events])

    hf = h5py.File(h5_path, 'r') if h5_path is not None else None
    try:
        scores = []
        for start in range(0, len(v_trajs), batch_size):
            end      = min(start + batch_size, len(v_trajs))
            sl       = slice(start, end)
            v_batch  = torch.from_numpy(v_trajs[sl]).to(device)
            p_batch  = torch.from_numpy(p_trajs[sl]).to(device)
            vm_batch = torch.from_numpy(v_masks[sl]).to(device)
            pm_batch = torch.from_numpy(p_masks[sl]).to(device)

            with torch.no_grad():
                if vision_only:
                    frames_batch = _load_h5_batch(hf, events, start, end, num_frames).to(device)
                    logits = model(frames_batch)
                elif hf is not None:
                    frames_batch = _load_h5_batch(hf, events, start, end, num_frames).to(device)
                    logits = model(v_batch, p_batch, frames_batch, vm_batch, pm_batch)
                else:
                    logits = model(v_batch, p_batch, vm_batch, pm_batch)
                probs = F.softmax(logits, dim=1)
                scores.extend(probs[:, 0].cpu().tolist())
    finally:
        if hf is not None:
            hf.close()

    return scores


def build_events_with_scores(
    parquet_dir, labels_pkl, video_ids, model, device,
    num_frames=32, top_k=DEFAULT_TOP_K, batch_size=64, h5_path=None, vision_only=False,
):
    # Mirror ViolationDataset: stretch-resample trajectories and build grounding
    # boxes whenever frames (h5) are used.
    tracking_dir = parquet_dir.parent.parent / "raw" / "tracking" if h5_path is not None else None
    events = _collect_events(parquet_dir, labels_pkl, video_ids, num_frames, top_k,
                             stretch=h5_path is not None, tracking_dir=tracking_dir)
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
    scores = _run_inference(model, events, device, num_frames, batch_size, h5_path, vision_only)

    for ev, sc in zip(events, scores):
        ev["score"]   = sc
        ev["score_n"] = 1.0 - sc
        del ev["_v_traj"], ev["_p_traj"], ev["_v_mask"], ev["_p_mask"], ev["_boxes"]

    logger.info(f"Built {len(events)} scored events")
    return events
