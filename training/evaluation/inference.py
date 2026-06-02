from __future__ import annotations

import logging
import pickle
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dataset.labels import parse_train_label
from dataset.trajectory import DEFAULT_TOP_K, build_group_trajectory, resample_trajectory, padding_mask, _to_frames, _to_loc
from dataset.frames import IMAGENET_MEAN, IMAGENET_STD

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


def _preload_video_frames(events, video_dir, num_frames, size=224):
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    vid_to_indices = defaultdict(list)
    for i, ev in enumerate(events):
        vid_to_indices[ev["video_id"]].append(i)

    blank_tensor = (torch.zeros(num_frames, 3, size, size) - mean) / std

    frame_tensors = {}
    for vid_id, ev_indices in vid_to_indices.items():
        vid_path = video_dir / f"{vid_id}.avi"
        if not vid_path.exists():
            logger.warning(f"Video not found: {vid_path}; substituting normalized black frames for {len(ev_indices)} events")
            for i in ev_indices:
                frame_tensors[i] = blank_tensor
            continue

        event_frame_idxs, all_needed = [], set()
        for i in ev_indices:
            ev   = events[i]
            idxs = np.linspace(ev["frame_start"], ev["frame_end"], num_frames, dtype=int).tolist()
            event_frame_idxs.append((i, idxs))
            all_needed.update(idxs)

        frame_cache = {}
        cap = cv2.VideoCapture(str(vid_path))
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for current_idx in range(total_frames):
                ret, frame = cap.read()
                if current_idx in all_needed:
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, (size, size))
                    else:
                        logger.warning(f"{vid_id}: cap.read() failed at frame {current_idx}; substituting black frame")
                        frame = np.zeros((size, size, 3), dtype=np.uint8)
                    frame_cache[current_idx] = frame
                    if len(frame_cache) == len(all_needed):
                        break
        finally:
            cap.release()

        missing = all_needed - set(frame_cache.keys())
        if missing:
            sample = sorted(missing)[:5]
            logger.warning(f"{vid_id}: {len(missing)} frame indices beyond video length ({total_frames} frames): {sample}{'...' if len(missing) > 5 else ''}")

        blank = np.zeros((size, size, 3), dtype=np.uint8)
        for i, idxs in event_frame_idxs:
            frames = np.stack([frame_cache.get(idx, blank) for idx in idxs])
            t = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
            frame_tensors[i] = (t - mean) / std

        logger.info(f"  {vid_id}: loaded {len(all_needed)} unique frames for {len(ev_indices)} events")

    return frame_tensors


def _build_event(vid, v_track_id, roi, group, gt_label, num_frames, top_k):
    all_frames, all_vloc = [], []
    for _, row in group.iterrows():
        f = _to_frames(row["frames"])
        v = _to_loc(row["v_loc_planar"])
        n = min(len(f), len(v))
        all_frames.append(f[:n])
        all_vloc.append(v[:n])
    frames_cat = np.concatenate(all_frames)
    vloc_cat   = np.vstack(all_vloc)
    order      = np.argsort(frames_cat, kind="stable")
    frames_cat = frames_cat[order]
    vloc_cat   = vloc_cat[order]

    _, _, vehicle_feat_raw, ped_feats_raw = build_group_trajectory(group, top_k)
    v_arr, v_len = resample_trajectory(vehicle_feat_raw, num_frames)
    p_arr, p_mask = _build_ped_stack(ped_feats_raw, num_frames, top_k)

    return {
        "video_id":    vid,
        "v_track_id":  int(v_track_id),
        "roi":         str(roi),
        "gt_label":    gt_label,
        "frame_start": int(frames_cat[0]),
        "frame_end":   int(frames_cat[-1]),
        "pos_start":   vloc_cat[0].tolist(),
        "pos_end":     vloc_cat[-1].tolist(),
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


def _run_inference(model, events, device, num_frames, batch_size, video_dir):
    use_vision = video_dir is not None

    v_trajs = np.stack([e["_v_traj"] for e in events])
    p_trajs = np.stack([e["_p_traj"] for e in events])
    v_masks = np.stack([e["_v_mask"] for e in events])
    p_masks = np.stack([e["_p_mask"] for e in events])

    if use_vision:
        logger.info("Pre-loading video frames (one pass per video) …")
        preloaded_frames = _preload_video_frames(events, video_dir, num_frames)

    scores = []
    for start in range(0, len(v_trajs), batch_size):
        sl       = slice(start, start + batch_size)
        v_batch  = torch.from_numpy(v_trajs[sl]).to(device)
        p_batch  = torch.from_numpy(p_trajs[sl]).to(device)
        vm_batch = torch.from_numpy(v_masks[sl]).to(device)
        pm_batch = torch.from_numpy(p_masks[sl]).to(device)

        with torch.no_grad():
            if use_vision:
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
    num_frames=32, top_k=DEFAULT_TOP_K, batch_size=64, video_dir=None,
):
    events = _collect_events(parquet_dir, labels_pkl, video_ids, num_frames, top_k)
    logger.info(f"Running inference on {len(events)} labeled events …")
    if not events:
        return events

    model.eval()
    scores = _run_inference(model, events, device, num_frames, batch_size, video_dir)

    for ev, sc in zip(events, scores):
        ev["score"]   = sc
        ev["score_n"] = 1.0 - sc
        del ev["_v_traj"], ev["_p_traj"], ev["_v_mask"], ev["_p_mask"]

    logger.info(f"Built {len(events)} scored events")
    return events
