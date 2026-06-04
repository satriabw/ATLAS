import numpy as np

DEFAULT_TOP_K = 5

def _to_loc(val) -> np.ndarray:
    arr = np.asarray(val)
    if arr.dtype == object:
        return np.stack(arr.tolist()).astype(np.float32)
    return arr.astype(np.float32).reshape(-1, 2)

def _to_scalar_seq(val) -> np.ndarray:
    return np.asarray(val, dtype=np.float32).ravel()

def _to_frames(val) -> np.ndarray:
    return np.asarray(val, dtype=np.int64).ravel()

def _to_dmin(val) -> float:
    return float(np.asarray(val, dtype=np.float64).ravel().min())

def _extract_row_arrays(rows_df):
    frames_parts, v_loc_parts, v_sp_parts, p_loc_parts, p_sp_parts = [], [], [], [], []
    for _, row in rows_df.iterrows():
        frames_parts.append(_to_frames(row['frames']))
        v_loc_parts.append(_to_loc(row['v_loc_planar']))
        v_sp_parts.append(_to_scalar_seq(row['v_speed']))
        p_loc_parts.append(_to_loc(row['p_loc_planar']))
        p_sp_parts.append(_to_scalar_seq(row['p_speed']))

    all_f = np.concatenate(frames_parts)
    order = np.argsort(all_f, kind='stable')
    return (
        all_f[order],
        np.vstack(v_loc_parts)[order],
        np.concatenate(v_sp_parts)[order].reshape(-1, 1),
        np.vstack(p_loc_parts)[order],
        np.concatenate(p_sp_parts)[order].reshape(-1, 1),
    )

def _build_vehicle_feat(group_df) -> np.ndarray:
    f_parts, loc_parts, sp_parts = [], [], []
    for _, row in group_df.iterrows():
        f_parts.append(_to_frames(row['frames']))
        loc_parts.append(_to_loc(row['v_loc_planar']))
        sp_parts.append(_to_scalar_seq(row['v_speed']))
    all_f   = np.concatenate(f_parts)
    all_loc = np.vstack(loc_parts)
    all_sp  = np.concatenate(sp_parts)
    order   = np.argsort(all_f, kind='stable')
    _, keep = np.unique(all_f[order], return_index=True)
    idx     = order[keep]
    v_loc   = all_loc[idx]
    v_centered = v_loc - v_loc[0:1]
    return np.concatenate([v_centered, all_sp[idx].reshape(-1, 1)], axis=1).astype(np.float32)

def _top_k_peds(group_df, ped_ids, top_k):
    if len(ped_ids) > 1 and 'd_min' in group_df.columns:
        dmin = {
            pid: group_df[group_df['p_track_id'] == pid]['d_min'].apply(_to_dmin).mean()
            for pid in ped_ids
        }
        return sorted(dmin, key=dmin.get)[:top_k]
    return list(ped_ids[:top_k])

def build_group_trajectory(group_df, top_k=DEFAULT_TOP_K):
    group_df = group_df.copy()
    group_df['_first_frame'] = group_df['frames'].apply(lambda f: int(_to_frames(f)[0]))
    group_df = group_df.sort_values('_first_frame').reset_index(drop=True)

    all_frames = np.concatenate([_to_frames(r['frames']) for _, r in group_df.iterrows()])
    start_frame = int(all_frames.min())

    ped_ids = group_df['p_track_id'].unique()
    vehicle_feat = _build_vehicle_feat(group_df)

    ped_feats = []
    for pid in _top_k_peds(group_df, ped_ids, top_k):
        ped_rows = group_df[group_df['p_track_id'] == pid]
        _, v_loc_k, _, p_loc_k, p_sp_k = _extract_row_arrays(ped_rows)
        p_rel = p_loc_k - v_loc_k
        ped_feats.append(np.concatenate([p_rel, p_sp_k], axis=1).astype(np.float32))

    return start_frame, vehicle_feat, ped_feats

def resample_trajectory(features, num_frames):
    T = features.shape[0]
    if T == num_frames:
        return features, T
    if T > num_frames:
        idx = np.linspace(0, T - 1, num_frames, dtype=int)
        return features[idx], num_frames
    padded = np.zeros((num_frames, features.shape[1]), dtype=np.float32)
    padded[:T] = features
    return padded, T

def padding_mask(valid_len, total_len) -> np.ndarray:
    mask = np.zeros(total_len, dtype=bool)
    mask[valid_len:] = True
    return mask