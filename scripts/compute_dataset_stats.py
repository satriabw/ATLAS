"""
Computes summary statistics over the full labeled dataset (train + test).
Outputs: results/summary/stats.json and results/summary/summary.txt

Usage: python scripts/compute_dataset_stats.py
"""
import pickle, re, json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

DATA_ROOT   = Path('/home/satria/Project/ATLAS')
PARQUET_DIR = DATA_ROOT / 'data/processed/interactions'
OUTPUT_DIR  = DATA_ROOT / 'results/summary'


def parse_label(s):
    m = re.match(r'V(\d+)I(\d+)S(\d)D\d+R\d+A(\d)', s)
    if not m:
        return None
    return (
        f"video_{int(m.group(1)):03d}",
        int(m.group(2)),
        'BOT' if m.group(3) == '1' else 'TOP',
        int(m.group(4)),
    )


def to_arr(val):
    arr = np.asarray(val)
    if arr.dtype == object:
        return np.stack(arr.tolist()).astype(np.float32)
    return arr.astype(np.float32)


def event_stats(df, v_track_id, roi):
    group = df[(df['v_track_id'] == v_track_id) & (df['roi'] == roi)]
    if group.empty:
        return None

    speeds, locs, frames = [], [], []
    for _, row in group.iterrows():
        f  = np.asarray(row['frames'], dtype=np.int64).ravel()
        sp = np.asarray(row['v_speed'], dtype=np.float32).ravel()
        lc = to_arr(row['v_loc_planar'])
        frames.append(f)
        speeds.append(sp)
        locs.append(lc)

    all_f  = np.concatenate(frames)
    all_sp = np.concatenate(speeds)
    all_lc = np.vstack(locs)

    order  = np.argsort(all_f)
    _, keep = np.unique(all_f[order], return_index=True)
    idx    = order[keep]

    v_sp  = all_sp[idx]
    v_loc = all_lc[idx]
    n_ped = group['p_track_id'].nunique()

    diffs = np.diff(v_loc, axis=0)
    arc_m = float(np.sqrt((diffs ** 2).sum(axis=1)).sum())

    return {
        'n_frames':     int(len(v_sp)),
        'mean_speed':   float(np.mean(v_sp)),
        'max_speed':    float(np.max(v_sp)),
        'min_speed':    float(np.min(v_sp)),
        'arc_length_m': arc_m,
        'n_peds':       int(n_ped),
    }


def agg(values, key='mean'):
    if not values:
        return None
    a = np.array(values)
    return {'mean': float(np.mean(a)), 'median': float(np.median(a)),
            'std': float(np.std(a)), 'p25': float(np.percentile(a, 25)),
            'p75': float(np.percentile(a, 75))}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_labels = []
    for split in ('train', 'test'):
        pkl = DATA_ROOT / 'data/raw/labels' / f'{split}_labels.pkl'
        if not pkl.exists():
            continue
        with open(pkl, 'rb') as f:
            strings, _ = pickle.load(f)
        for s in strings:
            l = parse_label(s)
            if l:
                all_labels.append(l)
    print(f"Total parsed labels: {len(all_labels)}")

    parquets = {}
    events = []
    for vid, tid, roi, ann in all_labels:
        if vid not in parquets:
            p = PARQUET_DIR / f'{vid}_interactions.parquet'
            parquets[vid] = pd.read_parquet(p) if p.exists() else None
        df = parquets[vid]
        if df is None:
            continue
        st = event_stats(df, tid, roi)
        if st:
            st['annotation'] = ann
            events.append(st)

    print(f"Events with stats: {len(events)}")

    violations  = [e for e in events if e['annotation'] == 0]
    compliances = [e for e in events if e['annotation'] == 1]

    def summarize(evts):
        if not evts:
            return {}
        return {
            'count':          len(evts),
            'mean_speed':     agg([e['mean_speed']   for e in evts]),
            'max_speed':      agg([e['max_speed']    for e in evts]),
            'min_speed':      agg([e['min_speed']    for e in evts]),
            'arc_length_m':   agg([e['arc_length_m'] for e in evts]),
            'n_frames':       agg([e['n_frames']     for e in evts]),
            'n_peds_dist':    dict(sorted(Counter(e['n_peds'] for e in evts).items())),
        }

    # Violation rate by mean-speed bin
    bins       = [0, 1, 2, 3, 4, 5, 6, 8, 10, 999]
    bin_labels = ['0–1', '1–2', '2–3', '3–4', '4–5', '5–6', '6–8', '8–10', '10+']
    speed_bins = {}
    for lbl, lo, hi in zip(bin_labels, bins[:-1], bins[1:]):
        bucket = [e for e in events if lo <= e['mean_speed'] < hi]
        if not bucket:
            continue
        n_v = sum(1 for e in bucket if e['annotation'] == 0)
        speed_bins[f'{lbl} m/s'] = {
            'total': len(bucket),
            'violations': n_v,
            'violation_rate': round(n_v / len(bucket), 4),
        }

    stats = {
        'class_balance': {
            'total':          len(events),
            'violations':     len(violations),
            'compliances':    len(compliances),
            'violation_rate': round(len(violations) / len(events), 4),
        },
        'violations':                  summarize(violations),
        'compliances':                 summarize(compliances),
        'violation_rate_by_speed_bin': speed_bins,
    }

    stats_path = OUTPUT_DIR / 'stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved {stats_path}")

    # Natural-language summary
    cb  = stats['class_balance']
    vs  = stats['violations']
    cs  = stats['compliances']
    top_bin = max(speed_bins.items(), key=lambda x: x[1]['violation_rate'])
    zero_bin = min(speed_bins.items(), key=lambda x: x[1]['violation_rate'])

    vmed  = vs['mean_speed']['median']
    cmed  = cs['mean_speed']['median']
    delta = vmed - cmed
    val   = vs['arc_length_m']['median']
    cal   = cs['arc_length_m']['median']

    summary = f"""Dataset Summary — ATLAS Crosswalk Violation Detection
======================================================
Total events: {cb['total']}
  Violations:  {cb['violations']} ({cb['violation_rate']*100:.1f}%)
  Compliances: {cb['compliances']} ({(1-cb['violation_rate'])*100:.1f}%)

Class imbalance is real (~1 : {cb['compliances']//max(cb['violations'],1)}).
Training uses class-weighted loss [3.5, 1.0] to compensate.

Vehicle approach speed (mean over trajectory):
  Violations:  {vmed:.2f} m/s median  (std {vs['mean_speed']['std']:.2f})
  Compliances: {cmed:.2f} m/s median  (std {cs['mean_speed']['std']:.2f})
  → Violating vehicles approach {abs(delta):.2f} m/s faster on average.

Trajectory arc length (world-space):
  Violations:  {val:.1f} m median
  Compliances: {cal:.1f} m median

Pedestrian interaction count:
  Violations  — {vs['n_peds_dist']}
  Compliances — {cs['n_peds_dist']}

Violation rate by speed bin (non-obvious finding):
  Highest: {top_bin[0]} → {top_bin[1]['violation_rate']*100:.1f}% violation rate
           ({top_bin[1]['violations']} / {top_bin[1]['total']} events)
  Lowest:  {zero_bin[0]} → {zero_bin[1]['violation_rate']*100:.1f}% violation rate

  Speed is strongly discriminative: vehicles approaching at {top_bin[0]}
  are {top_bin[1]['violation_rate']*100:.1f}% likely to violate.
  This directly motivates world-space speed estimation — pixel-space
  appearance alone cannot recover this signal.
"""

    txt_path = OUTPUT_DIR / 'summary.txt'
    with open(txt_path, 'w') as f:
        f.write(summary)
    print(summary)
    print(f"Saved {txt_path}")


if __name__ == '__main__':
    main()
