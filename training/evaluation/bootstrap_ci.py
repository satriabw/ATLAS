"""Video-level bootstrap CI for test APv (S2 gate, 2026-06-19).

Events within a video/vehicle interaction are NOT i.i.d., so we resample at the
VIDEO level (reviewer fix). Two modes:

  single:  --csv A            → APv mean + 95% CI for one model.
  paired:  --csv A --vs B     → CI on the DELTA APv(A) - APv(B), using the SAME
           resampled videos each iteration (paired). For the contribution gate:
           A=full vs B=no_vision / no_traj / placebo. Delta CI excluding 0 (and
           positive) = A significantly beats B.
"""
import argparse
import csv
from collections import defaultdict

import numpy as np

from ap_calculator import compute_ap


def _load(path):
    by_vid = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r['video_id'], r['v_track_id'], r['roi'])
            by_vid[r['video_id']].append({'key': key, 'gt_label': int(r['gt_label']),
                                          'score': float(r['score'])})
    return by_vid


def _apv(events):
    return compute_ap([{'gt_label': e['gt_label'], 'score': e['score']} for e in events],
                      target_class=0, score_key='score')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', required=True)
    p.add_argument('--vs', default=None, help='second CSV for paired delta CI')
    p.add_argument('--B', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    A = _load(args.csv)
    vids = sorted(A.keys())

    if args.vs is None:
        point = _apv([e for v in vids for e in A[v]])
        boot = []
        for _ in range(args.B):
            samp = rng.choice(vids, size=len(vids), replace=True)
            boot.append(_apv([e for v in samp for e in A[v]]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"APv = {point:.4f}   95% CI [{lo:.4f}, {hi:.4f}]   (B={args.B}, {len(vids)} videos)")
        return

    B = _load(args.vs)
    # align by event key per video
    Bmap = {e['key']: e for v in B for e in B[v]}
    point_a = _apv([e for v in vids for e in A[v]])
    point_b = _apv([Bmap[e['key']] for v in vids for e in A[v] if e['key'] in Bmap])
    deltas = []
    for _ in range(args.B):
        samp = rng.choice(vids, size=len(vids), replace=True)
        ea = [e for v in samp for e in A[v]]
        eb = [Bmap[e['key']] for e in ea if e['key'] in Bmap]
        deltas.append(_apv(ea) - _apv(eb))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    sig = "SIGNIFICANT (CI excludes 0)" if lo > 0 else ("negative" if hi < 0 else "n.s. (CI spans 0)")
    print(f"APv(A)={point_a:.4f}  APv(B)={point_b:.4f}  delta={point_a - point_b:+.4f}")
    print(f"paired delta 95% CI [{lo:+.4f}, {hi:+.4f}]  → {sig}   (B={args.B}, {len(vids)} videos)")


if __name__ == '__main__':
    main()
