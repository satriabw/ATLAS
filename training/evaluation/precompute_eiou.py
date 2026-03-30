"""Precompute World-EIoU values for all test-split samples.

Mirrors the logic of Crosswalk/eiou_cal.py but using world coordinates.

For each sample in test_labels.pkl:
  - Violation events  : matched against annotation GT events from label_txt
                        via greedy best-tIoU matching per video.
                        world_eiou = tIoU × (SPIoU_start + SPIoU_end) / 2
                        (SPIoU uses parquet world positions for both pred and GT,
                        since label_txt has no world coordinates → SPIoU = 1.0
                        → world_eiou = tIoU)
  - Compliance events : no annotation GT exists → eiou = 1.0 (same default
                        as Crosswalk ap_cal.py when a sample is not in the file)

Output format (one line per sample):
    <sample_name> <eiou_value>

e.g.
    V002I00129S1D0R0A0 0.657
    V002I00001S0D1R0A1 1.000

Usage (from training/):
    python evaluation/precompute_eiou.py \\
        --labels-pkl  /home/satria/Project/ATLAS/data/raw/labels/test_labels.pkl \\
        --label-txt   /home/satria/Project/ATLAS/data/annotations/label_txt \\
        --parquet-dir /home/satria/Project/ATLAS/data/processed/interactions \\
        --output      /home/satria/Project/ATLAS/data/eiou_values.txt \\
        --split       test
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.world_eiou import calculate_tiou

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_loc(val) -> np.ndarray:
    arr = np.asarray(val)
    if arr.dtype == object:
        return np.stack(arr.tolist()).astype(np.float32)
    return arr.astype(np.float32).reshape(-1, 2)


def _to_frames(val) -> np.ndarray:
    return np.asarray(val, dtype=np.int64).ravel()


def parse_label_string(s: str):
    """Parse 'V002I00129S1D0R0A0' → (video_id, track_id, roi, annotation).

    annotation: 0 = violation, 1 = compliance  (pkl convention)
    """
    m = re.match(r'V(\d+)I(\d+)S(\d)D\d+R\d+A(\d)', s)
    if not m:
        raise ValueError(f"Cannot parse label string: {s!r}")
    video_id   = f"video_{int(m.group(1)):03d}"
    track_id   = int(m.group(2))
    roi        = 'BOT' if m.group(3) == '1' else 'TOP'
    annotation = int(m.group(4))
    return video_id, track_id, roi, annotation


def parse_label_txt(path: Path) -> list[dict]:
    """Parse one label_txt file into a list of annotation GT events.

    Format per event:
        group_id
        2
        1 <frame_in>  0 <x1> <y1> <x2> <y2>
        2 <frame_out> 0 <x1> <y1> <x2> <y2>

    All events are violations.  Pixel bbox centres are stored for reference
    but are not used in the EIoU calculation (no world-coord GT available).
    """
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    events: list[dict] = []
    i = 0
    while i < len(lines):
        try:
            group_id = int(lines[i]); i += 1
            count    = int(lines[i]); i += 1
            kf: list[dict] = []
            for _ in range(min(count, 2)):
                parts = lines[i].split(); i += 1
                frame = int(parts[1])
                x1, y1, x2, y2 = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                kf.append({'frame': frame, 'cx': (x1 + x2) / 2, 'cy': (y1 + y2) / 2})
            if len(kf) < 2:
                continue
            events.append({
                'group_id':    group_id,
                'frame_start': kf[0]['frame'],
                'frame_end':   kf[1]['frame'],
                'cx_start':    kf[0]['cx'],
                'cy_start':    kf[0]['cy'],
                'cx_end':      kf[1]['cx'],
                'cy_end':      kf[1]['cy'],
            })
        except Exception as exc:
            logger.warning(f"Parse error in {path.name} at line {i}: {exc}")
            i += 1
    return events


def build_parquet_events(parquet_dir: Path, video_ids: list[str]) -> dict[tuple, dict]:
    """Build a lookup (video_id, v_track_id, roi) → event dict from parquets."""
    index: dict[tuple, dict] = {}
    for vid in video_ids:
        path = parquet_dir / f"{vid}_interactions.parquet"
        if not path.exists():
            logger.warning(f"Parquet not found: {path}")
            continue
        df = pd.read_parquet(path)
        for (v_track_id, roi), group in df.groupby(['v_track_id', 'roi']):
            all_frames, all_vloc = [], []
            for _, row in group.iterrows():
                f = _to_frames(row['frames'])
                v = _to_loc(row['v_loc_planar'])
                n = min(len(f), len(v))
                all_frames.append(f[:n]); all_vloc.append(v[:n])
            fc = np.concatenate(all_frames)
            vc = np.vstack(all_vloc)
            order = np.argsort(fc, kind='stable')
            fc = fc[order]; vc = vc[order]
            index[(vid, int(v_track_id), str(roi))] = {
                'frame_start': int(fc[0]),
                'frame_end':   int(fc[-1]),
                'pos_start':   vc[0].tolist(),
                'pos_end':     vc[-1].tolist(),
            }
    return index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    parser = argparse.ArgumentParser(description='Precompute World-EIoU values')
    parser.add_argument('--labels-pkl',  type=Path,
                        default=Path('/home/satria/Project/ATLAS/data/raw/labels/test_labels.pkl'))
    parser.add_argument('--label-txt',   type=Path,
                        default=Path('/home/satria/Project/ATLAS/data/annotations/label_txt'))
    parser.add_argument('--parquet-dir', type=Path,
                        default=Path('/home/satria/Project/ATLAS/data/processed/interactions'))
    parser.add_argument('--output',      type=Path,
                        default=Path('/home/satria/Project/ATLAS/data/eiou_values.txt'))
    parser.add_argument('--split',       choices=['train', 'test', 'all'], default='test',
                        help='Which videos to process: test=even, train=odd, all=both')
    parser.add_argument('--min-tiou',    type=float, default=0.0,
                        help='Minimum tIoU to consider a match (default 0 = any overlap)')
    args = parser.parse_args()

    # ── load test labels ────────────────────────────────────────────────────
    with open(args.labels_pkl, 'rb') as f:
        label_strings, _ = pickle.load(f)

    # Determine video set from split
    all_nums = set()
    for s in label_strings:
        m = re.match(r'V(\d+)', s)
        if m:
            all_nums.add(int(m.group(1)))

    if args.split == 'test':
        target_nums = {n for n in all_nums if n % 2 == 0}
    elif args.split == 'train':
        target_nums = {n for n in all_nums if n % 2 != 0}
    else:
        target_nums = all_nums

    video_ids = [f"video_{n:03d}" for n in sorted(target_nums)]
    video_set = set(video_ids)
    logger.info(f"Processing {len(video_ids)} videos ({args.split} split)")

    # Filter label strings to target videos
    samples: list[tuple[str, str, int, str, int]] = []  # (raw_str, video_id, track_id, roi, annotation)
    for s in label_strings:
        try:
            vid, tid, roi, ann = parse_label_string(s)
        except ValueError as e:
            logger.warning(e); continue
        if vid in video_set:
            samples.append((s, vid, tid, roi, ann))

    logger.info(f"Samples: {len(samples)} total  "
                f"({sum(1 for *_, a in samples if a == 0)} violations, "
                f"{sum(1 for *_, a in samples if a == 1)} compliance)")

    # ── build parquet event index ────────────────────────────────────────────
    parquet_index = build_parquet_events(args.parquet_dir, video_ids)
    logger.info(f"Loaded {len(parquet_index)} parquet events")

    # ── load annotation GT events per video ─────────────────────────────────
    ann_by_video: dict[str, list[dict]] = {}
    for vid in video_ids:
        txt = args.label_txt / f"{vid}.txt"
        if txt.exists():
            ann_by_video[vid] = parse_label_txt(txt)
        else:
            logger.warning(f"label_txt not found: {txt}")
            ann_by_video[vid] = []
    total_ann = sum(len(v) for v in ann_by_video.values())
    logger.info(f"Loaded {total_ann} annotation GT events from label_txt")

    # ── compute EIoU per sample ─────────────────────────────────────────────
    results: list[tuple[str, float]] = []
    no_parquet = 0
    no_match   = 0

    for raw_str, vid, tid, roi, ann in samples:
        key = (vid, tid, roi)
        pe  = parquet_index.get(key)

        if pe is None:
            no_parquet += 1
            # Can't compute — default to 0.0 for violations, 1.0 for compliance
            eiou = 1.0 if ann == 1 else 0.0
            results.append((raw_str, eiou))
            continue

        if ann == 1:
            # Compliance: no annotation GT → EIoU defaults to 1.0
            results.append((raw_str, 1.0))
            continue

        # Violation: match against annotation GT events by tIoU
        ann_events = ann_by_video.get(vid, [])
        best_tiou  = 0.0
        for ae in ann_events:
            t = calculate_tiou(pe['frame_start'], pe['frame_end'],
                               ae['frame_start'], ae['frame_end'])
            if t > best_tiou:
                best_tiou = t

        if best_tiou < args.min_tiou:
            no_match += 1
            eiou = 0.0
        else:
            # world_eiou = tIoU × mean(SPIoU_start, SPIoU_end)
            # GT world positions are proxied from parquet → SPIoU = 1.0
            eiou = best_tiou

        results.append((raw_str, eiou))

    logger.info(f"Computed EIoU for {len(results)} samples  "
                f"({no_parquet} missing parquet, {no_match} unmatched violations)")

    # ── write output ─────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        for name, val in results:
            f.write(f"{name} {val:.4f}\n")

    logger.info(f"Saved {len(results)} EIoU values → {args.output}")

    # ── summary stats ────────────────────────────────────────────────────────
    viol_vals = [v for (s, _, _, _, a), (_, v) in zip(samples, results) if a == 0]
    comp_vals = [v for (s, _, _, _, a), (_, v) in zip(samples, results) if a == 1]

    if viol_vals:
        print(f"\nViolation EIoU  n={len(viol_vals):4d}  "
              f"mean={sum(viol_vals)/len(viol_vals):.3f}  "
              f"min={min(viol_vals):.3f}  max={max(viol_vals):.3f}")
    if comp_vals:
        print(f"Compliance EIoU n={len(comp_vals):4d}  "
              f"mean={sum(comp_vals)/len(comp_vals):.3f}  "
              f"(all 1.000 — no annotation GT)")

    bins = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.01]
    print("\nViolation EIoU distribution:")
    for lo, hi in zip(bins[:-1], bins[1:]):
        cnt = sum(1 for v in viol_vals if lo <= v < hi)
        bar = '#' * (cnt // 3)
        print(f"  [{lo:.2f}, {hi:.2f}): {cnt:4d}  {bar}")


if __name__ == '__main__':
    main()
