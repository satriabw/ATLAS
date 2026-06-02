import argparse
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def collect_expected_keys(parquet_dir: Path) -> dict[str, list[str]]:
    expected_by_video: dict[str, list[str]] = {}
    for pq_path in sorted(parquet_dir.glob("*.parquet")):
        df = pd.read_parquet(pq_path, columns=["video_id", "v_track_id", "roi"])
        for _, row in df.drop_duplicates(["v_track_id", "roi"]).iterrows():
            video_num = int(str(row["video_id"]).split("_")[-1])
            key = f"V{video_num:03d}_{row['v_track_id']}_{row['roi']}"
            video_tag = f"V{video_num:03d}"
            expected_by_video.setdefault(video_tag, []).append(key)
    return expected_by_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default="data/frames.h5")
    parser.add_argument("--parquet-dir", default="data/processed/interactions")
    args = parser.parse_args()

    h5_path = Path(args.h5)
    parquet_dir = Path(args.parquet_dir)

    if not h5_path.exists():
        print(f"FAIL: H5 file not found: {h5_path}")
        sys.exit(1)
    if not parquet_dir.exists():
        print(f"FAIL: Parquet directory not found: {parquet_dir}")
        sys.exit(1)

    print(f"Collecting expected keys from {parquet_dir} ...")
    expected_by_video = collect_expected_keys(parquet_dir)
    all_expected = {k for keys in expected_by_video.values() for k in keys}

    with h5py.File(h5_path, "r") as hf:
        all_present = set(hf.keys())

        missing = all_expected - all_present
        extra = all_present - all_expected

        print(f"Expected keys : {len(all_expected)}")
        print(f"Present keys  : {len(all_present)}")
        print(f"Missing keys  : {len(missing)}")
        print(f"Extra keys    : {len(extra)}")

        if missing:
            missing_by_video: dict[str, list[str]] = {}
            for k in missing:
                video_tag = k[:4]
                missing_by_video.setdefault(video_tag, []).append(k)
            print("\nMissing keys by video:")
            for video_tag in sorted(missing_by_video):
                print(f"  {video_tag}: {len(missing_by_video[video_tag])} missing")

        if extra:
            print(f"\nExtra keys (sample up to 10): {sorted(extra)[:10]}")

        sample_keys = random.sample(sorted(all_present & all_expected), min(20, len(all_present & all_expected)))
        shape_failures, dtype_failures, range_failures = [], [], []
        for key in sample_keys:
            ds = hf[key]
            arr = ds[:]
            if arr.shape != (32, 224, 224, 3):
                shape_failures.append((key, arr.shape))
            if arr.dtype != np.uint8:
                dtype_failures.append((key, str(arr.dtype)))
            if arr.min() < 0 or arr.max() > 255:
                range_failures.append((key, int(arr.min()), int(arr.max())))

        print(f"\nSample check ({len(sample_keys)} keys):")
        print(f"  Shape failures : {len(shape_failures)}")
        print(f"  Dtype failures : {len(dtype_failures)}")
        print(f"  Range failures : {len(range_failures)}")
        for k, s in shape_failures:
            print(f"    shape mismatch: {k} -> {s}")
        for k, d in dtype_failures:
            print(f"    dtype mismatch: {k} -> {d}")
        for k, lo, hi in range_failures:
            print(f"    range mismatch: {k} -> [{lo}, {hi}]")

    fail = bool(missing or shape_failures or dtype_failures or range_failures)
    print()
    if fail:
        print(f"FAIL: missing={len(missing)}, shape={len(shape_failures)}, dtype={len(dtype_failures)}, range={len(range_failures)}")
        sys.exit(1)
    else:
        print("PASS")


if __name__ == "__main__":
    main()
