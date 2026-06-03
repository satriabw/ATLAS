import argparse
import logging
import os
import tempfile
import zipfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

ZIP_RANGES = {
    "29983897.zip": range(1, 41),
    "30050131.zip": range(41, 81),
    "30051331.zip": range(81, 121),
}


def _detect_lz4():
    probe = f"/tmp/_lz4probe_{os.getpid()}.h5"
    try:
        with h5py.File(probe, "w") as f:
            f.create_dataset("x", data=np.zeros(4, dtype=np.uint8), compression="lz4")
        return "lz4"
    except Exception:
        return None


def _read_frames(cap, frame_indices, size):
    out = np.zeros((len(frame_indices), size, size, 3), dtype=np.uint8)
    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = cv2.resize(frame, (size, size))
        out[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return out


def build_h5(output: Path, num_frames: int, size: int, video_start: int = 1, video_end: int = 120):
    compression = _detect_lz4()
    parquet_dir = DATA_DIR / "processed" / "interactions"

    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output, "a") as h5f:
        for video_num in tqdm(range(video_start, video_end + 1), desc="Videos"):
            video_name = f"video_{video_num:03d}"
            parquet_path = parquet_dir / f"{video_name}_interactions.parquet"
            if not parquet_path.exists():
                log.warning("Parquet not found: %s", parquet_path)
                continue

            zip_name = next(
                (z for z, r in ZIP_RANGES.items() if video_num in r), None
            )
            if zip_name is None:
                log.warning("No zip for %s — skipping", video_name)
                continue

            zip_path = DATA_DIR / zip_name
            avi_name = f"{video_name}.avi"

            df = pd.read_parquet(parquet_path)
            groups = [
                (tid, roi, g)
                for (tid, roi), g in df.groupby(["v_track_id", "roi"])
            ]

            keys = [f"V{video_num:03d}_{tid}_{roi}" for (tid, roi, _) in groups]
            pending = [
                (key, tid, roi, g)
                for key, (tid, roi, g) in zip(keys, groups)
                if key not in h5f
            ]
            if not pending:
                continue

            with zipfile.ZipFile(zip_path) as zf:
                if avi_name not in zf.namelist():
                    log.warning("%s not in %s — skipping", avi_name, zip_name)
                    continue
                with tempfile.NamedTemporaryFile(suffix=".avi", delete=True) as tmp:
                    tmp.write(zf.read(avi_name))
                    tmp.flush()
                    cap = cv2.VideoCapture(tmp.name)
                    try:
                        for key, tid, roi, g in pending:
                            all_frames = np.concatenate(g["frames"].values)
                            start_frame = int(all_frames.min())
                            end_frame = int(all_frames.max())
                            indices = np.linspace(
                                start_frame, end_frame, num_frames, dtype=int
                            )
                            tensor = _read_frames(cap, indices, size)
                            h5f.create_dataset(
                                key,
                                data=tensor,
                                compression=compression,
                            )
                    finally:
                        cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/frames.h5")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--video-start", type=int, default=1)
    parser.add_argument("--video-end", type=int, default=120)
    args = parser.parse_args()

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    build_h5(output, args.num_frames, args.size, args.video_start, args.video_end)
    log.info("Done → %s", output)


if __name__ == "__main__":
    main()
