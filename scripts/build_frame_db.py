"""Build a full-frame database h5: every frame of every video, native res, JPEG.

Schema (decode-once, never-rebuild): one resizable vlen-uint8 dataset per video,
keyed "video_NNN". Element i holds the JPEG bytes of 0-based video frame i (BGR,
as cv2 decodes it). Tracking/parquet frame numbers are 1-based, so tracking
frame k -> dataset index k-1 (matches build_h5_r2.read_crops). Any crop window,
temporal grid, resize, or grounding mask is derivable later by reading frames and
cropping in tracking (1200x1100) coordinates == native resolution here.

Resumable: a video is skipped only if its dataset exists AND attrs["complete"].
Interrupted/partial datasets are deleted and rebuilt.
"""
import argparse
import logging
import time
import tempfile
import zipfile
from pathlib import Path

import cv2
import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%F %T")
log = logging.getLogger(__name__)

ZIP_RANGES = {
    "29983897.zip": range(1, 41),
    "30050131.zip": range(41, 81),
    "30051331.zip": range(81, 121),
}


def build(output, video_start, video_end, jpeg_quality):
    output.parent.mkdir(parents=True, exist_ok=True)
    enc_param = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    with h5py.File(output, "a") as h5f:
        for vnum in range(video_start, video_end + 1):
            key = f"video_{vnum:03d}"
            if key in h5f and h5f[key].attrs.get("complete", False):
                log.info("%s already complete (%d frames) — skip", key, len(h5f[key]))
                continue
            if key in h5f:
                log.warning("%s exists but incomplete — rebuilding", key)
                del h5f[key]

            zip_name = next((z for z, r in ZIP_RANGES.items() if vnum in r), None)
            if zip_name is None:
                log.warning("no zip for %s — skip", key)
                continue
            avi_name = f"{key}.avi"
            with zipfile.ZipFile(DATA_DIR / zip_name) as zf:
                if avi_name not in zf.namelist():
                    log.warning("%s not in %s — skip", avi_name, zip_name)
                    continue
                with tempfile.NamedTemporaryFile(suffix=".avi") as tmp:
                    tmp.write(zf.read(avi_name)); tmp.flush()
                    cap = cv2.VideoCapture(tmp.name)
                    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = float(cap.get(cv2.CAP_PROP_FPS))

                    ds = h5f.create_dataset(
                        key, shape=(0,), maxshape=(None,),
                        dtype=h5py.vlen_dtype(np.uint8), chunks=(64,))
                    ds.attrs.update(width=W, height=H, fps=fps,
                                    jpeg_quality=jpeg_quality, frame_index_base=0)

                    t0 = time.time(); i = 0; nbytes = 0
                    while True:
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            break
                        buf = cv2.imencode(".jpg", frame, enc_param)[1]
                        ds.resize((i + 1,))
                        ds[i] = np.frombuffer(buf.tobytes(), dtype=np.uint8)
                        nbytes += buf.size; i += 1
                    cap.release()
                    ds.attrs["complete"] = True
                    h5f.flush()
                    log.info("%s: %d frames %dx%d  %.1f MB  %.0fs",
                             key, i, W, H, nbytes / 1e6, time.time() - t0)
    log.info("=== DONE -> %s ===", output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/media/raid5/satria_atlas/video/frames_db.h5")
    p.add_argument("--video-start", type=int, default=1)
    p.add_argument("--video-end", type=int, default=120)
    p.add_argument("--jpeg-quality", type=int, default=90)
    args = p.parse_args()
    build(Path(args.output), args.video_start, args.video_end, args.jpeg_quality)


if __name__ == "__main__":
    main()
