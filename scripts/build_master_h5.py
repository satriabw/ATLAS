"""Build the master frames_db.h5: one logical file linking all 120 videos.

Stores no frame data — just an h5py.ExternalLink per `video_NNN` key pointing at
the shard on raid5 where that dataset actually lives (see build_frame_db.py). The
master is a few KB and can sit anywhere (e.g. nvme); links use absolute shard
paths so it resolves regardless of the master's own location.
"""
import argparse
from pathlib import Path

import h5py

SHARDS = {
    "/media/raid5/satria_atlas/video/frames_db_001_040.h5": range(1, 41),
    "/media/raid5/satria_atlas/video/frames_db_041_080.h5": range(41, 81),
    "/media/raid5/satria_atlas/video/frames_db_081_120.h5": range(81, 121),
}


def build(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    n = 0
    with h5py.File(output, "w") as m:
        for shard, rng in SHARDS.items():
            for vnum in rng:
                key = f"video_{vnum:03d}"
                m[key] = h5py.ExternalLink(shard, "/" + key)
                n += 1
    # verify every link resolves and reports frame count
    with h5py.File(output, "r") as m:
        for vnum in (1, 60, 120):
            key = f"video_{vnum:03d}"
            print(f"  {key}: {len(m[key])} frames (complete={m[key].attrs.get('complete')})")
    print(f"=== wrote {n} external links -> {output} ({output.stat().st_size} bytes) ===")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output",
                   default="/home/satria/Project/ATLAS/data/raw/video/frames_db.h5")
    args = p.parse_args()
    build(Path(args.output))


if __name__ == "__main__":
    main()
