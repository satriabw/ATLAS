import argparse
import sys
from pathlib import Path

import h5py
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Partial H5 files to merge")
    parser.add_argument("--output", default="data/frames.h5")
    args = parser.parse_args()

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    inputs = [Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p for p in args.inputs]
    missing = [p for p in inputs if not p.exists()]
    if missing:
        print(f"ERROR: files not found: {missing}")
        sys.exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output, "a") as out_f:
        for src_path in inputs:
            with h5py.File(src_path, "r") as src_f:
                keys = list(src_f.keys())
                print(f"{src_path.name}: {len(keys)} keys")
                for key in tqdm(keys, desc=src_path.name):
                    if key not in out_f:
                        src_f.copy(key, out_f)

    total = 0
    with h5py.File(output, "r") as f:
        total = len(f.keys())
    print(f"Done → {output}  ({total} total keys)")


if __name__ == "__main__":
    main()
