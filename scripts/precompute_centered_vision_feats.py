"""S1 of the aligned-joint-fusion experiment (2026-06-19, plan =
artifacts/docs/2026-06-19_joint_fusion/plan.md).

Precompute FROZEN per-slot appearance features for the centered ±32 (64-slot)
union-crop vision h5, so the joint-fusion model trains over fixed vision features
(no vision CNN in the graph → the §11 poisoning mechanism is structurally absent).

For every key in frames_union_centered.h5:
  - decode the 64 aligned slots → (64, 3, 112, 112) ImageNet-normalized RGB,
  - frozen ResNet18 (ImageNet) penultimate pool → (64, 512),
  - ZERO the feature rows at padded slots (frames==-1 from the centered window),
    because an ImageNet-normalized BLACK pad frame does NOT give a zero feature —
    a frozen net emits a nonzero content-free artifact the downstream attention
    would otherwise fixate on (reviewer-2 correctness fix #3).

Output: data/raw/video/centered_vision_feats.h5, one (64,512) float32 dataset per
key, with attrs `n_valid` (== count of non-padded slots) for validation, plus a
(64,) uint8 `pad_mask` dataset sibling. Keyed identically to the crop h5.
"""
import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'training'))
from dataset.centered_window import load_centered_dataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'


def _extractor(device):
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    feat = nn.Sequential(*list(m.children())[:-1])  # → (B, 512, 1, 1) incl. avgpool
    feat.eval().to(device)
    for p in feat.parameters():
        p.requires_grad = False
    return feat


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_root', default=str(Path(__file__).resolve().parent.parent))
    p.add_argument('--h5_name', default='frames_union_centered.h5')
    p.add_argument('--half', type=int, default=32)
    p.add_argument('--out', default=None,
                   help='default: data/raw/video/centered_vision_feats.h5')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feat = _extractor(device)
    out = Path(args.out) if args.out else DATA_DIR / 'raw' / 'video' / 'centered_vision_feats.h5'

    written = set()
    with h5py.File(out, 'w') as fout:
        # train (odd vids) + test (even vids) together cover every labeled key
        for split in ('train', 'test'):
            ds = load_centered_dataset(Path(args.data_root), label_file=split,
                                       half=args.half, h5_name=args.h5_name)
            log.info("[%s] %d labeled samples", split, len(ds))
            for i in range(len(ds)):
                s = ds[i]
                key = f"V{s['video_id'][-3:]}_{s['tracking_id']}_{s['roi']}"
                if key in written:
                    continue
                frames = s['frames'][:, :3].to(device)          # (64,3,112,112) RGB
                pad = s['v_padding_mask'].numpy().astype(bool)    # True = padded
                with torch.no_grad():
                    v = feat(frames).flatten(1).cpu().numpy().astype(np.float32)  # (64,512)
                v[pad] = 0.0                                       # feature-level pad zeroing
                n_valid = int((~pad).sum())
                g = fout.create_dataset(key, data=v, compression='gzip')
                g.attrs['n_valid'] = n_valid
                fout.create_dataset(f"{key}__pad", data=pad.astype(np.uint8))
                written.add(key)
                if len(written) % 500 == 0:
                    log.info("  %d keys written", len(written))
    log.info("DONE — %d keys → %s", len(written), out)


if __name__ == '__main__':
    main()
