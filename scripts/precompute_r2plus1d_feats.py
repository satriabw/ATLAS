"""Precompute FROZEN pooled r2plus1d features (2026-07-09, plan =
artifacts/docs/2026-07-09_gated_r2/plan.md).

Loads the fine-tuned r2plus1d checkpoint (best_r2plus1d_r2_rebuild.pth) and
extracts one pooled pre-proj backbone vector per event key:
  clip → Kinetics renorm + 112px resize (same as VisionEncoder3D.forward)
  → vision_encoder.features → mean over (T', 7, 7) → (512,) float32.

--bed whole    : ViolationDataset over frames_r2.h5 (32-slot linspace crops —
                 the exact preprocessing the backbone was fine-tuned on).
--bed centered : load_centered_dataset over frames_union_centered.h5 (64-slot
                 clips incl. black pad slots, as consumed by the centered arm).

Output: one (512,) dataset per key, keyed V{vid}_{tid}_{roi} like the crop h5s.
"""
import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'training'))
from dataset.loader import load_violation_dataset            # noqa: E402
from dataset.centered_window import load_centered_dataset    # noqa: E402
from models.classifier import VisionOnlyModel                # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def _encoder(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = VisionOnlyModel(num_classes=2, num_frames=32, backbone='r2plus1d')
    model.load_state_dict(ck['model_state_dict'])
    enc = model.vision_encoder.eval().to(device)
    for p in enc.parameters():
        p.requires_grad = False
    return enc


@torch.no_grad()
def _extract(enc, frames):
    # frames (B,F,C,H,W) ImageNet-normalized; mirrors VisionEncoder3D.forward
    # up to the backbone, then pools everything (no proj).
    x = frames[:, :, :3].permute(0, 2, 1, 3, 4)
    x = x * enc._renorm_a + enc._renorm_b
    B, C, T, H, W = x.shape
    if H != 112 or W != 112:
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
        x = x.reshape(B, T, C, 112, 112).permute(0, 2, 1, 3, 4)
    feats = enc.features(x)                     # (B, 512, T', 7, 7)
    return feats.mean(dim=(2, 3, 4))            # (B, 512)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_root', default=str(Path(__file__).resolve().parent.parent))
    p.add_argument('--ckpt', default='training/checkpoints/best_r2plus1d_r2_rebuild.pth')
    p.add_argument('--bed', choices=['whole', 'centered'], required=True)
    p.add_argument('--out', required=True, help='output h5 path')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--limit', type=int, default=None, help='smoke test: stop after N keys per split')
    args = p.parse_args()

    root = Path(args.data_root)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = _encoder(root / args.ckpt if not Path(args.ckpt).is_absolute() else args.ckpt, device)

    written = set()
    with h5py.File(args.out, 'w') as fout:
        for split in ('train', 'test'):
            if args.bed == 'whole':
                ds = load_violation_dataset(root, split, num_frames=32,
                                            use_vision=True, h5_name='frames_r2.h5')
            else:
                ds = load_centered_dataset(root, label_file=split, half=32,
                                           h5_name='frames_union_centered.h5')
            log.info("[%s/%s] %d labeled samples", args.bed, split, len(ds))
            n_split = 0
            for start in range(0, len(ds), args.batch_size):
                idx = range(start, min(start + args.batch_size, len(ds)))
                todo = []
                for i in idx:
                    l = ds.labels[i]
                    k = f"V{l.video_id[-3:]}_{l.tracking_id}_{l.roi}"
                    if k not in written:
                        todo.append((i, k))
                if not todo:
                    continue
                frames = torch.stack([ds[i]['frames'] for i, _ in todo]).to(device)
                v = _extract(enc, frames).cpu().numpy().astype(np.float32)
                for (_, k), vec in zip(todo, v):
                    fout.create_dataset(k, data=vec)
                    written.add(k)
                n_split += len(todo)
                if args.limit and n_split >= args.limit:
                    break
                if len(written) % 500 < args.batch_size:
                    log.info("  %d keys written", len(written))
    log.info("DONE — %d keys → %s", len(written), args.out)


if __name__ == '__main__':
    main()
