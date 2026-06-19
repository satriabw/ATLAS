"""Vision-only training on a centered-window tight-crop h5 (2026-06-18).

Reuses train.py's train() loop, model and val-APv selection UNCHANGED — only the
data source differs (a frames_{vehicle,ped}_centered.h5 built by
scripts/build_h5_centered_crop.py, served at num_frames=64). Mirrors how
train_centered.py reuses train() for the trajectory A/B. See
docs/2026-06-18_centered_crop_vision/plan.md.
"""
import argparse
import logging
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from dataset import load_violation_dataset
from train import train, _scene_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description='Vision-only training on centered-window crops')
    p.add_argument('--data_root',    default='/home/satria/Project/ATLAS')
    p.add_argument('--h5',           required=True, help='crop h5 name under data/raw/video/')
    p.add_argument('--arm',          required=True, help='label for the preserved checkpoint, e.g. vehicle/ped')
    p.add_argument('--num_frames',   type=int,   default=64)
    p.add_argument('--videos',       nargs='+', type=int, default=None)
    p.add_argument('--epochs',       type=int,   default=30)
    p.add_argument('--batch_size',   type=int,   default=4)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--backbone_lr',  type=float, default=1e-5)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--clip_norm',    type=float, default=5.0)
    p.add_argument('--top_k',        type=int,   default=5)
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--patience',     type=int,   default=15)
    p.add_argument('--freeze',       choices=['early', 'full'], default='early')
    p.add_argument('--run_name',     default=None)
    p.add_argument('--wandb_project', default='ATLAS')
    p.add_argument('--no_wandb',     action='store_true')
    p.add_argument('--no_notify',    action='store_true')
    p.add_argument('--no_amp',       action='store_true')
    args = p.parse_args()

    # attributes train() / _save_ckpt expect (vision-only path)
    args.vision_only = True
    args.fused       = False
    args.mode        = 'vision'
    args.backbone    = 'r2plus1d'
    args.overfit     = False
    args.run_name    = args.run_name or f'vision_centered_{args.arm}'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    full_dataset = load_violation_dataset(
        data_root=Path(args.data_root), label_file='train', num_frames=args.num_frames,
        top_k=args.top_k, video_filter=args.videos, use_vision=True, h5_name=args.h5)
    train_dataset, val_dataset, train_idx = _scene_split(full_dataset, args.seed)
    train_labels = [full_dataset.labels[i].annotation for i in train_idx]
    logger.info(f"Train label distribution: Violations(0)={train_labels.count(0)}, "
                f"Compliance(1)={train_labels.count(1)}")

    # vision-only: unweighted CE (class weights push a weak branch to a constant prior)
    criterion = torch.nn.CrossEntropyLoss()

    train(args, train_dataset, val_dataset, criterion)

    # preserve the arm's checkpoint so the two arms don't clobber best_vision.pth
    ckpt = Path(__file__).parent / 'checkpoints' / 'best_vision.pth'
    if ckpt.exists():
        dst = ckpt.parent / f'best_vision_{args.arm}.pth'
        shutil.copy(ckpt, dst)
        logger.info(f"Vision checkpoint preserved → {dst.name}")


if __name__ == '__main__':
    main()
