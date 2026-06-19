"""Step 2 — grounded centered vision (2026-06-18).

Trains VisionOnlyModel(resnet18, in_channels=5) on the centered ±32 union crops with
subject-vehicle / pedestrian box-mask channels rasterised ON THE CENTERED GRID
(dataset/centered_window.py grounding). The masks tell vision WHICH vehicle-ped pair
is the event — the disambiguation ungrounded RGB lacks.

  --arm grounded   : real centered masks.
  --arm ungrounded : same arch, mask channels zeroed (--zero_masks) — A/B control.

Reuses train.train() with an injected model (num_frames=64). Saves best_vision_{arm}.pth.
"""
import argparse
import logging
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from dataset.centered_window import load_centered_dataset
from models import VisionOnlyModel
from train import train, _scene_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description='Grounded centered vision (Step 2)')
    p.add_argument('--data_root',    default='/home/satria/Project/ATLAS')
    p.add_argument('--h5',           default='frames_union_centered.h5')
    p.add_argument('--arm',          required=True, help='grounded | ungrounded (label for ckpt)')
    p.add_argument('--zero_masks',   action='store_true', help='zero mask channels (ungrounded control)')
    p.add_argument('--window',       type=int,   default=64)
    p.add_argument('--videos',       nargs='+', type=int, default=None)
    p.add_argument('--epochs',       type=int,   default=30)
    p.add_argument('--batch_size',   type=int,   default=4)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--backbone_lr',  type=float, default=1e-5)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--clip_norm',    type=float, default=5.0)
    p.add_argument('--top_k',        type=int,   default=5)
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--patience',     type=int,   default=8)
    p.add_argument('--freeze',       choices=['early', 'full'], default='early')
    p.add_argument('--run_name',     default=None)
    p.add_argument('--wandb_project', default='ATLAS')
    p.add_argument('--no_wandb',     action='store_true')
    p.add_argument('--no_notify',    action='store_true')
    p.add_argument('--no_amp',       action='store_true')
    args = p.parse_args()
    half = args.window // 2

    args.num_frames  = args.window
    args.vision_only = True
    args.fused       = False
    args.mode        = 'vision'
    args.backbone    = 'resnet18'
    args.overfit     = False
    args.run_name    = args.run_name or f'vision_centered_grounded_{args.arm}'

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    full = load_centered_dataset(
        Path(args.data_root), label_file='train', top_k=args.top_k, half=half,
        video_filter=args.videos, h5_name=args.h5, ground=True, zero_masks=args.zero_masks)
    tr, va, tr_idx = _scene_split(full, args.seed)
    tl = [full.labels[i].annotation for i in tr_idx]
    logger.info(f"Train label distribution: Violations(0)={tl.count(0)}, Compliance(1)={tl.count(1)}")

    criterion = torch.nn.CrossEntropyLoss()  # vision-only: unweighted (weak branch → prior collapse)
    model = VisionOnlyModel(num_classes=2, num_frames=args.window, backbone='resnet18',
                            freeze_vision=(args.freeze == 'early'))
    train(args, tr, va, criterion, model=model)

    ckpt = Path(__file__).parent / 'checkpoints' / 'best_vision.pth'
    if ckpt.exists():
        dst = ckpt.parent / f'best_vision_{args.arm}.pth'
        shutil.copy(ckpt, dst)
        logger.info(f"Vision checkpoint preserved → {dst.name}")


if __name__ == '__main__':
    main()
