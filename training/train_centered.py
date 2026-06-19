"""A/B: train the trajectory model on centered-window data (2026-06-17).

Reuses train.py's training loop, model, optimizer and val-APv selection UNCHANGED
— only the data pipeline differs (dataset/centered_window.py vs the production
whole-track linspace resample). So the comparison isolates the data-sampling
variable. Baseline control = `python train.py --mode trajectory` with matched
args. See docs/centered_window_experiment.md.
"""
import argparse
import logging
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from dataset.centered_window import load_centered_dataset
from train import train, _scene_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description='Train trajectory model on centered-window data')
    p.add_argument('--data_root',   default='/home/satria/Project/ATLAS')
    p.add_argument('--videos',      nargs='+', type=int, default=None)
    p.add_argument('--epochs',      type=int,   default=50)
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--clip_norm',   type=float, default=5.0)
    p.add_argument('--top_k',       type=int,   default=5)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--patience',    type=int,   default=15)
    p.add_argument('--window',      type=int,   default=32, help='centered window length (must be even); half=window//2')
    p.add_argument('--run_name',    default='traj_centered')
    p.add_argument('--wandb_project', default='ATLAS')
    p.add_argument('--no_wandb',    action='store_true')
    p.add_argument('--no_notify',   action='store_true')
    p.add_argument('--no_amp',      action='store_true')
    args = p.parse_args()

    # attributes train() / _save_ckpt expect (trajectory-only path)
    args.vision_only = False
    args.fused       = False
    args.mode        = 'trajectory'
    args.backbone    = 'resnet18'
    args.backbone_lr = 1e-5
    args.h5          = 'frames.h5'
    args.overfit     = False
    assert args.window % 2 == 0, '--window must be even'
    half = args.window // 2
    args.num_frames = args.window  # CrossAttentionModel reshapes ped_feat by num_frames

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_dataset = load_centered_dataset(
        Path(args.data_root), label_file='train', top_k=args.top_k, video_filter=args.videos, half=half)
    train_dataset, val_dataset, train_idx = _scene_split(full_dataset, args.seed)
    train_labels = [full_dataset.labels[i].annotation for i in train_idx]
    logger.info(f"Train label distribution: Violations(0)={train_labels.count(0)}, "
                f"Compliance(1)={train_labels.count(1)}")

    w = torch.tensor([3.5, 1.0], dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=w)

    train(args, train_dataset, val_dataset, criterion)

    # preserve the centered checkpoint so it doesn't clash with the baseline's
    ckpt = Path(__file__).parent / 'checkpoints' / 'best_model.pth'
    if ckpt.exists():
        dst = ckpt.parent / f'best_traj_centered_w{args.window}.pth'
        shutil.copy(ckpt, dst)
        logger.info(f"Centered checkpoint preserved → {dst.name}")


if __name__ == '__main__':
    main()
