"""Train PooledFusedModel on the CENTERED ±32 window for BOTH branches (2026-06-18).

The earlier train_fused_pooled.py fed whole-track trajectory + centered ±32 vision
(it used the production load_violation_dataset, whose trajectory is the full
interaction span). This entry instead feeds the centered ±32 trajectory
(dataset/centered_window.py, half=32 → window 64) alongside the temporally-aligned
centered ±32 vision crops (frames_union_centered.h5, built from the SAME
build_centered_window grid). So both branches share the ±32 window — the true
apples-to-apples centered fusion vs traj-centered (0.683) / vision-centered (0.558).

Reuses train.train() with an injected model, like train_fused_pooled.py. Checkpoint
preserved as best_fused_centered.pth, tagged model_type='fused_pooled'.
"""
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch

from dataset.centered_window import load_centered_dataset
from models import PooledFusedModel
from train import train, _scene_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description='Train pooled fused model on centered ±32 window (both branches)')
    p.add_argument('--data_root',    default='/home/satria/Project/ATLAS')
    p.add_argument('--h5',           default='frames_union_centered.h5', help='centered-crop vision h5')
    p.add_argument('--window',       type=int,   default=64, help='centered window length (even); half=window//2')
    p.add_argument('--videos',       nargs='+', type=int, default=None)
    p.add_argument('--epochs',       type=int,   default=40)
    p.add_argument('--batch_size',   type=int,   default=4)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--backbone_lr',  type=float, default=1e-5)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--clip_norm',    type=float, default=5.0)
    p.add_argument('--top_k',        type=int,   default=5)
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--patience',     type=int,   default=5)
    p.add_argument('--freeze',       choices=['early', 'full'], default='early')
    p.add_argument('--run_name',     default=None)
    p.add_argument('--wandb_project', default='ATLAS')
    p.add_argument('--no_wandb',     action='store_true')
    p.add_argument('--no_notify',    action='store_true')
    p.add_argument('--no_amp',       action='store_true')
    args = p.parse_args()

    assert args.window % 2 == 0, "--window must be even"
    half = args.window // 2

    # attributes train() / _save_ckpt expect (fused path)
    args.num_frames  = args.window
    args.vision_only = False
    args.fused       = True
    args.mode        = 'fused'
    args.backbone    = 'resnet18'
    args.overfit     = False
    args.run_name    = args.run_name or 'fused_centered'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    full_dataset = load_centered_dataset(
        Path(args.data_root), label_file='train', top_k=args.top_k,
        video_filter=args.videos, half=half, h5_name=args.h5)
    train_dataset, val_dataset, train_idx = _scene_split(full_dataset, args.seed)
    train_labels = [full_dataset.labels[i].annotation for i in train_idx]
    logger.info(f"Train label distribution: Violations(0)={train_labels.count(0)}, "
                f"Compliance(1)={train_labels.count(1)}")

    # same weighted CE the trajectory/fused paths use (class imbalance)
    w = torch.tensor([3.5, 1.0], dtype=torch.float32, device=torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'))
    criterion = torch.nn.CrossEntropyLoss(weight=w)

    model = PooledFusedModel(num_classes=2, top_k=args.top_k, num_frames=args.window,
                             freeze_vision=args.freeze)
    train(args, train_dataset, val_dataset, criterion, model=model)

    ckpt = Path(__file__).parent / 'checkpoints' / 'best_fused.pth'
    if ckpt.exists():
        dst = ckpt.parent / 'best_fused_centered.pth'
        d = torch.load(ckpt, map_location='cpu', weights_only=False)
        d['model_type'] = 'fused_pooled'
        torch.save(d, dst)
        logger.info(f"Centered-fused checkpoint preserved → {dst.name}")


if __name__ == '__main__':
    main()
