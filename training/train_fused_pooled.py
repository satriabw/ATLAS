"""Train PooledFusedModel (pool-then-concat fusion) — reuses train.train() with an
injected model, like train_centered.py / train_centered_vision.py. FusedModel and
the rest of the production code are untouched. The checkpoint is preserved as
best_fused_pooled.pth so it doesn't clobber FusedModel's best_fused.pth.
"""
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch

from dataset import load_violation_dataset
from models import PooledFusedModel
from train import train, _scene_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description='Train pool-then-concat fused model')
    p.add_argument('--data_root',    default='/home/satria/Project/ATLAS')
    p.add_argument('--h5',           default='frames_r2.h5', help='vision h5 under data/raw/video/')
    p.add_argument('--num_frames',   type=int,   default=32)
    p.add_argument('--videos',       nargs='+', type=int, default=None)
    p.add_argument('--epochs',       type=int,   default=30)
    p.add_argument('--batch_size',   type=int,   default=8)
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

    # attributes train() / _save_ckpt expect (fused path → frames passed in _forward)
    args.vision_only = False
    args.fused       = True
    args.mode        = 'fused'
    args.backbone    = 'resnet18'
    args.overfit     = False
    args.run_name    = args.run_name or 'fused_pooled'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_dataset = load_violation_dataset(
        data_root=Path(args.data_root), label_file='train', num_frames=args.num_frames,
        top_k=args.top_k, video_filter=args.videos, use_vision=True, h5_name=args.h5)
    train_dataset, val_dataset, train_idx = _scene_split(full_dataset, args.seed)
    train_labels = [full_dataset.labels[i].annotation for i in train_idx]
    logger.info(f"Train label distribution: Violations(0)={train_labels.count(0)}, "
                f"Compliance(1)={train_labels.count(1)}")

    # same weighted CE the trajectory/fused paths use (class imbalance)
    w = torch.tensor([3.5, 1.0], dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=w)

    model = PooledFusedModel(num_classes=2, top_k=args.top_k, num_frames=args.num_frames,
                             freeze_vision=args.freeze)
    train(args, train_dataset, val_dataset, criterion, model=model)

    # preserve under a distinct name AND tag model_type so evaluate_model.py
    # auto-detects PooledFusedModel (train() saved it as model_type='fused').
    ckpt = Path(__file__).parent / 'checkpoints' / 'best_fused.pth'
    if ckpt.exists():
        dst = ckpt.parent / 'best_fused_pooled.pth'
        d = torch.load(ckpt, map_location='cpu', weights_only=False)
        d['model_type'] = 'fused_pooled'
        torch.save(d, dst)
        logger.info(f"Pooled-fused checkpoint preserved → {dst.name}")


if __name__ == '__main__':
    main()
