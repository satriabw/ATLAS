"""Train the aligned joint-fusion model (S2, 2026-06-19, plan =
artifacts/docs/2026-06-19_joint_fusion/plan.md).

Dedicated compact loop (the model returns aux outputs, so it doesn't fit
train.train's signature). Trajectory + frozen per-slot appearance, main CE +
vision-aux CE (anti-dominance). val-APv checkpoint selection. Saves
checkpoints/best_aligned_fusion_{tag}.pth with config for eval reconstruction.

--shuffle-vision trains the placebo arm; --ablate {no_vision,no_traj} for the
contribution gate (also available at eval time).
"""
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.aligned_fusion_data import AlignedFusionDataset
from train import _scene_split
from models.aligned_fusion import AlignedFusionModel
from evaluation.ap_calculator import compute_ap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _move(batch, device):
    return (batch['vehicle_feat'].to(device), batch['ped_feat'].to(device),
            batch['vis_feat'].to(device), batch['v_padding_mask'].to(device),
            batch['p_padding_mask'].to(device), batch['label'].to(device))


def _validate(model, loader, device, main_crit):
    model.eval()
    preds, tot = [], 0.0
    with torch.no_grad():
        for b in loader:
            vf, pf, vis, vm, pm, y = _move(b, device)
            main, _, _ = model(vf, pf, vis, vm, pm)
            tot += main_crit(main, y).item()
            pv = torch.softmax(main.float(), 1)[:, 0]
            preds.extend({'gt_label': int(t), 'score': float(s)} for t, s in zip(y.cpu(), pv.cpu()))
    return tot / max(len(loader), 1), compute_ap(preds, target_class=0, score_key='score')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_root', default='/home/satria/Project/ATLAS')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--clip_norm', type=float, default=5.0)
    p.add_argument('--top_k', type=int, default=5)
    p.add_argument('--half', type=int, default=32)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--aux_w', type=float, default=0.5, help='vision aux-head loss weight')
    p.add_argument('--pool', choices=['attn', 'max'], default='attn')
    p.add_argument('--shuffle-vision', action='store_true', help='placebo arm')
    p.add_argument('--ablate', choices=['no_vision', 'no_traj'], default=None)
    p.add_argument('--tag', default=None)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    full = AlignedFusionDataset(args.data_root, 'train', top_k=args.top_k, half=args.half,
                                shuffle_vision=args.shuffle_vision, seed=args.seed)
    train_ds, val_ds, train_idx = _scene_split(full, 42)  # split seed fixed = same videos
    tl = [full.labels[i].annotation for i in train_idx]
    logger.info("Train: %d (V=%d C=%d)  Val: %d", len(train_ds), tl.count(0), tl.count(1), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    model = AlignedFusionModel(top_k=args.top_k, num_frames=2 * args.half, pool=args.pool).to(device)
    model.ablate = args.ablate
    w = torch.tensor([3.5, 1.0], device=device)
    main_crit = nn.CrossEntropyLoss(weight=w)
    aux_crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    tag = args.tag or (f"s{args.seed}" + ("_shuffle" if args.shuffle_vision else "")
                       + (f"_{args.ablate}" if args.ablate else ""))
    ckpt = Path(__file__).parent / 'checkpoints' / f'best_aligned_fusion_{tag}.pth'
    ckpt.parent.mkdir(exist_ok=True)

    best_apv, best_ep, since = float('-inf'), 0, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        tr = 0.0
        for b in train_loader:
            vf, pf, vis, vm, pm, y = _move(b, device)
            main, aux_vis, _ = model(vf, pf, vis, vm, pm)
            loss = main_crit(main, y) + args.aux_w * aux_crit(aux_vis, y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            opt.step(); tr += loss.item()
        vl, vapv = _validate(model, val_loader, device, main_crit)
        logger.info("Ep %d/%d  train %.4f  val %.4f  val APv %.4f", ep, args.epochs,
                    tr / max(len(train_loader), 1), vl, vapv)
        if vapv > best_apv:
            best_apv, best_ep, since = vapv, ep, 0
            torch.save({'model_state_dict': model.state_dict(), 'val_apv': vapv, 'epoch': ep,
                        'config': {'top_k': args.top_k, 'half': args.half, 'num_frames': 2 * args.half, 'pool': args.pool},
                        'tag': tag, 'seed': args.seed, 'shuffle_vision': args.shuffle_vision},
                       ckpt)
        else:
            since += 1
            if since >= args.patience:
                logger.info("Early stop at ep %d", ep); break
    logger.info("DONE tag=%s  best val APv %.4f (ep %d)  → %s", tag, best_apv, best_ep, ckpt.name)


if __name__ == '__main__':
    main()
