"""Train one rung of the fusion bisection ladder (2026-07-08, plan =
artifacts/docs/2026-07-08_fusion_failure_investigation/plan.md).

Same protocol as train_gated_fusion.py (dataset, split, class weights,
val-APv checkpoint selection); model is always the plain-concat
GatedFusionModel(gate=False). Rungs differ only in initialization and
what is trainable:
  --init-traj CKPT   load vehicle_encoder / ped_encoder / cross_attn from a
                     prior checkpoint (traj-only or gated — same module names)
  --freeze-traj      freeze those three modules (head-only training)
  --traj-lr LR       unfrozen traj core trains at its own (lower) LR
  --no-vision        train with the vision vector zeroed (ablate='no_vision')
  --shuffle-vision   placebo vision stream

Saves checkpoints/best_ladder_{tag}.pth, readable by
evaluation/evaluate_gated_fusion.py unchanged.
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
from dataset.wholetrack_fusion_data import WholeTrackFusionDataset
from train import _scene_split
from models.gated_fusion import GatedFusionModel
from evaluation.ap_calculator import compute_ap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TRAJ_MODULES = ('vehicle_encoder', 'ped_encoder', 'cross_attn')


def _move(batch, device):
    return (batch['vehicle_feat'].to(device), batch['ped_feat'].to(device),
            batch['vis_feat'].to(device), batch['v_padding_mask'].to(device),
            batch['p_padding_mask'].to(device), batch['label'].to(device))


def _validate(model, loader, device, crit):
    model.eval()
    preds, tot = [], 0.0
    with torch.no_grad():
        for b in loader:
            vf, pf, vis, vm, pm, y = _move(b, device)
            logits, _ = model(vf, pf, vis, vm, pm)
            tot += crit(logits, y).item()
            pv = torch.softmax(logits.float(), 1)[:, 0]
            preds.extend({'gt_label': int(t), 'score': float(s)} for t, s in zip(y.cpu(), pv.cpu()))
    return tot / max(len(loader), 1), compute_ap(preds, target_class=0, score_key='score')


def _load_traj_core(model, ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)['model_state_dict']
    core = {k: v for k, v in sd.items() if k.split('.')[0] in TRAJ_MODULES}
    missing, unexpected = model.load_state_dict(core, strict=False)
    assert not unexpected, unexpected
    still = [k for k in missing if k.split('.')[0] in TRAJ_MODULES]
    assert not still, f"traj-core keys not covered by {ckpt_path}: {still}"
    logger.info("Loaded %d traj-core tensors from %s", len(core), ckpt_path)


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
    p.add_argument('--patience', type=int, default=7)
    p.add_argument('--init-traj', default=None, help='checkpoint to load the traj core from')
    p.add_argument('--freeze-traj', action='store_true')
    p.add_argument('--traj-lr', type=float, default=None, help='separate LR for the traj core')
    p.add_argument('--no-vision', action='store_true', help='train with vision zeroed (R1b/R1g)')
    p.add_argument('--gate', action='store_true', help='use the sigmoid gate block (default: plain concat)')
    p.add_argument('--shuffle-vision', action='store_true', help='placebo arm')
    p.add_argument('--bed', choices=['centered', 'whole'], default='centered',
                   help='trajectory bed: centered ±half slots or whole-track 32-slot')
    p.add_argument('--feats', default=None,
                   help='vision feats h5 name (default per bed: centered_vision_feats.h5 / r2_whole_feats.h5)')
    p.add_argument('--tag', required=True)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.bed == 'whole':
        feats = args.feats or 'r2_whole_feats.h5'
        num_frames = 32
        full = WholeTrackFusionDataset(args.data_root, 'train', feats_name=feats,
                                       top_k=args.top_k, num_frames=num_frames,
                                       shuffle_vision=args.shuffle_vision, seed=args.seed)
    else:
        feats = args.feats or 'centered_vision_feats.h5'
        num_frames = 2 * args.half
        full = AlignedFusionDataset(args.data_root, 'train', feats_name=feats,
                                    top_k=args.top_k, half=args.half,
                                    shuffle_vision=args.shuffle_vision, seed=args.seed)
    train_ds, val_ds, train_idx = _scene_split(full, 42)  # split seed fixed = same videos
    tl = [full.labels[i].annotation for i in train_idx]
    logger.info("Train: %d (V=%d C=%d)  Val: %d", len(train_ds), tl.count(0), tl.count(1), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    model = GatedFusionModel(top_k=args.top_k, num_frames=num_frames, gate=args.gate).to(device)
    if args.no_vision:
        model.ablate = 'no_vision'
    if args.init_traj:
        _load_traj_core(model, args.init_traj, device)

    traj_params = [q for m in TRAJ_MODULES for q in getattr(model, m).parameters()]
    head_params = [q for n, q in model.named_parameters()
                   if n.split('.')[0] not in TRAJ_MODULES]
    if args.freeze_traj:
        for q in traj_params:
            q.requires_grad_(False)
        opt = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.traj_lr is not None:
        opt = torch.optim.AdamW([{'params': traj_params, 'lr': args.traj_lr},
                                 {'params': head_params, 'lr': args.lr}],
                                weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    w = torch.tensor([3.5, 1.0], device=device)
    crit = nn.CrossEntropyLoss(weight=w)

    ckpt = Path(__file__).parent / 'checkpoints' / f'best_ladder_{args.tag}.pth'
    ckpt.parent.mkdir(exist_ok=True)

    best_apv, best_ep, since = float('-inf'), 0, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        if args.freeze_traj:
            for m in TRAJ_MODULES:
                getattr(model, m).eval()
        tr = 0.0
        for b in train_loader:
            vf, pf, vis, vm, pm, y = _move(b, device)
            logits, _ = model(vf, pf, vis, vm, pm)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad],
                                     args.clip_norm)
            opt.step(); tr += loss.item()
        vl, vapv = _validate(model, val_loader, device, crit)
        logger.info("Ep %d/%d  train %.4f  val %.4f  val APv %.4f", ep, args.epochs,
                    tr / max(len(train_loader), 1), vl, vapv)
        if vapv > best_apv:
            best_apv, best_ep, since = vapv, ep, 0
            torch.save({'model_state_dict': model.state_dict(), 'val_apv': vapv, 'epoch': ep,
                        'config': {'top_k': args.top_k, 'half': args.half,
                                   'num_frames': num_frames, 'gate': args.gate,
                                   'bed': args.bed, 'feats': feats},
                        'tag': args.tag, 'seed': args.seed,
                        'ladder': {'init_traj': args.init_traj, 'freeze_traj': args.freeze_traj,
                                   'traj_lr': args.traj_lr, 'no_vision': args.no_vision,
                                   'shuffle_vision': args.shuffle_vision}},
                       ckpt)
        else:
            since += 1
            if since >= args.patience:
                logger.info("Early stop at ep %d", ep); break
    logger.info("DONE tag=%s  best val APv %.4f (ep %d)  → %s", args.tag, best_apv, best_ep, ckpt.name)


if __name__ == '__main__':
    main()
