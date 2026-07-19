"""Joint gated multimodal training, instrumented (2026-07-10, plan =
artifacts/docs/2026-07-10_joint_gated/plan.md).

End-to-end training of JointGatedFusionModel: traj core initialized from the
0.802 whole-track checkpoint, r2plus1d backbone from the 0.643 rebuild, head
from scratch. Differential LRs protect the pretrained parts (modality-competition
countermeasure validated 2026-07-08).

The per-epoch diagnostics are the point of this run (localize WHICH mechanism
breaks): val APv full / no_vision / no_traj, gate means, relative L2 weight
drift of traj core and backbone trainable layers, train/val loss.

After training, reloads the best checkpoint and runs the test eval once
(APv + gate readout + per-event CSV, same format as evaluate_gated_fusion.py).
"""
import argparse
import csv
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.loader import load_violation_dataset
from train import _scene_split
from train_fusion_ladder import TRAJ_MODULES, _load_traj_core
from models.joint_gated import JointGatedFusionModel
from evaluation.ap_calculator import compute_ap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HEAD_MODULES = ('vis_adapter', 'proj_traj', 'proj_vis', 'gate_fc', 'classifier')


def _move(batch, device):
    return (batch['vehicle_feat'].to(device), batch['ped_feat'].to(device),
            batch['frames'].to(device), batch['v_padding_mask'].to(device),
            batch['p_padding_mask'].to(device), batch['label'].to(device))


def _load_vision_backbone(model, ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)['model_state_dict']
    sub = {k: v for k, v in sd.items() if k.startswith('vision_encoder.')}
    missing, unexpected = model.load_state_dict(sub, strict=False)
    assert not unexpected, unexpected
    still = [k for k in missing if k.startswith('vision_encoder.')]
    assert not still, f"vision_encoder keys not covered by {ckpt_path}: {still}"
    logger.info("Loaded %d vision_encoder tensors from %s", len(sub), ckpt_path)


def _flat(params):
    return torch.cat([p.detach().float().reshape(-1) for p in params])


class DriftMeter:
    def __init__(self, model):
        self.traj_params = [p for m in TRAJ_MODULES for p in getattr(model, m).parameters()]
        self.vis_params = [p for p in model.vision_encoder.parameters() if p.requires_grad]
        self.traj0 = _flat(self.traj_params).clone()
        self.vis0 = _flat(self.vis_params).clone()

    def read(self):
        t = _flat(self.traj_params)
        v = _flat(self.vis_params)
        return (torch.norm(t - self.traj0) / torch.norm(self.traj0)).item(), \
               (torch.norm(v - self.vis0) / torch.norm(self.vis0)).item()


@torch.no_grad()
def _val_pass(model, loader, device, crit, ablate=None):
    model.eval()
    model.ablate = ablate
    preds, tot, gts, gvs = [], 0.0, [], []
    for b in loader:
        vf, pf, fr, vm, pm, y = _move(b, device)
        logits, g = model(vf, pf, fr, vm, pm)
        tot += crit(logits, y).item()
        pv = torch.softmax(logits.float(), 1)[:, 0]
        preds.extend({'gt_label': int(t), 'score': float(s)} for t, s in zip(y.cpu(), pv.cpu()))
        if g is not None and ablate is None:
            gt, gv = g.chunk(2, dim=-1)
            gts.append(gt.mean().item()); gvs.append(gv.mean().item())
    model.ablate = None
    apv = compute_ap(preds, target_class=0, score_key='score')
    gate = (float(np.mean(gts)), float(np.mean(gvs))) if gts else (float('nan'),) * 2
    return tot / max(len(loader), 1), apv, gate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_root', default='/home/satria/Project/ATLAS')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4, help='head LR')
    p.add_argument('--traj-lr', type=float, default=1e-5)
    p.add_argument('--backbone-lr', type=float, default=1e-5)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--clip_norm', type=float, default=5.0)
    p.add_argument('--top_k', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--patience', type=int, default=6)
    p.add_argument('--h5', default='frames_r2.h5')
    p.add_argument('--init-traj', default='checkpoints/best_traj_whole.pth')
    p.add_argument('--init-vision', default='checkpoints/best_r2plus1d_r2_rebuild.pth')
    p.add_argument('--tag', required=True)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    full = load_violation_dataset(Path(args.data_root), 'train', num_frames=32,
                                  top_k=args.top_k, use_vision=True, h5_name=args.h5)
    train_ds, val_ds, train_idx = _scene_split(full, 42)
    tl = [full.labels[i].annotation for i in train_idx]
    logger.info("Train: %d (V=%d C=%d)  Val: %d", len(train_ds), tl.count(0), tl.count(1), len(val_ds))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)

    model = JointGatedFusionModel(top_k=args.top_k, num_frames=32, gate=True).to(device)
    _load_traj_core(model, args.init_traj, device)
    _load_vision_backbone(model, args.init_vision, device)

    traj_params = [q for m in TRAJ_MODULES for q in getattr(model, m).parameters()]
    head_params = [q for m in HEAD_MODULES for q in getattr(model, m).parameters()]
    vis_params = [q for q in model.vision_encoder.parameters() if q.requires_grad]
    opt = torch.optim.AdamW([
        {'params': head_params, 'lr': args.lr},
        {'params': traj_params, 'lr': args.traj_lr},
        {'params': vis_params, 'lr': args.backbone_lr},
    ], weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([3.5, 1.0]).to(device))
    scaler = torch.amp.GradScaler('cuda')
    drift = DriftMeter(model)

    ckpt = Path('checkpoints') / f'best_joint_{args.tag}.pth'
    best_apv, since = -1.0, 0
    for ep in range(1, args.epochs + 1):
        model.train(); model.ablate = None
        tot = 0.0
        for b in train_loader:
            vf, pf, fr, vm, pm, y = _move(b, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                logits, _ = model(vf, pf, fr, vm, pm)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            scaler.step(opt); scaler.update()
            tot += loss.item()
        train_loss = tot / max(len(train_loader), 1)

        val_loss, apv_full, (g_traj, g_vis) = _val_pass(model, val_loader, device, crit)
        _, apv_novis, _ = _val_pass(model, val_loader, device, crit, ablate='no_vision')
        _, apv_notraj, _ = _val_pass(model, val_loader, device, crit, ablate='no_traj')
        d_traj, d_vis = drift.read()
        logger.info(
            "Ep %d/%d  train %.4f  val %.4f  APv full %.4f | no_vis %.4f | no_traj %.4f  "
            "gate t %.3f v %.3f  drift traj %.4f vis %.4f",
            ep, args.epochs, train_loss, val_loss, apv_full, apv_novis, apv_notraj,
            g_traj, g_vis, d_traj, d_vis)

        if apv_full > best_apv:
            best_apv, since = apv_full, 0
            torch.save({'model_state_dict': model.state_dict(),
                        'config': {'top_k': args.top_k, 'num_frames': 32, 'gate': True,
                                   'joint': True, 'h5': args.h5},
                        'tag': args.tag, 'seed': args.seed, 'epoch': ep},
                       ckpt)
        else:
            since += 1
            if since >= args.patience:
                logger.info("Early stop at epoch %d", ep)
                break
    logger.info("DONE tag=%s  best val APv %.4f  → %s", args.tag, best_apv, ckpt)

    # ---- single test eval (once, at the end) ----
    del full, train_loader, val_loader
    torch.cuda.empty_cache()
    test_ds = load_violation_dataset(Path(args.data_root), 'test', num_frames=32,
                                     top_k=args.top_k, use_vision=True, h5_name=args.h5)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=2)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)['model_state_dict'])
    model.eval(); model.ablate = None
    rows = []
    with torch.no_grad():
        for bi, b in enumerate(test_loader):
            vf, pf, fr, vm, pm, y = _move(b, device)
            logits, g = model(vf, pf, fr, vm, pm)
            pv = torch.softmax(logits.float(), 1)[:, 0].cpu().numpy()
            g_t, g_v = (h.mean(1).cpu().numpy() for h in g.chunk(2, dim=-1))
            base = bi * args.batch_size
            for k in range(len(pv)):
                lbl = test_ds.labels[base + k]
                rows.append({'video_id': lbl.video_id, 'v_track_id': lbl.tracking_id,
                             'roi': lbl.roi, 'gt_label': lbl.annotation, 'score': float(pv[k]),
                             'gate_traj': float(g_t[k]), 'gate_vis': float(g_v[k])})
    apv = compute_ap(rows, target_class=0, score_key='score')
    gt = np.array([r['gate_traj'] for r in rows]); gv = np.array([r['gate_vis'] for r in rows])
    sat = np.mean((gv < 0.05) | (gv > 0.95))
    print(f"=== joint-gated [{args.tag}]  n={len(rows)}  test APv = {apv:.4f} ===")
    print(f"gate readout: traj mean {gt.mean():.3f} ± {gt.std():.3f}   "
          f"vis mean {gv.mean():.3f} ± {gv.std():.3f}   vis saturated: {sat:.1%}")
    out = Path('checkpoints') / f'best_joint_{args.tag}_predictions.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['video_id', 'v_track_id', 'roi', 'gt_label',
                                          'score', 'gate_traj', 'gate_vis'])
        w.writeheader(); w.writerows(rows)
    print(f"CSV → {out}")


if __name__ == '__main__':
    main()
