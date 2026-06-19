"""Step 1 — frozen-branch fusion head (2026-06-18).

Tests whether the FUSION ARCHITECTURE (not the signal) was what broke the joint
fused models. Both branches are FROZEN standalone centered checkpoints, so the
combiner cannot corrupt the trajectory encoder. A tiny head learns to combine
their 2-d pre-softmax logits.

Bars (printed): traj-centered alone, and analytic centered late log-odds fusion
(weight swept on val). Success = learned head >= log-odds AND > traj alone.

Run from training/. Needs best_traj_centered_w{window}.pth (CrossAttentionModel)
and a frozen vision checkpoint (VisionOnlyModel, backbone from its ckpt).
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.centered_window import load_centered_dataset
from dataset.trajectory import DEFAULT_TOP_K
from models import CrossAttentionModel, VisionOnlyModel
from train import _scene_split
from evaluation.ap_calculator import compute_map

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


@torch.no_grad()
def _extract(traj_model, vis_model, ds, device, bs):
    """Return (traj_logits, vis_logits, labels, video_ids, v_track_ids) over ds."""
    loader = DataLoader(ds, batch_size=bs, shuffle=False)
    tl, vl, ys, vids, tids = [], [], [], [], []
    for b in loader:
        vf, pf = b['vehicle_feat'].to(device), b['ped_feat'].to(device)
        vm, pm = b['v_padding_mask'].to(device), b['p_padding_mask'].to(device)
        tl.append(traj_model(vf, pf, vm, pm).cpu())
        vl.append(vis_model(b['frames'].to(device)).cpu())
        ys.append(b['label'])
        vids += list(b['video_id']); tids += [int(t) for t in b['tracking_id']]
    return (torch.cat(tl), torch.cat(vl), torch.cat(ys), vids, tids)


def _preds(scores, ys, vids, tids):
    return [{'video_id': vids[i], 'v_track_id': tids[i], 'gt_label': int(ys[i]),
             'score': float(scores[i]), 'score_n': 1.0 - float(scores[i]), 'eiou': 1.0}
            for i in range(len(scores))]


def _apv(scores, ys, vids, tids):
    return compute_map(_preds(scores, ys, vids, tids))['APv']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='/home/satria/Project/ATLAS')
    ap.add_argument('--window',    type=int, default=64)
    ap.add_argument('--h5',        default='frames_union_centered.h5')
    ap.add_argument('--traj-ckpt', default='checkpoints/best_traj_centered_w64.pth')
    ap.add_argument('--vis-ckpt',  default='checkpoints/best_vision_union.pth')
    ap.add_argument('--top-k',     type=int, default=DEFAULT_TOP_K)
    ap.add_argument('--batch-size',type=int, default=32)
    ap.add_argument('--epochs',    type=int, default=200)
    ap.add_argument('--seed',      type=int, default=42)
    args = ap.parse_args()
    half = args.window // 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # frozen branches
    tj = torch.load(args.traj_ckpt, map_location=device, weights_only=False)
    traj_model = CrossAttentionModel(num_classes=2, top_k=args.top_k, num_frames=args.window).to(device)
    traj_model.load_state_dict(tj['model_state_dict']); traj_model.eval()
    vj = torch.load(args.vis_ckpt, map_location=device, weights_only=False)
    # VisionEncoder3D (r2plus1d) is identifiable by its unique _renorm buffers.
    backbone = 'r2plus1d' if any('_renorm' in k for k in vj['model_state_dict']) else 'resnet18'
    vis_model = VisionOnlyModel(num_classes=2, num_frames=args.window, backbone=backbone).to(device)
    vis_model.load_state_dict(vj['model_state_dict']); vis_model.eval()
    logger.info(f"traj={args.traj_ckpt} vis={args.vis_ckpt} (backbone={backbone})")

    # datasets: train (odd) → scene-split train/val ; test (even)
    full = load_centered_dataset(Path(args.data_root), 'train', top_k=args.top_k, half=half, h5_name=args.h5)
    tr_ds, va_ds, _ = _scene_split(full, args.seed)
    te_ds = load_centered_dataset(Path(args.data_root), 'test', top_k=args.top_k,
                                  video_filter=list(range(2, 121, 2)), half=half, h5_name=args.h5)

    tr = _extract(traj_model, vis_model, tr_ds, device, args.batch_size)
    va = _extract(traj_model, vis_model, va_ds, device, args.batch_size)
    te = _extract(traj_model, vis_model, te_ds, device, args.batch_size)

    # --- branch + analytic log-odds bars (on test) ---
    sm = lambda lg: torch.softmax(lg, dim=1)[:, 0]
    print(f"\n{'arm':<26}{'val APv':>9}{'test APv':>10}")
    print(f"{'traj-centered alone':<26}{_apv(sm(va[0]),va[2],va[3],va[4]):>9.4f}{_apv(sm(te[0]),te[2],te[3],te[4]):>10.4f}")
    print(f"{'vision-centered alone':<26}{_apv(sm(va[1]),va[2],va[3],va[4]):>9.4f}{_apv(sm(te[1]),te[2],te[3],te[4]):>10.4f}")

    # log-odds late fusion: combine P(violation) in log-odds space, sweep w on val
    eps = 1e-6
    def logodds(p): p = p.clamp(eps, 1 - eps); return torch.log(p / (1 - p))
    lo_tr_v, lo_tr_t = logodds(sm(va[0])), logodds(sm(te[0]))
    lo_vs_v, lo_vs_t = logodds(sm(va[1])), logodds(sm(te[1]))
    best_w, best_va = 0.5, -1
    for w in np.linspace(0, 1, 21):
        s = torch.sigmoid(w * lo_tr_v + (1 - w) * lo_vs_v)
        a = _apv(s, va[2], va[3], va[4])
        if a > best_va: best_va, best_w = a, w
    lo_test = _apv(torch.sigmoid(best_w * lo_tr_t + (1 - best_w) * lo_vs_t), te[2], te[3], te[4])
    print(f"{'log-odds (w_traj=%.2f)'%best_w:<26}{best_va:>9.4f}{lo_test:>10.4f}")

    # --- learned head on [traj_logits ; vis_logits] (4 -> 2) ---
    Xtr = torch.cat([tr[0], tr[1]], dim=1).to(device); Ytr = tr[2].to(device)
    Xva = torch.cat([va[0], va[1]], dim=1).to(device)
    Xte = torch.cat([te[0], te[1]], dim=1).to(device)
    head = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Dropout(0.3), nn.Linear(16, 2)).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([3.5, 1.0], device=device))
    best_state, best_head_va = None, -1
    for ep in range(args.epochs):
        head.train(); opt.zero_grad()
        loss = crit(head(Xtr), Ytr); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            sv = torch.softmax(head(Xva), dim=1)[:, 0].cpu()
        a = _apv(sv, va[2], va[3], va[4])
        if a > best_head_va:
            best_head_va = a; best_state = {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state); head.eval()
    with torch.no_grad():
        st = torch.softmax(head(Xte), dim=1)[:, 0].cpu()
    head_test = _apv(st, te[2], te[3], te[4])
    print(f"{'learned head (frozen)':<26}{best_head_va:>9.4f}{head_test:>10.4f}")
    print(f"\nVERDICT: learned head test APv {head_test:.4f} vs log-odds {lo_test:.4f} "
          f"vs traj {_apv(sm(te[0]),te[2],te[3],te[4]):.4f}")


if __name__ == '__main__':
    main()
