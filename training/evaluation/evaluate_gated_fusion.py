"""Evaluate the gated fusion model on the test set (2026-07-08).

Computes test APv, writes a per-event predictions CSV (with video_id for the
video-level bootstrap, plus per-event mean gate values gate_traj / gate_vis
for the explainability readout), and prints gate summary stats:
  - mean/std of the vision- and traj-half gate values across events
  - fraction of events with a saturated vision gate (≈0 or ≈1) — the
    "gate collapsed" check from the plan.
Supports --ablate {no_vision,no_traj} and --shuffle-vision (placebo stream).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset.aligned_fusion_data import AlignedFusionDataset
from dataset.wholetrack_fusion_data import WholeTrackFusionDataset
from models.gated_fusion import GatedFusionModel
from evaluation.ap_calculator import compute_ap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_root', default='/home/satria/Project/ATLAS')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--ablate', choices=['no_vision', 'no_traj'], default=None)
    p.add_argument('--shuffle-vision', action='store_true', help='eval the placebo test stream')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--out-csv', default=None)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ck['config']

    # bed/feats absent in pre-2026-07-09 checkpoints → centered defaults
    if cfg.get('bed', 'centered') == 'whole':
        ds = WholeTrackFusionDataset(args.data_root, 'test',
                                     feats_name=cfg.get('feats', 'r2_whole_feats.h5'),
                                     top_k=cfg['top_k'], num_frames=cfg['num_frames'],
                                     shuffle_vision=args.shuffle_vision, seed=0)
    else:
        ds = AlignedFusionDataset(args.data_root, 'test',
                                  feats_name=cfg.get('feats', 'centered_vision_feats.h5'),
                                  top_k=cfg['top_k'], half=cfg['half'],
                                  shuffle_vision=args.shuffle_vision, seed=0)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0)

    model = GatedFusionModel(top_k=cfg['top_k'], num_frames=cfg['num_frames'],
                             gate=cfg['gate']).to(device)
    model.load_state_dict(ck['model_state_dict'])
    model.ablate = args.ablate
    model.eval()

    rows = []
    with torch.no_grad():
        for bi, b in enumerate(loader):
            vf = b['vehicle_feat'].to(device); pf = b['ped_feat'].to(device)
            vis = b['vis_feat'].to(device); vm = b['v_padding_mask'].to(device)
            pm = b['p_padding_mask'].to(device)
            logits, g = model(vf, pf, vis, vm, pm)
            pv = torch.softmax(logits.float(), 1)[:, 0].cpu().numpy()
            if g is not None:
                g_traj, g_vis = (h.mean(1).cpu().numpy() for h in g.chunk(2, dim=-1))
            else:
                g_traj = g_vis = np.full(len(pv), np.nan)
            base = bi * args.batch_size
            for k in range(len(pv)):
                lbl = ds.labels[base + k]
                rows.append({'video_id': lbl.video_id, 'v_track_id': lbl.tracking_id,
                             'roi': lbl.roi, 'gt_label': lbl.annotation, 'score': float(pv[k]),
                             'gate_traj': float(g_traj[k]), 'gate_vis': float(g_vis[k])})

    apv = compute_ap(rows, target_class=0, score_key='score')
    tag = args.ablate or ('placebo' if args.shuffle_vision else 'full')
    print(f"=== gated-fusion [{tag}] gate={cfg['gate']}  n={len(rows)}  test APv = {apv:.4f} ===")

    if cfg['gate']:
        gt = np.array([r['gate_traj'] for r in rows])
        gv = np.array([r['gate_vis'] for r in rows])
        sat = np.mean((gv < 0.05) | (gv > 0.95))
        print(f"gate readout: traj mean {gt.mean():.3f} ± {gt.std():.3f}   "
              f"vis mean {gv.mean():.3f} ± {gv.std():.3f}   "
              f"vis saturated (<0.05 or >0.95): {sat:.1%}")

    if args.out_csv:
        out = Path(args.out_csv)
        with open(out, 'w', newline='') as f:
            wcsv = csv.DictWriter(f, fieldnames=['video_id', 'v_track_id', 'roi', 'gt_label',
                                                 'score', 'gate_traj', 'gate_vis'])
            wcsv.writeheader(); wcsv.writerows(rows)
        print(f"CSV → {out}")


if __name__ == '__main__':
    main()
