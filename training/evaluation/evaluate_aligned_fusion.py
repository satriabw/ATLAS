"""Evaluate the aligned joint-fusion model on the test set (S2, 2026-06-19).

Computes test APv, writes a per-event predictions CSV (with video_id for
video-level bootstrap), supports --ablate {no_vision,no_traj} for the
contribution gate, and (when not ablated) reports the attention-validity check:
does the temporal selector's argmax land near the construction-known interaction
slot (centre = half)?  Attention != explanation (Jain & Wallace), so this is a
required falsifiable check, not decoration.
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
from models.aligned_fusion import AlignedFusionModel
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
    half = cfg['half']

    ds = AlignedFusionDataset(args.data_root, 'test', top_k=cfg['top_k'], half=half,
                              shuffle_vision=args.shuffle_vision, seed=0)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0)

    model = AlignedFusionModel(top_k=cfg['top_k'], num_frames=cfg['num_frames'],
                               pool=cfg.get('pool', 'attn')).to(device)
    model.load_state_dict(ck['model_state_dict'])
    model.ablate = args.ablate
    model.eval()

    rows, argmax_slots = [], []
    with torch.no_grad():
        for bi, b in enumerate(loader):
            vf = b['vehicle_feat'].to(device); pf = b['ped_feat'].to(device)
            vis = b['vis_feat'].to(device); vm = b['v_padding_mask'].to(device)
            pm = b['p_padding_mask'].to(device)
            main, _, a = model(vf, pf, vis, vm, pm)
            pv = torch.softmax(main.float(), 1)[:, 0].cpu().numpy()
            am = a.argmax(1).cpu().numpy()
            n = len(pv)
            base = bi * args.batch_size
            for k in range(n):
                lbl = ds.labels[base + k]
                rows.append({'video_id': lbl.video_id, 'v_track_id': lbl.tracking_id,
                             'roi': lbl.roi, 'gt_label': lbl.annotation, 'score': float(pv[k])})
                argmax_slots.append(int(am[k]))

    apv = compute_ap(rows, target_class=0, score_key='score')
    tag = args.ablate or ('placebo' if args.shuffle_vision else 'full')
    print(f"=== aligned-fusion [{tag}]  n={len(rows)}  test APv = {apv:.4f} ===")

    if args.ablate is None and not args.shuffle_vision and cfg.get('pool', 'attn') == 'attn':
        slots = np.array(argmax_slots)
        near = np.mean(np.abs(slots - half) <= 8)
        # uniform-attention baseline: P(|U(0,2h-1) - h| <= 8)
        unif = min(2 * 8 + 1, 2 * half) / (2 * half)
        print(f"attention-validity: argmax within ±8 of centre(slot {half}) = "
              f"{near:.3f}  (uniform baseline {unif:.3f})  median argmax {int(np.median(slots))}")

    if args.out_csv:
        out = Path(args.out_csv)
        with open(out, 'w', newline='') as f:
            wcsv = csv.DictWriter(f, fieldnames=['video_id', 'v_track_id', 'roi', 'gt_label', 'score'])
            wcsv.writeheader(); wcsv.writerows(rows)
        print(f"CSV → {out}")


if __name__ == '__main__':
    main()
