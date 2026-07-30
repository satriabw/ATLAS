"""Additivity probe: does frozen r2plus1d vision ADD to the trajectory representation,
beyond extra capacity? Representation-level test, no fusion-model retraining.

Arms (identical probe head for all — the fusion classifier shape Linear->BN->ReLU->Drop->Linear):
  traj      : head on f_traj (128)                      — the frozen anchor motion repr
  concat    : head on [f_traj | vision (512)] (640)     — traj + real frozen-Kinetics r2plus1d
  placebo   : head on [f_traj | shuffled vision] (640)  — same capacity, vision content broken

f_traj is computed exactly as GatedFusionModel does (vehicle/ped GRU + cross-attn + masked
max-pool), weights from the frozen traj core in best_ladder_lf_kin_s0.pth (== best_traj_whole
0.802 anchor). Vision = raw pooled clip vector from r2_whole_feats_kinetics.h5 (frozen Kinetics,
the leakage-free set; fine-tuned feats leak and would inflate additivity).

Protocol matches the arc: fit on train (odd videos), eval held-out test (even videos), class
weights [3.5,1]. Per-event test scores written to CSV (video_id,v_track_id,roi,gt_label,score)
for the existing video-level paired bootstrap (evaluation/bootstrap_ci.py, B=2000). Both seeds.
Placebo is the capacity control (catches "concat>traj" being capacity, not vision) exactly like
the AP-level placebo arms earlier in this arc.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'training'))
from dataset.wholetrack_fusion_data import WholeTrackFusionDataset
from models.gated_fusion import GatedFusionModel
from models.classifier import _encode_peds

ROOT = '/home/satria/Project/ATLAS'
FEATS = sys.argv[2] if len(sys.argv) > 2 else 'r2_whole_feats_kinetics.h5'
CKPT = 'training/checkpoints/best_ladder_lf_kin_s0.pth'
OUT = Path(ROOT) / 'artifacts/experiments/2026-07-23_leakage_free_fusion'
TOP_K, NUM_FRAMES = 5, 64


def extract(split, model, device):
    ds = WholeTrackFusionDataset(ROOT, split, feats_name=FEATS, top_k=TOP_K, num_frames=NUM_FRAMES)
    loader = DataLoader(ds, batch_size=64, num_workers=0)
    ftraj, vis, y = [], [], []
    with torch.no_grad():
        for b in loader:
            vf = b['vehicle_feat'].to(device); pf = b['ped_feat'].to(device)
            vm = b['v_padding_mask'].to(device); pm = b['p_padding_mask'].to(device)
            veh = model.vehicle_encoder(vf)
            ped, pkm = _encode_peds(model.ped_encoder, pf, pm, TOP_K, NUM_FRAMES)
            att, _ = model.cross_attn(veh, ped, ped, key_padding_mask=pkm)
            att = att + veh
            att = att.masked_fill(vm.unsqueeze(-1), float('-inf'))
            f = torch.nan_to_num(att.max(1).values, neginf=0.0)          # (B,128)
            ftraj.append(f.cpu().numpy()); vis.append(b['vis_feat'].numpy())
    keys = [(l.video_id, str(l.tracking_id), l.roi) for l in ds.labels]
    lab = np.array([l.annotation for l in ds.labels], dtype=np.int64)
    return np.concatenate(ftraj), np.concatenate(vis), lab, keys


HEAD = sys.argv[1] if len(sys.argv) > 1 else 'linear'   # 'linear' (convex) | 'mlp'


def head(d):
    if HEAD == 'linear':
        return nn.Linear(d, 2)   # multinomial logistic regression = CONVEX: one global
        #                          optimum, no shared-capacity dynamics to get stuck in.
    return nn.Sequential(nn.Linear(d, 64), nn.BatchNorm1d(64), nn.ReLU(),
                         nn.Dropout(0.3), nn.Linear(64, 2))


def fit_scores(Xtr, ytr, Xte, seed, epochs=1500, wd=1e-2):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_t = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    Xte_t = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    ytr_t = torch.tensor(ytr)
    net = head(Xtr.shape[1]); lossf = nn.CrossEntropyLoss(weight=torch.tensor([3.5, 1.0]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
    net.train()
    for _ in range(epochs):
        opt.zero_grad(); lossf(net(Xtr_t), ytr_t).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return torch.softmax(net(Xte_t), 1)[:, 0].numpy()   # P(violation)


def write_csv(path, keys, y, scores):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['video_id', 'v_track_id', 'roi', 'gt_label', 'score'])
        for (vid, tid, roi), lbl, s in zip(keys, y, scores):
            w.writerow([vid, tid, roi, int(lbl), float(s)])


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = GatedFusionModel(top_k=TOP_K, num_frames=NUM_FRAMES, gate=True).to(device)
    m.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)['model_state_dict'])
    m.eval()

    ftr, vtr, ytr, _ = extract('train', m, device)
    fte, vte, yte, kte = extract('test', m, device)
    from evaluation.ap_calculator import compute_ap
    def apv(y, s): return compute_ap([{'gt_label': int(y[i]), 'score': float(s[i])}
                                      for i in range(len(y))], target_class=0, score_key='score')

    for seed in (0, 1):
        pr = np.random.RandomState(100 + seed)
        perm_tr = pr.permutation(len(vtr)); perm_te = pr.permutation(len(vte))
        arms = {
            'traj':    (ftr,                              fte),
            'concat':  (np.hstack([ftr, vtr]),            np.hstack([fte, vte])),
            'placebo': (np.hstack([ftr, vtr[perm_tr]]),   np.hstack([fte, vte[perm_te]])),
        }
        ftag = 'quad' if 'quadrant' in FEATS else 'union'
        print(f"\n## seed {seed}  (head={HEAD}, feats={ftag})")
        for name, (Xtr, Xte) in arms.items():
            s = fit_scores(Xtr, ytr, Xte, seed)
            write_csv(OUT / f'add_{ftag}_{HEAD}_{name}_s{seed}.csv', kte, yte, s)
            print(f"  {name:8s} d={Xtr.shape[1]:3d}  test APv {apv(yte, s):.4f}")


if __name__ == '__main__':
    main()
