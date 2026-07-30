"""Direct linear/shallow probe on the FROZEN vision representation → violation label.

Answers "does this representation carry label signal?" without any fusion model, by
fitting a probe on the raw pooled r2plus1d clip vector (exactly the 512-d `vpool` that
enters the fusion model before its trained adapter) and scoring held-out test APv.

Protocol: fit on ALL train (odd videos), evaluate on the held-out test set (even videos)
— the same scene-separated split the 0.802 traj anchor uses, so APv is directly
comparable. Controls: label-permutation null (refit on shuffled feature<->label pairs,
N=30) and the violation-prevalence floor. If real APv clears the null/prevalence
meaningfully → representation has signal (fusion is failing to use it, gating/alignment
back on the table). If real APv is flat at the null → direct confirmation of "not
label-discriminative" → the crop is the lever.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'training'))
from dataset.wholetrack_fusion_data import WholeTrackFusionDataset
from evaluation.ap_calculator import compute_ap

ROOT = '/home/satria/Project/ATLAS'


def load(split, feats):
    ds = WholeTrackFusionDataset(ROOT, split, feats_name=feats, top_k=5, num_frames=64)
    f = h5py.File(Path(ROOT) / 'data/raw/video' / feats, 'r')
    X = np.stack([f[k][:] for k in ds.keys]).astype(np.float32)   # (N,512) pooled clip vec
    y = np.array([l.annotation for l in ds.labels], dtype=np.int64)  # 0=violation,1=compliance
    f.close()
    return X, y


def apv(rows):
    return compute_ap(rows, target_class=0, score_key='score')


def fit_probe(Xtr, ytr, Xte, yte, hidden=0, epochs=400, wd=1e-2, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_t = torch.tensor((Xtr - mu) / sd); Xte_t = torch.tensor((Xte - mu) / sd)
    ytr_t = torch.tensor(ytr)
    if hidden > 0:
        net = nn.Sequential(nn.Linear(512, hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, 2))
    else:
        net = nn.Linear(512, 2)
    opt = nn.CrossEntropyLoss(weight=torch.tensor([3.5, 1.0]))  # match pipeline class weights
    optim = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
    net.train()
    for _ in range(epochs):
        optim.zero_grad(); loss = opt(net(Xtr_t), ytr_t); loss.backward(); optim.step()
    net.eval()
    with torch.no_grad():
        pv_te = torch.softmax(net(Xte_t), 1)[:, 0].numpy()
        pv_tr = torch.softmax(net(Xtr_t), 1)[:, 0].numpy()
    rows_te = [{'gt_label': int(yte[i]), 'score': float(pv_te[i])} for i in range(len(yte))]
    rows_tr = [{'gt_label': int(ytr[i]), 'score': float(pv_tr[i])} for i in range(len(ytr))]
    return apv(rows_te), apv(rows_tr)


def report(name, feats):
    Xtr, ytr = load('train', feats); Xte, yte = load('test', feats)
    prev = float((yte == 0).mean())
    print(f"\n######## {name}  ({feats}) ########")
    print(f"train N={len(ytr)} (V={int((ytr==0).sum())})   test N={len(yte)} (V={int((yte==0).sum())})   "
          f"test violation prevalence = {prev:.4f}  (APv floor)")
    for hidden in (0, 128):
        tag = 'linear' if hidden == 0 else f'MLP-{hidden}'
        te, tr = fit_probe(Xtr, ytr, Xte, yte, hidden=hidden)
        # label-permutation null on the SAME probe/hyperparams
        rng = np.random.RandomState(0); null = []
        for s in range(30):
            yp = ytr[rng.permutation(len(ytr))]
            null.append(fit_probe(Xtr, yp, Xte, yte, hidden=hidden, seed=s)[0])
        null = np.array(null)
        p = float((null >= te).mean())
        z = (te - null.mean()) / (null.std() + 1e-9)
        print(f"  {tag:8s}  test APv {te:.4f}  (train {tr:.4f})   "
              f"null {null.mean():.4f}±{null.std():.4f}  z={z:+.1f}  p={p:.3f}")


if len(sys.argv) > 1:
    report(f'CUSTOM {sys.argv[1]}', sys.argv[1])
else:
    report('FROZEN Kinetics (primary)', 'r2_whole_feats_kinetics.h5')
    report('ATLAS fine-tuned (context, leaky)', 'r2_whole_feats.h5')
print("\nanchors: traj-only 0.8019 | fine-tuned standalone r2plus1d 0.643 | union frozen-Kinetics 0.44 | prevalence ~0.257")
