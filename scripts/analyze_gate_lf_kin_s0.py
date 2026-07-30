"""Gate-behaviour analysis on the ALREADY-TRAINED lf_kin_s0 model (no training).

Question (user, 2026-07-27): the gate is open (vis~0.497) yet performance == placebo.
Two hypotheses, distinguishable from the trained model alone:
  (A) mechanism problem — gate barely varies across instances AND doesn't shift when
      real vision is swapped for shuffled vision. It learned a fixed ~50/50 blend that
      ignores vision content.
  (B) representation problem — gate DOES vary per instance and DOES respond to real vs
      shuffled, but pooling already destroyed the useful signal before the gate.

We run the SAME model twice on the SAME test events: once with real vision, once with
shuffled vision (dataset shuffle_vision=True, seed 0), and capture the full per-channel
vis-gate vector g_vis (128-d) plus the pooled vision vector f_vis. Then:
  - across-instance variation of the vis gate (per-channel and channel-mean)
  - per-instance shift real->shuffled (does the gate move when content changes?)
  - as a bonus discriminator, across-instance variation of f_vis itself (did pooling
    leave any instance-to-instance signal for the gate to act on?)
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'training'))
from dataset.wholetrack_fusion_data import WholeTrackFusionDataset
from models.gated_fusion import GatedFusionModel

CKPT = 'training/checkpoints/best_ladder_lf_kin_s0.pth'
ROOT = '/home/satria/Project/ATLAS'


def run(shuffle):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = ck['config']
    ds = WholeTrackFusionDataset(ROOT, 'test', feats_name=cfg['feats'],
                                 top_k=cfg['top_k'], num_frames=cfg['num_frames'],
                                 shuffle_vision=shuffle, seed=0)
    loader = DataLoader(ds, batch_size=32, num_workers=0)
    model = GatedFusionModel(top_k=cfg['top_k'], num_frames=cfg['num_frames'], gate=cfg['gate']).to(device)
    model.load_state_dict(ck['model_state_dict']); model.eval()

    fvis_store = []
    h = model.vis_adapter.register_forward_hook(lambda m, i, o: fvis_store.append(o.detach().cpu()))

    gvis, gtraj, pv = [], [], []
    with torch.no_grad():
        for b in loader:
            vf = b['vehicle_feat'].to(device); pf = b['ped_feat'].to(device)
            vis = b['vis_feat'].to(device); vm = b['v_padding_mask'].to(device); pm = b['p_padding_mask'].to(device)
            logits, g = model(vf, pf, vis, vm, pm)
            gt, gv = g.chunk(2, dim=-1)
            gtraj.append(gt.cpu()); gvis.append(gv.cpu())
            pv.append(torch.softmax(logits.float(), 1)[:, 0].cpu())
    h.remove()
    return (torch.cat(gvis).numpy(), torch.cat(gtraj).numpy(),
            torch.cat(fvis_store).numpy(), torch.cat(pv).numpy())


gv_r, gt_r, fvis_r, pv_r = run(shuffle=False)
gv_s, _, fvis_s, pv_s = run(shuffle=True)
N = len(gv_r)
print(f"n test events = {N}\n")

# --- (1) across-instance variation of the vis gate ---
gvm_r = gv_r.mean(1)                       # per-event channel-mean vis gate
print("== vis gate, across-instance variation (real vision) ==")
print(f"  channel-mean scalar: mean {gvm_r.mean():.4f}  std {gvm_r.std():.4f}  "
      f"range [{gvm_r.min():.4f}, {gvm_r.max():.4f}]")
per_ch_std = gv_r.std(0)                   # std across events, per channel
print(f"  per-channel across-event std: mean {per_ch_std.mean():.4f}  "
      f"median {np.median(per_ch_std):.4f}  max {per_ch_std.max():.4f}")
print(f"  (a gate that conditions on content should show sizeable per-instance std)\n")

# --- (2) does the gate move when real vision -> shuffled vision? (same model, same events) ---
gvm_s = gv_s.mean(1)
d_scalar = gvm_r - gvm_s
d_perch = np.abs(gv_r - gv_s)             # (N,128)
print("== vis gate shift when vision content is swapped (real -> shuffled) ==")
print(f"  channel-mean scalar delta: mean {d_scalar.mean():+.5f}  mean|delta| {np.abs(d_scalar).mean():.5f}  "
      f"std {d_scalar.std():.5f}")
print(f"  per-channel mean|delta|: {d_perch.mean():.5f}  (vs per-channel std {per_ch_std.mean():.4f})")
r = np.corrcoef(gvm_r, gvm_s)[0, 1]
print(f"  corr(real vis-gate, shuffled vis-gate) across events: {r:+.4f}  "
      f"(~1.0 = gate ignores which vision it's fed)\n")

# --- (3) bonus discriminator: did pooling leave any instance signal in f_vis? ---
fmean = fvis_r.mean(0, keepdims=True)
rel_spread = np.linalg.norm(fvis_r - fmean, axis=1) / (np.linalg.norm(fmean) + 1e-8)
cos_to_mean = (fvis_r @ fmean.T).ravel() / (np.linalg.norm(fvis_r, axis=1) * np.linalg.norm(fmean) + 1e-8)
print("== pooled vision vector f_vis, across-instance variation (real vision) ==")
print(f"  per-event ||f_vis - mean|| / ||mean||: median {np.median(rel_spread):.4f}")
print(f"  cosine(f_vis, mean f_vis): median {np.median(cos_to_mean):.4f}  "
      f"(near 1.0 = pooled vision nearly constant across events)")
d_fvis = np.linalg.norm(fvis_r - fvis_s, axis=1) / (np.linalg.norm(fvis_r, axis=1) + 1e-8)
print(f"  relative change ||f_vis_real - f_vis_shuf|| / ||f_vis_real||: median {np.median(d_fvis):.4f}\n")

print("== read-out ==")
mech = (gvm_r.std() < 0.02) and (np.abs(d_scalar).mean() < 0.01) and (r > 0.9)
print(f"  gate varies across instances?  std {gvm_r.std():.4f}  -> {'NO' if gvm_r.std()<0.02 else 'yes'}")
print(f"  gate responds to real vs shuffled?  mean|delta| {np.abs(d_scalar).mean():.5f}  "
      f"-> {'NO' if np.abs(d_scalar).mean()<0.01 else 'yes'}")
print(f"  => {'(A) MECHANISM problem: gate is a fixed blend, ignores vision content' if mech else '(B) gate is adaptive -> representation/pooling problem'}")
