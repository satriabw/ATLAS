"""Gated fusion model (2026-07-08, plan =
artifacts/docs/2026-07-08_gated_fusion/plan.md).

The gated fusion block from LRC-WeatherNet (214148-0073.pdf, Fig. 3): each
stream is reduced to ONE vector, a sigmoid gate looks at both and outputs a
0-1 weight per channel, the ORIGINAL vectors are multiplied by their gate
half, and the gated vectors are concatenated into the classifier.

  motion:      traj cross-attention core → masked max-pool     f_traj (B,H)
  appearance:  frozen per-slot feats → masked mean → adapter   f_vis  (B,H)
  gate:        σ(W_g [ReLU(W_t f_traj) | ReLU(W_v f_vis)])     g      (B,2H)
  fuse:        [f_traj ⊙ g_traj | f_vis ⊙ g_vis]               (B,2H)
  classify:    Linear → BN → ReLU → dropout → Linear           (B,2)

gate=False removes steps 3-4 (plain concat control, everything else equal).
Vision features are precomputed (no CNN in the graph). self.ablate ∈
{None,'no_vision','no_traj'} zeroes a stream at eval for the contribution check.
"""
import torch
import torch.nn as nn

from .trajectory_encoder import TrajectoryEncoder
from .classifier import _encode_peds


class GatedFusionModel(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, num_classes=2, top_k=5,
                 num_frames=64, vis_dim=512, gate=True):
        super().__init__()
        self.top_k = top_k
        self.num_frames = num_frames
        self.gate = gate
        self.vehicle_encoder = TrajectoryEncoder(input_dim=3, hidden_dim=hidden_dim)
        self.ped_encoder     = TrajectoryEncoder(input_dim=3, hidden_dim=hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        # adapter = last layer of the vision "backbone": equalizes widths (512→H)
        # so neither stream outnumbers the other in the concat (plan: equal widths).
        self.vis_adapter = nn.Sequential(nn.Linear(vis_dim, hidden_dim), nn.ReLU())
        # gate block (Fig. 3): per-stream projection MLPs + one sigmoid gating layer
        self.proj_traj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.proj_vis  = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.gate_fc   = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        # classifier head (paper §III-E shape, project scale)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, num_classes))
        self.ablate = None

    def forward(self, vehicle_feat, ped_feat, vis_feat, v_padding_mask=None, p_padding_mask=None):
        # --- motion stream: same core + masked max-pool as the 0.681 CrossAttentionModel
        vehicle_enc = self.vehicle_encoder(vehicle_feat)                     # (B,T,H)
        ped_enc, ped_key_mask = _encode_peds(
            self.ped_encoder, ped_feat, p_padding_mask, self.top_k, self.num_frames)
        attended, _ = self.cross_attn(vehicle_enc, ped_enc, ped_enc, key_padding_mask=ped_key_mask)
        attended = attended + vehicle_enc
        if v_padding_mask is not None:
            attended = attended.masked_fill(v_padding_mask.unsqueeze(-1), float('-inf'))
        f_traj = torch.nan_to_num(attended.max(dim=1).values, neginf=0.0)    # (B,H)

        # --- appearance stream: (B,512) = already-pooled clip feature (r2plus1d
        # bed) used as-is; (B,T,512) = per-slot feats, masked mean over valid
        # slots (pad slots are feature-level zeros in the h5, mask excludes
        # them from the average)
        if vis_feat.dim() == 2:
            vpool = vis_feat
        elif v_padding_mask is not None:
            valid = (~v_padding_mask).float().unsqueeze(-1)
            vpool = (vis_feat * valid).sum(1) / valid.sum(1).clamp(min=1)
        else:
            vpool = vis_feat.mean(1)
        f_vis = self.vis_adapter(vpool)                                      # (B,H)

        if self.ablate == 'no_vision':
            f_vis = torch.zeros_like(f_vis)
        elif self.ablate == 'no_traj':
            f_traj = torch.zeros_like(f_traj)

        if self.gate:
            g = torch.sigmoid(self.gate_fc(
                torch.cat([self.proj_traj(f_traj), self.proj_vis(f_vis)], dim=-1)))  # (B,2H)
            g_traj, g_vis = g.chunk(2, dim=-1)
            fused = torch.cat([f_traj * g_traj, f_vis * g_vis], dim=-1)
        else:
            g = None
            fused = torch.cat([f_traj, f_vis], dim=-1)
        return self.classifier(fused), g
