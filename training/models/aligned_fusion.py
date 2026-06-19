"""Aligned joint fusion model (2026-06-19, plan =
artifacts/docs/2026-06-19_joint_fusion/plan.md).

Simple, explainable, temporally-aligned joint fusion of trajectory + FROZEN
precomputed appearance features (centered_vision_feats.h5, per-slot aligned).

  motion:      vehicle/ped traj → cross-attention core → attended_t   (B,T,H)
  appearance:  frozen vis_feat → Linear proj            → vis_t        (B,T,H)
  fuse:        f_t = Linear([attended_t | vis_t])                      (B,T,H)
  select:      masked-softmax temporal attention a_t over valid slots  (B,T)
  classify:    z = Σ_t a_t·f_t → MLP                                   (B,2)
  aux:         vis_head on masked-mean-pooled vis_t (anti-dominance)   (B,2)

No vision CNN in the graph (features are precomputed) → the §11 poisoning
mechanism is structurally absent. self.ablate ∈ {None,'no_vision','no_traj'}
zeroes a branch for the contribution gate.
"""
import torch
import torch.nn as nn

from .trajectory_encoder import TrajectoryEncoder
from .classifier import _encode_peds


class AlignedFusionModel(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, num_classes=2, top_k=5,
                 num_frames=64, vis_dim=512, pool='attn'):
        super().__init__()
        self.top_k = top_k
        self.num_frames = num_frames
        self.pool = pool   # 'attn' = masked-softmax temporal selector; 'max' = masked max-pool
        self.vehicle_encoder = TrajectoryEncoder(input_dim=3, hidden_dim=hidden_dim)
        self.ped_encoder     = TrajectoryEncoder(input_dim=3, hidden_dim=hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.vis_proj   = nn.Linear(vis_dim, hidden_dim)
        self.fuse       = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1)   # temporal attention selector
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes))
        # unimodal vision aux head — forces appearance to be independently
        # discriminative before fusion can ignore it (anti-dominance fix).
        self.vis_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes))
        self.ablate = None

    def forward(self, vehicle_feat, ped_feat, vis_feat, v_padding_mask=None, p_padding_mask=None):
        vehicle_enc = self.vehicle_encoder(vehicle_feat)                      # (B,T,H)
        ped_enc, ped_key_mask = _encode_peds(
            self.ped_encoder, ped_feat, p_padding_mask, self.top_k, self.num_frames)
        attended, _ = self.cross_attn(vehicle_enc, ped_enc, ped_enc, key_padding_mask=ped_key_mask)
        attended = attended + vehicle_enc                                     # motion per slot
        vis = self.vis_proj(vis_feat)                                        # appearance per slot

        if self.ablate == 'no_vision':
            vis = torch.zeros_like(vis)
        elif self.ablate == 'no_traj':
            attended = torch.zeros_like(attended)

        f = self.fuse(torch.cat([attended, vis], dim=-1))                    # (B,T,H)

        if self.pool == 'max':
            # masked max-pool (same head as the standalone 0.681 CrossAttentionModel)
            fm = f.masked_fill(v_padding_mask.unsqueeze(-1), float('-inf')) if v_padding_mask is not None else f
            z = torch.nan_to_num(fm.max(dim=1).values, neginf=0.0)           # (B,H)
            a = torch.zeros(f.shape[0], f.shape[1], device=f.device)         # no attention in max mode
        else:
            # masked-softmax temporal attention selector: additive -inf on the
            # logits (NOT on f), nan-guard for the degenerate all-padded row.
            logits = self.attn_score(f).squeeze(-1)                          # (B,T)
            if v_padding_mask is not None:
                logits = logits.masked_fill(v_padding_mask, float('-inf'))
            a = torch.softmax(logits, dim=1)
            a = torch.nan_to_num(a, nan=0.0)
            z = (a.unsqueeze(-1) * f).sum(dim=1)                             # (B,H)
        main = self.classifier(z)

        # aux vision head on masked-mean-pooled appearance (uses real vis even
        # under ablation at train time — ablate is only set at eval).
        if v_padding_mask is not None:
            valid = (~v_padding_mask).float().unsqueeze(-1)
            vpool = (vis * valid).sum(1) / valid.sum(1).clamp(min=1)
        else:
            vpool = vis.mean(1)
        aux_vis = self.vis_head(vpool)
        return main, aux_vis, a
