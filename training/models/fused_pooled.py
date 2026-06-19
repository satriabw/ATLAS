"""Pool-then-concat fusion (2026-06-18 feedback) — a deliberately simple fused
model, kept separate from models.FusedModel (which is left intact).

Differences from FusedModel, per the architecture feedback:
- Each branch is max-pooled over time INDEPENDENTLY, then the two pooled vectors
  are concatenated and classified. There is no per-timestep fusion GRU, so the
  trajectory and video frames need not be temporally aligned (and no
  normalized-time positional-encoding feature is needed → traj input_dim=3).
- Vision is RGB-only (in_channels=3): the subject/pedestrian box-mask channels
  are a pixel-space copy of the trajectory, i.e. redundant with the traj branch.

Note this is still end-to-end joint training (vision gradients reach the traj
encoder through the shared classifier); the gated/detached cross-attention design
that protects the traj branch is deferred — see the fusion discussion.
"""
import torch
import torch.nn as nn

from .trajectory_encoder import TrajectoryEncoder
from .vision_encoder import VisionEncoder
from .classifier import _encode_peds


class PooledFusedModel(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, num_classes=2,
                 vision_backbone_dim=512, freeze_vision=True, top_k=5, num_frames=32):
        super().__init__()
        self.top_k      = top_k
        self.num_frames = num_frames

        # input_dim=3: (x, y, speed) only — pooling-before-fusion needs no
        # traj↔frame alignment, so no normalized-time positional encoding.
        self.vehicle_encoder = TrajectoryEncoder(input_dim=3, hidden_dim=hidden_dim)
        self.ped_encoder     = TrajectoryEncoder(input_dim=3, hidden_dim=hidden_dim)
        self.traj_cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # RGB only — mask channels dropped as redundant with the trajectory branch.
        self.vision_encoder = VisionEncoder(output_dim=vision_backbone_dim, freeze_early=freeze_vision, in_channels=3)
        self.vision_proj    = nn.Linear(vision_backbone_dim, hidden_dim)

        # concat of the two independently-pooled branch vectors → classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes),
        )

        # Debug-only branch ablation: None | 'no_vision' | 'no_traj'.
        self.ablate = None

    def forward(self, vehicle_feat, ped_feat, frames, v_padding_mask=None, p_padding_mask=None):
        # the vision dataset path appends a normalized-time column (feat_dim=4)
        # and mask channels (C=5); this model uses neither.
        vehicle_feat = vehicle_feat[..., :3]
        ped_feat     = ped_feat[..., :3]

        vehicle_enc = self.vehicle_encoder(vehicle_feat)
        ped_enc, ped_key_mask = _encode_peds(
            self.ped_encoder, ped_feat, p_padding_mask, self.top_k, self.num_frames, feat_dim=3
        )
        traj_context, _ = self.traj_cross_attn(
            query=vehicle_enc, key=ped_enc, value=ped_enc, key_padding_mask=ped_key_mask,
        )
        traj_context = traj_context + vehicle_enc
        if v_padding_mask is not None:
            traj_context = traj_context.masked_fill(v_padding_mask.unsqueeze(-1), float('-inf'))
        traj_pooled = torch.nan_to_num(traj_context.max(dim=1).values, neginf=0.0)  # (B, H)

        vis = self.vision_proj(self.vision_encoder(frames[:, :, :3]))  # RGB only → (B, F, H)
        vis_pooled = vis.max(dim=1).values                            # (B, H)

        if self.ablate == 'no_vision':
            vis_pooled = torch.zeros_like(vis_pooled)
        elif self.ablate == 'no_traj':
            traj_pooled = torch.zeros_like(traj_pooled)

        pooled = torch.cat([traj_pooled, vis_pooled], dim=-1)  # (B, 2H)
        return self.classifier(pooled)
