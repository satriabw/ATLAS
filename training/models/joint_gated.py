"""Joint gated multimodal model (2026-07-10, plan =
artifacts/docs/2026-07-10_joint_gated/plan.md).

The validated GatedFusionModel head with the r2plus1d backbone IN the graph:
vision is the pooled pre-proj 512-d clip vector (same pooling as
scripts/precompute_r2plus1d_feats._extract), computed live with gradients so
train events reach the backbone only through the shared loss — no frozen-feature
train/test leakage asymmetry.

Traj modules keep the shared core names (vehicle_encoder / ped_encoder /
cross_attn) so _load_traj_core works unchanged; the vision_encoder subtree
matches VisionOnlyModel's, so the fine-tuned rebuild checkpoint loads directly.
ablate='no_vision' skips the backbone forward entirely (used for the per-epoch
traj-stream preservation probe).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .trajectory_encoder import TrajectoryEncoder
from .classifier import _encode_peds
from .vision_encoder import VisionEncoder3D


class JointGatedFusionModel(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, num_classes=2, top_k=5,
                 num_frames=32, vis_dim=512, gate=True, freeze_vision='early'):
        super().__init__()
        self.top_k = top_k
        self.num_frames = num_frames
        self.gate = gate
        self.vehicle_encoder = TrajectoryEncoder(input_dim=3, hidden_dim=hidden_dim)
        self.ped_encoder     = TrajectoryEncoder(input_dim=3, hidden_dim=hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.vision_encoder = VisionEncoder3D(output_dim=vis_dim, freeze_early=freeze_vision)
        self.vis_adapter = nn.Sequential(nn.Linear(vis_dim, hidden_dim), nn.ReLU())
        self.proj_traj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.proj_vis  = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.gate_fc   = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, num_classes))
        self.ablate = None

    def _clip_feat(self, frames):
        # mirrors precompute_r2plus1d_feats._extract, with gradients
        enc = self.vision_encoder
        x = frames[:, :, :3].permute(0, 2, 1, 3, 4)
        x = x * enc._renorm_a + enc._renorm_b
        B, C, T, H, W = x.shape
        if H != 112 or W != 112:
            x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
            x = x.reshape(B, T, C, 112, 112).permute(0, 2, 1, 3, 4)
        return enc.features(x).mean(dim=(2, 3, 4))                           # (B,512)

    def forward(self, vehicle_feat, ped_feat, frames, v_padding_mask=None, p_padding_mask=None):
        # the vision dataset path appends a time-PE traj column; this model uses (x,y,speed)
        vehicle_feat = vehicle_feat[..., :3]
        ped_feat     = ped_feat[..., :3]

        vehicle_enc = self.vehicle_encoder(vehicle_feat)                     # (B,T,H)
        ped_enc, ped_key_mask = _encode_peds(
            self.ped_encoder, ped_feat, p_padding_mask, self.top_k, self.num_frames)
        attended, _ = self.cross_attn(vehicle_enc, ped_enc, ped_enc, key_padding_mask=ped_key_mask)
        attended = attended + vehicle_enc
        if v_padding_mask is not None:
            attended = attended.masked_fill(v_padding_mask.unsqueeze(-1), float('-inf'))
        f_traj = torch.nan_to_num(attended.max(dim=1).values, neginf=0.0)    # (B,H)

        if self.ablate == 'no_vision':
            f_vis = torch.zeros_like(f_traj)
        else:
            f_vis = self.vis_adapter(self._clip_feat(frames))                # (B,H)
        if self.ablate == 'no_traj':
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
