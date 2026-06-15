import logging

import torch
import torch.nn as nn
from .trajectory_encoder import TrajectoryEncoder
from .vision_encoder import VisionEncoder, VisionEncoder3D

logger = logging.getLogger(__name__)


def _encode_peds(encoder, ped_feat, p_padding_mask, top_k, num_frames, feat_dim=3):
    B = ped_feat.shape[0]
    assert ped_feat.shape[1] == top_k * num_frames, (
        f"ped_feat dim 1 is {ped_feat.shape[1]}, expected top_k*num_frames="
        f"{top_k}*{num_frames}={top_k * num_frames}. "
        "Likely a mismatch between dataset and model hyperparameters."
    )
    assert ped_feat.shape[2] == feat_dim, f"ped_feat dim 2 is {ped_feat.shape[2]}, expected {feat_dim}"
    if p_padding_mask is not None:
        assert p_padding_mask.shape == (B, top_k * num_frames), (
            f"p_padding_mask shape {p_padding_mask.shape} != expected ({B}, {top_k * num_frames})"
        )
    ped_flat = ped_feat.view(B * top_k, num_frames, -1)
    ped_enc_flat = encoder(ped_flat)

    if p_padding_mask is not None:
        p_mask_flat = p_padding_mask.view(B * top_k, num_frames)
        ped_enc_flat = ped_enc_flat.masked_fill(p_mask_flat.unsqueeze(-1), float('-inf'))

    ped_enc = torch.nan_to_num(ped_enc_flat.max(dim=1).values, neginf=0.0).view(B, top_k, -1)

    ped_key_mask = (
        p_padding_mask.view(B, top_k, num_frames).all(dim=-1)
        if p_padding_mask is not None else None
    )
    return ped_enc, ped_key_mask


class CrossAttentionModel(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, num_classes=2, top_k=5, num_frames=32):
        super().__init__()
        self.top_k      = top_k
        self.num_frames = num_frames
        self.vehicle_encoder = TrajectoryEncoder(input_dim=3, hidden_dim=hidden_dim)
        self.ped_encoder     = TrajectoryEncoder(input_dim=3, hidden_dim=hidden_dim)
        self.cross_attn  = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.classifier  = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes),
        )

    def forward(self, vehicle_feat, ped_feat, v_padding_mask=None, p_padding_mask=None):
        vehicle_enc = self.vehicle_encoder(vehicle_feat)
        ped_enc, ped_key_mask = _encode_peds(
            self.ped_encoder, ped_feat, p_padding_mask, self.top_k, self.num_frames
        )

        if ped_key_mask is not None and ped_key_mask.all(dim=-1).any():
            logger.warning("Batch contains samples with no valid pedestrian trajectories; attention output will be zeroed")

        attended, _ = self.cross_attn(
            query=vehicle_enc, key=ped_enc, value=ped_enc, key_padding_mask=ped_key_mask,
        )
        attended = attended + vehicle_enc

        if v_padding_mask is not None:
            attended = attended.masked_fill(v_padding_mask.unsqueeze(-1), float('-inf'))

        pooled = torch.nan_to_num(attended.max(dim=1).values, neginf=0.0)
        return self.classifier(pooled)


class FusedModel(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, num_classes=2, vision_backbone_dim=512, freeze_vision=True, top_k=5, num_frames=32):
        super().__init__()
        self.top_k      = top_k
        self.num_frames = num_frames

        # input_dim=4: (x, y, speed) + normalized-time positional encoding, so the
        # fusion GRU can align trajectory slots with the uniform h5 frame grid.
        self.vehicle_encoder = TrajectoryEncoder(input_dim=4, hidden_dim=hidden_dim)
        self.ped_encoder     = TrajectoryEncoder(input_dim=4, hidden_dim=hidden_dim)
        self.traj_cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # in_channels=5: RGB + subject-vehicle / top-K-pedestrian box-mask channels.
        self.vision_encoder = VisionEncoder(output_dim=vision_backbone_dim, freeze_early=freeze_vision, in_channels=5)
        self.vision_proj    = nn.Linear(vision_backbone_dim, hidden_dim)

        # Intermediate fusion: trajectory steps and frames are aligned 32-slot grids
        # (stretch resampling mirrors build_h5's linspace), so this GRU sees appearance
        # evolving in step with motion.
        self.fusion_encoder = TrajectoryEncoder(input_dim=hidden_dim * 2, hidden_dim=hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes),
        )

        # Debug-only branch ablation: None | 'no_vision' | 'no_traj' (zero that fusion input).
        self.ablate = None

    def forward(self, vehicle_feat, ped_feat, frames, v_padding_mask=None, p_padding_mask=None):
        vehicle_enc = self.vehicle_encoder(vehicle_feat)
        ped_enc, ped_key_mask = _encode_peds(
            self.ped_encoder, ped_feat, p_padding_mask, self.top_k, self.num_frames, feat_dim=4
        )

        if ped_key_mask is not None and ped_key_mask.all(dim=-1).any():
            logger.warning("Batch contains samples with no valid pedestrian trajectories; attention output will be zeroed")

        traj_context, _ = self.traj_cross_attn(
            query=vehicle_enc, key=ped_enc, value=ped_enc, key_padding_mask=ped_key_mask,
        )
        traj_context = traj_context + vehicle_enc

        vis = self.vision_proj(self.vision_encoder(frames))        # (B, F, hidden)

        if self.ablate == 'no_vision':
            vis = torch.zeros_like(vis)
        elif self.ablate == 'no_traj':
            traj_context = torch.zeros_like(traj_context)

        fused = self.fusion_encoder(torch.cat([traj_context, vis], dim=-1))  # (B, T, hidden)

        if v_padding_mask is not None:
            fused = fused.masked_fill(v_padding_mask.unsqueeze(-1), float('-inf'))

        pooled = torch.nan_to_num(fused.max(dim=1).values, neginf=0.0)
        return self.classifier(pooled)


class VisionOnlyModel(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=2, vision_backbone_dim=512, freeze_vision=True, num_frames=32, backbone='resnet18'):
        super().__init__()
        self.num_frames = num_frames

        if backbone == 'r2plus1d':
            # Temporal anchor: RGB-only, ignores the mask channels.
            self.vision_encoder = VisionEncoder3D(output_dim=vision_backbone_dim, freeze_early=freeze_vision)
        else:
            # in_channels=5: RGB + subject-vehicle / top-K-pedestrian box-mask channels.
            self.vision_encoder = VisionEncoder(output_dim=vision_backbone_dim, freeze_early=freeze_vision, in_channels=5)
        self.vision_proj = nn.Linear(vision_backbone_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes),
        )

    def forward(self, frames):
        vis    = self.vision_proj(self.vision_encoder(frames))  # (B, F, hidden_dim)
        pooled = vis.max(dim=1).values
        return self.classifier(pooled)
