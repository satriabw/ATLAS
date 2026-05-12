import torch
import torch.nn as nn
from .trajectory_encoder import TrajectoryEncoder
from .vision_encoder import VisionEncoder


class CrossAttentionModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_classes: int = 2,
        top_k: int = 5,
        num_frames: int = 32,
    ):
        super().__init__()
        self.top_k = top_k
        self.num_frames = num_frames
        self.vehicle_encoder = TrajectoryEncoder(input_dim=3, hidden_dim=hidden_dim)
        self.ped_encoder = TrajectoryEncoder(input_dim=3, hidden_dim=hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        vehicle_feat: torch.Tensor,           # (B, T_v, 3)
        ped_feat: torch.Tensor,               # (B, top_k * num_frames, 3)
        v_padding_mask: torch.Tensor = None,  # (B, T_v)  True = padded
        p_padding_mask: torch.Tensor = None,  # (B, top_k * num_frames)  True = padded
    ) -> torch.Tensor:
        B = vehicle_feat.shape[0]

        vehicle_enc = self.vehicle_encoder(vehicle_feat)  # (B, T_v, H)

        # Process each pedestrian trajectory independently through the encoder.
        # Reshaping to (B*K, num_frames, 3) ensures the GRU hidden state is reset
        # between pedestrians and does not bleed across pedestrian boundaries.
        ped_flat = ped_feat.view(B * self.top_k, self.num_frames, -1)   # (B*K, T_p, 3)
        ped_enc_flat = self.ped_encoder(ped_flat)                        # (B*K, T_p, H)

        if p_padding_mask is not None:
            p_mask_flat = p_padding_mask.view(B * self.top_k, self.num_frames)  # (B*K, T_p)
            ped_enc_flat = ped_enc_flat.masked_fill(p_mask_flat.unsqueeze(-1), float('-inf'))

        # Pool each pedestrian to a single context vector.
        ped_enc_pooled = ped_enc_flat.max(dim=1).values          # (B*K, H)
        # Dummy (all-padded) pedestrians produce -inf; replace with zeros so they
        # carry no information when used as attention keys.
        ped_enc_pooled = torch.nan_to_num(ped_enc_pooled, neginf=0.0)
        ped_enc = ped_enc_pooled.view(B, self.top_k, -1)         # (B, K, H)

        # Per-pedestrian key padding mask: True if the entire track is dummy/padded.
        if p_padding_mask is not None:
            ped_key_mask = (
                p_padding_mask.view(B, self.top_k, self.num_frames).all(dim=-1)
            )  # (B, K)
        else:
            ped_key_mask = None

        # Cross-attention: vehicle queries pedestrian encodings.
        attended, _ = self.cross_attn(
            query=vehicle_enc, key=ped_enc, value=ped_enc,
            key_padding_mask=ped_key_mask,
        )  # (B, T_v, H)
        attended = attended + vehicle_enc  # residual: preserve vehicle kinematics

        # Mask padded vehicle positions before max-pool so they don't win.
        if v_padding_mask is not None:
            attended = attended.masked_fill(v_padding_mask.unsqueeze(-1), float('-inf'))

        pooled = attended.max(dim=1).values  # (B, H)
        pooled = torch.nan_to_num(pooled, neginf=0.0)
        return self.classifier(pooled)


class FusedModel(nn.Module):
    """Vision-queries-trajectory cross-attention fusion.

    Step 1 — trajectory interaction:
        vehicle_enc  (B, T_v, H)  queries  ped_enc  (B, K, H)
        → traj_context  (B, T_v, H)   [vehicle-ped interaction features]

    Step 2 — vision queries trajectory:
        vis_proj  (B, F, H)  queries  traj_context  (B, T_v, H)
        → fused  (B, F, H)             [vision grounded in trajectory]

    Step 3 — pool + classify:
        max-pool over T_v and F frames  →  (B, H)  →  MLP  →  logits
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_classes: int = 2,
        vision_backbone_dim: int = 512,
        freeze_vision: bool = False,
        top_k: int = 5,
        num_frames: int = 32,
    ):
        super().__init__()
        self.top_k = top_k
        self.num_frames = num_frames

        # Trajectory branch
        self.vehicle_encoder  = TrajectoryEncoder(input_dim=3, hidden_dim=hidden_dim)
        self.ped_encoder      = TrajectoryEncoder(input_dim=3, hidden_dim=hidden_dim)
        self.traj_cross_attn  = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True,
        )

        # Vision branch
        self.vision_encoder = VisionEncoder(output_dim=vision_backbone_dim)
        if freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
        self.vision_proj = nn.Linear(vision_backbone_dim, hidden_dim)

        # Vision-queries-trajectory fusion attention
        self.fusion_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        vehicle_feat: torch.Tensor,         # (B, T_v, 3)
        ped_feat: torch.Tensor,             # (B, top_k * num_frames, 3)
        frames: torch.Tensor,               # (B, num_frames, C, H, W)
        v_padding_mask: torch.Tensor = None,
        p_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B = vehicle_feat.shape[0]

        vehicle_enc = self.vehicle_encoder(vehicle_feat)   # (B, T_v, H)

        # Process each pedestrian trajectory independently (same fix as CrossAttentionModel).
        ped_flat = ped_feat.view(B * self.top_k, self.num_frames, -1)   # (B*K, T_p, 3)
        ped_enc_flat = self.ped_encoder(ped_flat)                        # (B*K, T_p, H)

        if p_padding_mask is not None:
            p_mask_flat = p_padding_mask.view(B * self.top_k, self.num_frames)
            ped_enc_flat = ped_enc_flat.masked_fill(p_mask_flat.unsqueeze(-1), float('-inf'))

        ped_enc_pooled = ped_enc_flat.max(dim=1).values          # (B*K, H)
        ped_enc_pooled = torch.nan_to_num(ped_enc_pooled, neginf=0.0)
        ped_enc = ped_enc_pooled.view(B, self.top_k, -1)         # (B, K, H)

        if p_padding_mask is not None:
            ped_key_mask = (
                p_padding_mask.view(B, self.top_k, self.num_frames).all(dim=-1)
            )  # (B, K)
        else:
            ped_key_mask = None

        # Step 1: vehicle-ped trajectory interaction
        traj_context, _ = self.traj_cross_attn(
            query=vehicle_enc, key=ped_enc, value=ped_enc,
            key_padding_mask=ped_key_mask,
        )                                                    # (B, T_v, H)
        traj_context = traj_context + vehicle_enc            # residual: preserve vehicle kinematics

        # Step 2: vision queries trajectory context
        vis_frames = self.vision_encoder(frames)             # (B, F, backbone_dim)
        vis_proj   = self.vision_proj(vis_frames)            # (B, F, H)
        fused, _ = self.fusion_cross_attn(
            query=vis_proj, key=traj_context, value=traj_context,
            key_padding_mask=v_padding_mask,                 # ignore padded traj steps
        )                                                    # (B, F, H)

        # Step 3: pool + classify.
        # Mask padded vehicle timesteps before pooling so they cannot win max-pool.
        if v_padding_mask is not None:
            traj_context_for_pool = traj_context.masked_fill(
                v_padding_mask.unsqueeze(-1), float('-inf')
            )
        else:
            traj_context_for_pool = traj_context
        traj_pooled  = traj_context_for_pool.max(dim=1).values  # (B, H)
        traj_pooled  = torch.nan_to_num(traj_pooled, neginf=0.0)
        fused_pooled = fused.max(dim=1).values                   # (B, H)
        fused_pooled = torch.nan_to_num(fused_pooled, neginf=0.0)
        pooled = torch.cat([traj_pooled, fused_pooled], dim=-1)  # (B, 2H)
        return self.classifier(pooled)
