import torch
import torch.nn as nn
from .trajectory_encoder import TrajectoryEncoder
from .vision_encoder import VisionEncoder


class CrossAttentionModel(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, num_classes: int = 2):
        super().__init__()
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
        vehicle_feat: torch.Tensor,
        ped_feat: torch.Tensor,
        v_padding_mask: torch.Tensor = None,  # (B, T_v)  True = padded
        p_padding_mask: torch.Tensor = None,  # (B, T_p)  True = padded
    ) -> torch.Tensor:
        vehicle_enc = self.vehicle_encoder(vehicle_feat)  # (B, T_v, 128)
        ped_enc     = self.ped_encoder(ped_feat)          # (B, T_p, 128)

        # Cross-attention: vehicle queries pedestrian sequence.
        # key_padding_mask marks padded ped positions so attention ignores them.
        attended, _ = self.cross_attn(
            query=vehicle_enc, key=ped_enc, value=ped_enc,
            key_padding_mask=p_padding_mask,
        )

        # Mask padded vehicle positions before max-pool so they don't win.
        if v_padding_mask is not None:
            attended = attended.masked_fill(v_padding_mask.unsqueeze(-1), float('-inf'))

        pooled = attended.max(dim=1).values  # (B, 128)
        return self.classifier(pooled)


class FusedModel(nn.Module):
    """Vision-queries-trajectory cross-attention fusion.

    Step 1 — trajectory interaction:
        vehicle_enc  (B, T_v, H)  queries  ped_enc  (B, T_p, H)
        → traj_context  (B, T_v, H)   [vehicle-ped interaction features]

    Step 2 — vision queries trajectory:
        vis_proj  (B, F, H)  queries  traj_context  (B, T_v, H)
        → fused  (B, F, H)             [vision grounded in trajectory]

    Step 3 — pool + classify:
        mean-pool over F frames  →  (B, H)  →  MLP  →  logits
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_classes: int = 2,
        vision_backbone_dim: int = 512,
        freeze_vision: bool = False,
    ):
        super().__init__()
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
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        vehicle_feat: torch.Tensor,         # (B, T_v, 3)
        ped_feat: torch.Tensor,             # (B, T_p, 3)
        frames: torch.Tensor,               # (B, num_frames, C, H, W)
        v_padding_mask: torch.Tensor = None,
        p_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Step 1: vehicle-ped trajectory interaction
        vehicle_enc  = self.vehicle_encoder(vehicle_feat)   # (B, T_v, H)
        ped_enc      = self.ped_encoder(ped_feat)            # (B, T_p, H)
        traj_context, _ = self.traj_cross_attn(
            query=vehicle_enc, key=ped_enc, value=ped_enc,
            key_padding_mask=p_padding_mask,
        )                                                    # (B, T_v, H)

        # Step 2: vision queries trajectory context
        vis_frames = self.vision_encoder(frames)             # (B, F, backbone_dim)
        vis_proj   = self.vision_proj(vis_frames)            # (B, F, H)
        fused, _ = self.fusion_cross_attn(
            query=vis_proj, key=traj_context, value=traj_context,
            key_padding_mask=v_padding_mask,                 # ignore padded traj steps
        )                                                    # (B, F, H)

        # Step 3: pool + classify
        # Direct trajectory path (residual) ensures trajectory signal reaches the
        # classifier even when the vision branch is uninformative.
        traj_pooled  = traj_context.max(dim=1).values   # (B, H)
        fused_pooled = fused.max(dim=1).values           # (B, H)
        pooled = traj_pooled + fused_pooled              # (B, H)
        return self.classifier(pooled)
