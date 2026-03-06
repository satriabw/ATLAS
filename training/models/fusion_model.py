import torch
import torch.nn as nn
from .trajectory_encoder import TrajectoryEncoder


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

    def forward(self, vehicle_feat: torch.Tensor, ped_feat: torch.Tensor) -> torch.Tensor:
        vehicle_enc = self.vehicle_encoder(vehicle_feat)  # (B, T, 128)
        ped_enc = self.ped_encoder(ped_feat)              # (B, T, 128)
        attended, _ = self.cross_attn(query=vehicle_enc, key=ped_enc, value=ped_enc)
        pooled = attended.max(dim=1).values               # (B, 128)
        return self.classifier(pooled)
