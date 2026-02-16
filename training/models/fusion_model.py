import torch
import torch.nn as nn
from .vision_encoder import VisionEncoder
from .trajectory_encoder import TrajectoryEncoder


class MultiModalFusionModel(nn.Module):
    def __init__(self, num_classes=2, vision_dim=512, trajectory_dim=128):
        super().__init__()
        self.vision_encoder = VisionEncoder(output_dim=vision_dim)
        self.trajectory_encoder = TrajectoryEncoder(output_dim=trajectory_dim)
        
        fusion_dim = vision_dim + trajectory_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, frames, trajectory):
        vision_features = self.vision_encoder(frames)
        trajectory_features = self.trajectory_encoder(trajectory)
        
        fused_features = torch.cat([vision_features, trajectory_features], dim=1)
        
        logits = self.classifier(fused_features)
        
        return logits
