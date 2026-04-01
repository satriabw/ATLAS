import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


_RESNET18_FEAT_DIM = 512


class VisionEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(_RESNET18_FEAT_DIM, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        """
        Args:
            x: (B, num_frames, C, H, W)
        Returns:
            (B, num_frames, output_dim)  — per-frame features, no temporal pooling
        """
        batch_size, num_frames, channels, height, width = x.shape
        x = x.view(batch_size * num_frames, channels, height, width)
        features = self.features(x)
        pooled = self.pool(features).view(batch_size * num_frames, _RESNET18_FEAT_DIM)
        return self.proj(pooled).view(batch_size, num_frames, self.output_dim)
