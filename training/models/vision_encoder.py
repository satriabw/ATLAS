import torch
import torch.nn as nn
from torchvision.models import resnet18


class VisionEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        resnet = resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_dim = output_dim
        
    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.shape
        
        x = x.view(batch_size * num_frames, channels, height, width)
        
        features = self.features(x)
        pooled = self.pool(features)
        
        features_flat = pooled.view(batch_size, num_frames, -1)
        
        temporal_features = torch.mean(features_flat, dim=1)
        
        return temporal_features
