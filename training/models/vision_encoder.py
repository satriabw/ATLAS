import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


_RESNET18_FEAT_DIM = 512


class VisionEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features   = nn.Sequential(*list(resnet.children())[:-2])
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.proj       = nn.Linear(_RESNET18_FEAT_DIM, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        B, F, C, H, W = x.shape
        x       = x.view(B * F, C, H, W)
        pooled  = self.pool(self.features(x)).view(B * F, _RESNET18_FEAT_DIM)
        return self.proj(pooled).view(B, F, self.output_dim)
