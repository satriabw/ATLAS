import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


_RESNET18_FEAT_DIM = 512

# Dataset tensors are ImageNet-normalized; r2plus1d is Kinetics-pretrained.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)
_KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
_KINETICS_STD  = (0.22803, 0.22145, 0.216989)


class VisionEncoder(nn.Module):
    def __init__(self, output_dim=512, freeze_early=True, in_channels=3):
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        if in_channels != 3:
            # Inflate conv1 for extra grounding-mask channels: pretrained RGB
            # filters kept, new channels zero-init (output unchanged at init).
            conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                conv1.weight.zero_()
                conv1.weight[:, :3] = m.conv1.weight
            m.conv1 = conv1
        self.features   = nn.Sequential(*list(m.children())[:-2])   # (B,512,7,7) per frame
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.proj       = nn.Linear(_RESNET18_FEAT_DIM, output_dim)
        self.output_dim = output_dim

        # freeze: False/None = train all; 'early' = freeze conv1..layer2,
        # train layer3/4+proj; 'full' = freeze the whole backbone (head-only /
        # linear probe). With an inflated conv1, conv1 stays trainable in
        # either mode — its mask-channel weights are zero-init and would
        # otherwise never learn.
        freeze = 'early' if freeze_early is True else freeze_early
        if freeze:
            end = 8 if freeze == 'full' else 6
            start = 1 if in_channels != 3 else 0
            self._frozen = list(self.features[start:end])
            for mod in self._frozen:
                for p in mod.parameters():
                    p.requires_grad = False
        else:
            self._frozen = []

    def train(self, mode=True):
        super().train(mode)
        # Keep frozen submodules in eval so their BN running stats stay fixed.
        for mod in self._frozen:
            mod.eval()
        return self

    def forward(self, x):
        # x: (B, F, 3, H, W) — ImageNet-normalized, applied frame-by-frame
        B, F, C, H, W = x.shape
        x      = x.reshape(B * F, C, H, W)
        pooled = self.pool(self.features(x)).flatten(1)   # (B*F, 512)
        return self.proj(pooled).view(B, F, self.output_dim)  # (B, F, output_dim)


class VisionEncoder3D(nn.Module):
    """r2plus1d_18 backbone (Kinetics-pretrained), RGB only — the Phase-0
    temporal anchor. Consumes the same ImageNet-normalized (B,F,C,H,W) tensors
    as VisionEncoder (extra mask channels are ignored), renormalizes to
    Kinetics statistics and downsamples to 112px internally."""

    def __init__(self, output_dim=512, freeze_early=True):
        super().__init__()
        m = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
        self.features   = nn.Sequential(*list(m.children())[:-2])  # (B,512,T/8,7,7)
        self.pool       = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.proj       = nn.Linear(512, output_dim)
        self.output_dim = output_dim

        mean_i = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1, 1)
        std_i  = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1, 1)
        mean_k = torch.tensor(_KINETICS_MEAN).view(1, 3, 1, 1, 1)
        std_k  = torch.tensor(_KINETICS_STD).view(1, 3, 1, 1, 1)
        self.register_buffer('_renorm_a', std_i / std_k)
        self.register_buffer('_renorm_b', (mean_i - mean_k) / std_k)

        # freeze: 'early' = stem+layer1+layer2; 'full' = entire backbone.
        freeze = 'early' if freeze_early is True else freeze_early
        if freeze:
            end = 5 if freeze == 'full' else 3
            self._frozen = list(self.features[:end])
            for mod in self._frozen:
                for p in mod.parameters():
                    p.requires_grad = False
        else:
            self._frozen = []

    def train(self, mode=True):
        super().train(mode)
        for mod in self._frozen:
            mod.eval()
        return self

    def forward(self, x):
        # x: (B, F, C, H, W); use RGB only, to (B, 3, T, 112, 112) Kinetics-normalized
        x = x[:, :, :3].permute(0, 2, 1, 3, 4)
        x = x * self._renorm_a + self._renorm_b
        B, C, T, H, W = x.shape
        if H != 112 or W != 112:
            x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
            x = x.reshape(B, T, C, 112, 112).permute(0, 2, 1, 3, 4)
        feats  = self.features(x)                      # (B, 512, T', 7, 7)
        pooled = self.pool(feats).squeeze(-1).squeeze(-1).transpose(1, 2)  # (B, T', 512)
        return self.proj(pooled)                       # (B, T', output_dim)
