import torch.nn as nn


class VideoMAEROIHead(nn.Module):
    """Shallow classifier over frozen VideoMAE ROI features.

    Input = concatenated pooled features for the TOP and BOT crosswalk crops
    (2 x 768 = 1536). Head shape mirrors the classifiers in classifier.py.
    """

    def __init__(self, in_dim=1536, hidden=128, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, feats):
        return self.net(feats)
