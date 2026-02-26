import torch.nn as nn
from .trajectory_encoder import TrajectoryEncoder


class TrajectoryOnlyModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.trajectory_encoder = TrajectoryEncoder()

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, trajectory):
        features = self.trajectory_encoder(trajectory)
        return self.classifier(features)
