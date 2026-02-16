import torch
import torch.nn as nn


class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.output_dim = output_dim
        
    def forward(self, x):
        batch_size, num_timesteps, feature_dim = x.shape
        
        x = x.view(batch_size * num_timesteps, feature_dim)
        
        features = self.encoder(x)
        
        features = features.view(batch_size, num_timesteps, -1)
        
        temporal_features = torch.mean(features, dim=1)
        
        return temporal_features
