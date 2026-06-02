import torch
import torch.nn as nn


class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim: int = 3, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.output_dim = hidden_dim

    def forward(self, x):
        x = torch.relu(self.embedding(x))
        output, _ = self.gru(x)
        return output
