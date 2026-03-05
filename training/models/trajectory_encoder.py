import torch.nn as nn


class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        # Bidirectional GRU: hidden_size = hidden_dim // 2 so output is hidden_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        self.output_dim = hidden_dim  # (hidden_dim//2) * 2 directions

    def forward(self, x):
        output, _ = self.gru(x)
        return output  # (B, T, hidden_dim)
