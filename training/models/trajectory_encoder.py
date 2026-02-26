import torch.nn as nn


class TrajectoryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=7, hidden_size=128,
                          num_layers=2, batch_first=True, dropout=0.3)
        self.output_dim = 128

    def forward(self, x):
        # x: (B, T, 7)
        _, hidden = self.gru(x)
        # hidden: (num_layers, B, 128) — take last layer
        return hidden[-1]
