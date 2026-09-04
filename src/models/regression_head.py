"""Regression head used by E03-E06."""

from torch import nn


class RegressionHead(nn.Module):
    """
    Linear -> GELU -> Dropout -> Linear(1)

    Mặc định hidden layer = 128 units theo MODEL.md.
    """

    def __init__(self, input_size, hidden_size=128, dropout=0.1):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)
