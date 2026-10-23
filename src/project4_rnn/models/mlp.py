"""Multilayer Perceptron (MLP) baseline model for ECG5000 classification."""

import torch
import torch.nn as nn

from project4_rnn.models.base import BaseTimeSeriesModel


class MLP32(BaseTimeSeriesModel):
    """E01 Baseline: Multi-Layer Perceptron with 140 -> 32 -> 2 architecture.

    Layers:
        - Linear(140, 32): 140 * 32 + 32 = 4,512 parameters
        - ReLU activation
        - Linear(32, 2): 32 * 2 + 2 = 66 parameters
    Total expected parameters: 4,578.
    """

    def __init__(
        self,
        input_size: int = 140,
        hidden_size: int = 32,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 140) or (batch, 140, 1).

        Returns:
            Logits of shape (batch, num_classes).
        """
        # If input has trailing channel dimension (batch, 140, 1), flatten to (batch, 140)
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        elif x.ndim == 3:
            x = x.flatten(start_dim=1)

        h = self.relu(self.fc1(x))
        logits: torch.Tensor = self.fc2(h)
        return logits
