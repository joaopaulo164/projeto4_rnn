"""Multilayer Perceptron (MLP) baseline model for ElectricDevices classification."""

import torch
import torch.nn as nn

from project4_rnn.models.base import BaseTimeSeriesModel


class MLP32(BaseTimeSeriesModel):
    """E01 Baseline: Multi-Layer Perceptron with 96 -> 32 -> 7 architecture.

    Layers:
        - Linear(96, 32): 96 * 32 + 32 = 3,104 parameters
        - ReLU activation
        - Linear(32, 7): 32 * 7 + 7 = 231 parameters
    Total expected parameters: 3,335.
    """

    def __init__(
        self,
        input_size: int = 96,
        hidden_size: int = 32,
        num_classes: int = 7,
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
            x: Input tensor of shape (batch, 96) or (batch, 96, 1).

        Returns:
            Logits of shape (batch, 7).
        """
        # If input has trailing channel dimension (batch, 96, 1), flatten to (batch, 96)
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        elif x.ndim == 3:
            x = x.flatten(start_dim=1)

        h = self.relu(self.fc1(x))
        logits: torch.Tensor = self.fc2(h)
        return logits
