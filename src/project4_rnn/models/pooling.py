"""Temporal aggregation and pooling strategies for recurrent sequence representations."""

from typing import Literal

import torch
import torch.nn as nn

PoolingType = Literal["last", "mean", "max", "mean_max"]


class TemporalPooling(nn.Module):
    """Temporal pooling module supporting last state, mean, max, and concatenated mean+max."""

    def __init__(self, strategy: str) -> None:
        """Initialize temporal pooling strategy.

        Args:
            strategy: One of 'last', 'mean', 'max', 'mean_max'.

        Raises:
            ValueError: If strategy is unsupported.
        """
        super().__init__()
        valid_strategies = ("last", "mean", "max", "mean_max")
        if strategy not in valid_strategies:
            raise ValueError(
                f"Unknown pooling strategy '{strategy}'. Valid options: {valid_strategies}"
            )
        self.strategy = strategy

    @property
    def output_multiplier(self) -> int:
        """Dimension multiplier for the resulting representation vector.

        Returns:
            2 if mean_max (concatenated), 1 for last, mean, max.
        """
        if self.strategy == "mean_max":
            return 2
        return 1

    def forward(self, recurrent_output: torch.Tensor) -> torch.Tensor:
        """Aggregate temporal dimension from recurrent output.

        Args:
            recurrent_output: Tensor of shape (batch, seq_len, hidden_size).

        Returns:
            Aggregated tensor of shape (batch, hidden_size) or (batch, 2 * hidden_size).
        """
        if self.strategy == "last":
            # E02: Last hidden state representation
            return recurrent_output[:, -1, :]

        elif self.strategy == "mean":
            # E03: Mean temporal pooling
            return recurrent_output.mean(dim=1)

        elif self.strategy == "max":
            # E04: Max temporal pooling
            return recurrent_output.max(dim=1).values

        elif self.strategy == "mean_max":
            # E05: Concatenated mean and max pooling
            mean_representation = recurrent_output.mean(dim=1)
            max_representation = recurrent_output.max(dim=1).values
            return torch.cat(
                tensors=(
                    mean_representation,
                    max_representation,
                ),
                dim=1,
            )

        raise RuntimeError(f"Unhandled pooling strategy: {self.strategy}")
