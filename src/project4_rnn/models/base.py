"""Abstract base class and parameter inspection utilities for time series models."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseTimeSeriesModel(nn.Module, ABC):
    """Abstract base class for all ElectricDevices multiclass classification models."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor. Shape (batch, 96) for MLP or (batch, 96, 1) for RNN.

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        raise NotImplementedError("Subclasses must implement forward method.")

    def count_parameters(self) -> int:
        """Calculate total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        """Calculate number of trainable (requires_grad=True) parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_breakdown(self) -> dict[str, int]:
        """Return parameter count for each named module/parameter."""
        breakdown: dict[str, int] = {}
        for name, param in self.named_parameters():
            breakdown[name] = param.numel()
        return breakdown

    def get_model_summary(self) -> str:
        """Produce a formatted textual summary of layers and parameter counts."""
        lines = [
            f"Model: {self.__class__.__name__}",
            "=" * 60,
            f"{'Layer / Parameter Name':<40} {'Shape':<12} {'Count':>8}",
            "-" * 60,
        ]
        for name, param in self.named_parameters():
            lines.append(f"{name:<40} {str(list(param.shape)):<12} {param.numel():>8,}")
        lines.append("=" * 60)
        lines.append(f"Total Parameters:     {self.count_parameters():>8,}")
        lines.append(f"Trainable Parameters: {self.count_trainable_parameters():>8,}")
        lines.append("=" * 60)
        return "\n".join(lines)
