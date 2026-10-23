"""Recurrent neural network models (GRU and LSTM) with configurable pooling strategies."""

import torch
import torch.nn as nn

from project4_rnn.models.base import BaseTimeSeriesModel
from project4_rnn.models.pooling import TemporalPooling


class RecurrentClassifier(BaseTimeSeriesModel):
    """Recurrent neural network classifier supporting GRU and LSTM with temporal pooling.

    Configurations:
        - GRU-32 (last/mean/max): 3,426 parameters
        - GRU-32 (mean_max): 3,490 parameters
        - LSTM-32 (last/mean/max): 4,546 parameters
        - LSTM-32 (mean_max): 4,610 parameters
        - GRU-64 (last/mean/max): 12,994 parameters
        - GRU-64 (mean_max): 13,122 parameters
        - LSTM-64 (last/mean/max): 17,282 parameters
        - LSTM-64 (mean_max): 17,410 parameters
    """

    def __init__(
        self,
        cell_type: str = "gru",
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 1,
        bidirectional: bool = False,
        batch_first: bool = True,
        pooling: str = "last",
        num_classes: int = 2,
    ) -> None:
        """Initialize recurrent network.

        Args:
            cell_type: 'gru' or 'lstm'.
            input_size: Number of input features per time step (default 1).
            hidden_size: Hidden dimension (e.g. 32 or 64).
            num_layers: Number of recurrent layers (default 1).
            bidirectional: Recurrence directionality (default False).
            batch_first: If True, input is (batch, seq, feature) (default True).
            pooling: Temporal pooling strategy ('last', 'mean', 'max', 'mean_max').
            num_classes: Output logits dimension (default 2).
        """
        super().__init__()
        self.cell_type = cell_type.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.pooling_strategy = pooling
        self.num_classes = num_classes

        if self.cell_type == "gru":
            self.rnn: nn.Module = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=batch_first,
                bidirectional=bidirectional,
            )
        elif self.cell_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=batch_first,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(f"Unsupported cell_type: '{cell_type}'. Must be 'gru' or 'lstm'.")

        self.pooling = TemporalPooling(strategy=pooling)

        num_directions = 2 if bidirectional else 1
        classifier_input_dim = hidden_size * num_directions * self.pooling.output_multiplier
        self.classifier = nn.Linear(classifier_input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 140, 1) or (batch, 140).

        Returns:
            Logits of shape (batch, num_classes).
        """
        # Ensure 3D tensor: (batch, seq_len, num_features)
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        # RNN forward pass
        rnn_out, _ = self.rnn(x)  # shape: (batch, seq_len, hidden_size)

        # Temporal pooling
        pooled = self.pooling(rnn_out)  # shape: (batch, hidden_size) or (batch, 2 * hidden_size)

        # Linear classifier
        logits: torch.Tensor = self.classifier(pooled)  # shape: (batch, num_classes)
        return logits
