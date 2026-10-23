"""Unit tests for neural network architectures, pooling modules, and exact parameter counts."""

import pytest
import torch

from project4_rnn.config import ExperimentConfig, ModelConfig
from project4_rnn.models.factory import build_model
from project4_rnn.models.mlp import MLP32
from project4_rnn.models.pooling import TemporalPooling
from project4_rnn.models.recurrent import RecurrentClassifier


def test_mlp32_architecture_and_parameter_count() -> None:
    """Verify E01 MLP-32 has exactly 4,578 parameters."""
    model = MLP32(input_size=140, hidden_size=32, num_classes=2)
    assert model.count_parameters() == 4578
    assert model.count_trainable_parameters() == 4578

    x = torch.randn(8, 140)
    out = model(x)
    assert out.shape == (8, 2)


def test_temporal_pooling_strategies() -> None:
    """Verify temporal pooling output tensor shapes and multipliers."""
    batch_size, seq_len, hidden_size = 4, 140, 32
    rnn_out = torch.randn(batch_size, seq_len, hidden_size)

    # 1. Last state pooling
    p_last = TemporalPooling(strategy="last")
    assert p_last.output_multiplier == 1
    assert p_last(rnn_out).shape == (batch_size, hidden_size)

    # 2. Mean pooling
    p_mean = TemporalPooling(strategy="mean")
    assert p_mean.output_multiplier == 1
    assert p_mean(rnn_out).shape == (batch_size, hidden_size)

    # 3. Max pooling
    p_max = TemporalPooling(strategy="max")
    assert p_max.output_multiplier == 1
    assert p_max(rnn_out).shape == (batch_size, hidden_size)

    # 4. Mean + Max pooling
    p_mean_max = TemporalPooling(strategy="mean_max")
    assert p_mean_max.output_multiplier == 2
    assert p_mean_max(rnn_out).shape == (batch_size, 2 * hidden_size)


@pytest.mark.parametrize(
    "cell_type,hidden_size,pooling,expected_params",
    [
        ("gru", 32, "last", 3426),
        ("gru", 32, "mean", 3426),
        ("gru", 32, "max", 3426),
        ("gru", 32, "mean_max", 3490),
        ("lstm", 32, "last", 4546),
        ("lstm", 32, "mean", 4546),
        ("lstm", 32, "max", 4546),
        ("lstm", 32, "mean_max", 4610),
        ("gru", 64, "last", 12994),
        ("gru", 64, "mean", 12994),
        ("gru", 64, "max", 12994),
        ("gru", 64, "mean_max", 13122),
        ("lstm", 64, "last", 17282),
        ("lstm", 64, "mean", 17282),
        ("lstm", 64, "max", 17282),
        ("lstm", 64, "mean_max", 17410),
    ],
)
def test_recurrent_architectures_exact_parameter_counts(
    cell_type: str,
    hidden_size: int,
    pooling: str,
    expected_params: int,
) -> None:
    """Verify that all recurrent configurations match theoretical parameter counts strictly."""
    model = RecurrentClassifier(
        cell_type=cell_type,
        input_size=1,
        hidden_size=hidden_size,
        num_layers=1,
        bidirectional=False,
        batch_first=True,
        pooling=pooling,
        num_classes=2,
    )
    assert model.count_parameters() == expected_params
    assert model.count_trainable_parameters() == expected_params

    # Test forward pass with (batch, 140, 1) and (batch, 140)
    x3d = torch.randn(4, 140, 1)
    out3d = model(x3d)
    assert out3d.shape == (4, 2)

    x2d = torch.randn(4, 140)
    out2d = model(x2d)
    assert out2d.shape == (4, 2)


def test_build_model_factory_validation() -> None:
    """Verify build_model builds valid models and detects mismatch."""
    m_cfg = ModelConfig(
        architecture="mlp",
        input_size=140,
        hidden_size=32,
        num_classes=2,
    )
    exp_cfg = ExperimentConfig(
        experiment_id="E01_MLP_32",
        model=m_cfg,
    )
    model = build_model(exp_cfg, validate_params=True)
    assert model.count_parameters() == 4578
