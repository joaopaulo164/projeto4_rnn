"""Unit tests for neural network architectures, pooling modules, and exact parameter counts."""

import pytest
import torch

from project4_rnn.config import ExperimentConfig, ModelConfig
from project4_rnn.models.factory import build_model
from project4_rnn.models.mlp import MLP32
from project4_rnn.models.pooling import TemporalPooling
from project4_rnn.models.recurrent import RecurrentClassifier


def test_mlp32_architecture_and_parameter_count() -> None:
    """Verify E01 MLP-32 has exactly 3,335 parameters for ElectricDevices (input 96, 7 classes)."""
    model = MLP32(input_size=96, hidden_size=32, num_classes=7)
    assert model.count_parameters() == 3335
    assert model.count_trainable_parameters() == 3335

    x = torch.randn(8, 96)
    out = model(x)
    assert out.shape == (8, 7)


def test_temporal_pooling_strategies() -> None:
    """Verify temporal pooling output tensor shapes and multipliers."""
    batch_size, seq_len, hidden_size = 4, 96, 32
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
        ("gru", 32, "last", 3591),
        ("gru", 32, "mean", 3591),
        ("gru", 32, "max", 3591),
        ("gru", 32, "mean_max", 3815),
        ("lstm", 32, "last", 4711),
        ("lstm", 32, "mean", 4711),
        ("lstm", 32, "max", 4711),
        ("lstm", 32, "mean_max", 4935),
        ("gru", 64, "last", 13319),
        ("gru", 64, "mean", 13319),
        ("gru", 64, "max", 13319),
        ("gru", 64, "mean_max", 13767),
        ("lstm", 64, "last", 17607),
        ("lstm", 64, "mean", 17607),
        ("lstm", 64, "max", 17607),
        ("lstm", 64, "mean_max", 18055),
    ],
)
def test_recurrent_architectures_exact_parameter_counts(
    cell_type: str,
    hidden_size: int,
    pooling: str,
    expected_params: int,
) -> None:
    """Verify that all recurrent configurations match theoretical parameter counts strictly for 7 classes."""
    model = RecurrentClassifier(
        cell_type=cell_type,
        input_size=1,
        hidden_size=hidden_size,
        num_layers=1,
        bidirectional=False,
        batch_first=True,
        pooling=pooling,
        num_classes=7,
    )
    assert model.count_parameters() == expected_params
    assert model.count_trainable_parameters() == expected_params

    # Test forward pass with (batch, 96, 1) and (batch, 96)
    x3d = torch.randn(4, 96, 1)
    out3d = model(x3d)
    assert out3d.shape == (4, 7)

    x2d = torch.randn(4, 96)
    out2d = model(x2d)
    assert out2d.shape == (4, 7)


def test_build_model_factory_validation() -> None:
    """Verify build_model builds valid models and detects mismatch."""
    m_cfg = ModelConfig(
        architecture="mlp",
        input_size=96,
        hidden_size=32,
        num_classes=7,
    )
    exp_cfg = ExperimentConfig(
        experiment_id="E01_MLP_32",
        model=m_cfg,
    )
    model = build_model(exp_cfg, validate_params=True)
    assert model.count_parameters() == 3335
