"""Unit tests for dataset label mapping, splitting, validation, and PyTorch dataset creation."""

from pathlib import Path

import numpy as np
import pytest
import torch

from project4_rnn.data.dataset import ElectricDevicesDataset, create_dataloaders
from project4_rnn.data.preprocessing import (
    fit_and_apply_scaler,
    map_internal_to_official_labels,
    map_official_to_internal_labels,
)
from project4_rnn.data.validation import DatasetContractError, validate_dataset_arrays


def test_label_mapping() -> None:
    """Verify that official labels 1..7 map to internal labels 0..6 and invert correctly."""
    official_labels = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    internal = map_official_to_internal_labels(official_labels)
    expected_internal = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    np.testing.assert_array_equal(internal, expected_internal)

    inverted = map_internal_to_official_labels(internal)
    np.testing.assert_array_equal(inverted, official_labels)


def test_label_mapping_invalid_value() -> None:
    """Verify that invalid class labels raise ValueError."""
    invalid_labels = np.array([0, 1, 8], dtype=np.int64)
    with pytest.raises(ValueError, match="Unexpected official class label"):
        map_official_to_internal_labels(invalid_labels)


def test_normalization_without_leakage(tmp_path: Path) -> None:
    """Verify that scaler is fitted strictly on train partition."""
    X_train = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    X_val = np.array([[100.0, 200.0]], dtype=np.float32)
    X_test = np.array([[500.0, 600.0]], dtype=np.float32)

    scaler_path = tmp_path / "scaler.json"
    X_tr_s, X_val_s, X_test_s, stats = fit_and_apply_scaler(
        X_train, X_val, X_test, scaler_type="zscore", scaler_save_path=scaler_path
    )

    train_mean = float(np.mean(X_train))
    train_std = float(np.std(X_train))

    assert stats["mean"] == train_mean
    assert stats["std"] == train_std
    assert scaler_path.is_file()

    # Manual transform check
    expected_val = (X_val - train_mean) / train_std
    np.testing.assert_allclose(X_val_s, expected_val, rtol=1e-5)


def test_dataset_validation_detects_nan_and_inf() -> None:
    """Verify data validator detects NaN and Inf values."""
    X_train = np.ones((560, 96), dtype=np.float32)
    y_train = np.zeros(560, dtype=np.int64)
    X_val = np.ones((140, 96), dtype=np.float32)
    y_val = np.zeros(140, dtype=np.int64)
    X_test = np.ones((100, 96), dtype=np.float32)
    y_test = np.zeros(100, dtype=np.int64)

    # Valid check
    report = validate_dataset_arrays(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        expected_train_count=560,
        expected_val_count=140,
        expected_test_count=100,
    )
    assert report["overall_checks"]["is_valid"] is True

    # Inject NaN
    X_train_nan = X_train.copy()
    X_train_nan[0, 0] = np.nan
    with pytest.raises(DatasetContractError, match="Found 1 NaN"):
        validate_dataset_arrays(
            X_train_nan,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            expected_train_count=560,
            expected_val_count=140,
            expected_test_count=100,
        )

    # Inject Inf
    X_val_inf = X_val.copy()
    X_val_inf[0, 0] = np.inf
    with pytest.raises(DatasetContractError, match="Found 1 Inf"):
        validate_dataset_arrays(
            X_train,
            y_train,
            X_val_inf,
            y_val,
            X_test,
            y_test,
            expected_train_count=560,
            expected_val_count=140,
            expected_test_count=100,
        )


def test_pytorch_dataset_and_dataloaders() -> None:
    """Verify PyTorch dataset shapes and DataLoader reproducibility."""
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((100, 96)).astype(np.float32)
    y_train = rng.integers(0, 7, size=100, dtype=np.int64)
    X_val = rng.standard_normal((20, 96)).astype(np.float32)
    y_val = rng.integers(0, 7, size=20, dtype=np.int64)

    # 1. 2D dataset (MLP)
    ds_mlp = ElectricDevicesDataset(X_train, y_train, reshape_for_rnn=False)
    x, y, idx = ds_mlp[0]
    assert x.shape == (96,)
    assert isinstance(x, torch.Tensor)

    # 2. 3D dataset (RNN)
    ds_rnn = ElectricDevicesDataset(X_train, y_train, reshape_for_rnn=True)
    x_rnn, _, _ = ds_rnn[0]
    assert x_rnn.shape == (96, 1)

    # 3. DataLoader creation
    train_l, val_l, test_l = create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=32, seed=42, reshape_for_rnn=True
    )
    batch_x, batch_y, _ = next(iter(train_l))
    assert batch_x.shape == (32, 96, 1)
    assert batch_y.shape == (32,)
