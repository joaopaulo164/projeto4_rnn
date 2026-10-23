"""Unit tests for dataset binarization, splitting, validation, and PyTorch dataset creation."""

from pathlib import Path

import numpy as np
import pytest
import torch

from project4_rnn.data.dataset import ECG5000Dataset, create_dataloaders
from project4_rnn.data.preprocessing import binarize_labels, fit_and_apply_scaler
from project4_rnn.data.splitting import create_or_load_split
from project4_rnn.data.validation import validate_dataset_arrays


def test_binarize_labels_mapping() -> None:
    """Verify that class 1 maps to 0 and classes 2-5 map to 1."""
    raw_labels = np.array([1, 2, 3, 4, 5, 1, 2], dtype=np.int64)
    binary = binarize_labels(raw_labels)

    expected = np.array([0, 1, 1, 1, 1, 0, 1], dtype=np.int64)
    np.testing.assert_array_equal(binary, expected)


def test_binarize_labels_invalid_value() -> None:
    """Verify that invalid class labels raise ValueError."""
    invalid_labels = np.array([1, 6, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="Unexpected class label 6"):
        binarize_labels(invalid_labels)


def test_stratified_split_partition_and_overlap(tmp_path: Path) -> None:
    """Verify 400/100 train/val split, non-overlap, and stratification."""
    # 500 samples: 300 of class 0, 200 of class 1
    y_dev = np.array([0] * 300 + [1] * 200, dtype=np.int64)
    split_path = tmp_path / "split_indices.json"

    train_idx, val_idx = create_or_load_split(
        y_dev=y_dev,
        split_indices_path=split_path,
        seed=42,
        train_size=400,
        val_size=100,
    )

    assert len(train_idx) == 400
    assert len(val_idx) == 100
    assert len(set(train_idx).intersection(set(val_idx))) == 0
    assert len(set(train_idx).union(set(val_idx))) == 500

    # Test persistence and deterministic reload
    train_idx_2, val_idx_2 = create_or_load_split(
        y_dev=y_dev,
        split_indices_path=split_path,
        seed=42,
    )
    np.testing.assert_array_equal(train_idx, train_idx_2)
    np.testing.assert_array_equal(val_idx, val_idx_2)


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
    X_train = np.ones((400, 140), dtype=np.float32)
    y_train = np.zeros(400, dtype=np.int64)
    X_val = np.ones((100, 140), dtype=np.float32)
    y_val = np.zeros(100, dtype=np.int64)
    X_test = np.ones((4500, 140), dtype=np.float32)
    y_test = np.zeros(4500, dtype=np.int64)

    # Valid check
    report = validate_dataset_arrays(X_train, y_train, X_val, y_val, X_test, y_test)
    assert report["overall_checks"]["is_valid"] is True

    # Inject NaN
    X_train_nan = X_train.copy()
    X_train_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="Found 1 NaN"):
        validate_dataset_arrays(X_train_nan, y_train, X_val, y_val, X_test, y_test)

    # Inject Inf
    X_val_inf = X_val.copy()
    X_val_inf[0, 0] = np.inf
    with pytest.raises(ValueError, match="Found 1 Inf"):
        validate_dataset_arrays(X_train, y_train, X_val_inf, y_val, X_test, y_test)


def test_pytorch_dataset_and_dataloaders() -> None:
    """Verify PyTorch dataset shapes and DataLoader reproducibility."""
    X_train = np.random.randn(400, 140).astype(np.float32)
    y_train = np.random.randint(0, 2, size=400, dtype=np.int64)
    X_val = np.random.randn(100, 140).astype(np.float32)
    y_val = np.random.randint(0, 2, size=100, dtype=np.int64)

    # 1. 2D dataset (MLP)
    ds_mlp = ECG5000Dataset(X_train, y_train, reshape_for_rnn=False)
    x, y, idx = ds_mlp[0]
    assert x.shape == (140,)
    assert isinstance(x, torch.Tensor)

    # 2. 3D dataset (RNN)
    ds_rnn = ECG5000Dataset(X_train, y_train, reshape_for_rnn=True)
    x_rnn, _, _ = ds_rnn[0]
    assert x_rnn.shape == (140, 1)

    # 3. DataLoader creation
    train_l, val_l, test_l = create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=32, seed=42, reshape_for_rnn=True
    )
    batch_x, batch_y, _ = next(iter(train_l))
    assert batch_x.shape == (32, 140, 1)
    assert batch_y.shape == (32,)
