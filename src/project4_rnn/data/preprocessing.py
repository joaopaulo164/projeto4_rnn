"""Preprocessing, label encoding mapping (official <-> internal indices), and data leakage-free scaling."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from project4_rnn.constants import (
    INTERNAL_TO_OFFICIAL_LABEL,
    OFFICIAL_TO_INTERNAL_LABEL,
)
from project4_rnn.reproducibility import compute_object_hash

logger = logging.getLogger("project4_rnn.data.preprocessing")


def map_official_to_internal_labels(
    labels: np.ndarray,
    mapping: dict[int, int] | None = None,
) -> np.ndarray:
    """Map official dataset class labels (1 to 7) to contiguous internal indices (0 to 6).

    Args:
        labels: Array of official integer class labels.
        mapping: Optional dictionary mapping official class int -> internal index int.

    Returns:
        NumPy int64 array of internal indices suitable for PyTorch CrossEntropyLoss.

    Raises:
        ValueError: If an unexpected class label is encountered.
    """
    map_dict = mapping if mapping is not None else OFFICIAL_TO_INTERNAL_LABEL
    mapped = np.zeros(len(labels), dtype=np.int64)

    for i, label in enumerate(labels):
        val = int(label)
        if val in map_dict:
            mapped[i] = map_dict[val]
        else:
            raise ValueError(
                f"Unexpected official class label {val} at index {i}. "
                f"Expected label in {list(map_dict.keys())}"
            )

    return mapped


def map_internal_to_official_labels(
    internal_labels: np.ndarray,
    mapping: dict[int, int] | None = None,
) -> np.ndarray:
    """Map internal 0-indexed predictions (0 to 6) back to official class labels (1 to 7).

    Args:
        internal_labels: Array of internal 0-indexed integer labels.
        mapping: Optional dictionary mapping internal index int -> official class int.

    Returns:
        NumPy int64 array of official class labels.

    Raises:
        ValueError: If an unexpected internal index is encountered.
    """
    map_dict = mapping if mapping is not None else INTERNAL_TO_OFFICIAL_LABEL
    mapped = np.zeros(len(internal_labels), dtype=np.int64)

    for i, label in enumerate(internal_labels):
        val = int(label)
        if val in map_dict:
            mapped[i] = map_dict[val]
        else:
            raise ValueError(
                f"Unexpected internal label index {val} at index {i}. "
                f"Expected index in {list(map_dict.keys())}"
            )

    return mapped


def fit_and_apply_scaler(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    scaler_type: str = "zscore",
    scaler_save_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit normalization strictly on X_train and transform X_train, X_val, X_test.

    Prevents any data leakage from validation or test partitions.

    Args:
        X_train: Training features array of shape (N_train, L).
        X_val: Validation features array of shape (N_val, L).
        X_test: Test features array of shape (N_test, L).
        scaler_type: 'zscore' or 'minmax'.
        scaler_save_path: Optional path to persist scaler parameters JSON.

    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, scaler_stats).
    """
    if scaler_type == "zscore":
        mean = float(np.mean(X_train))
        std = float(np.std(X_train))
        if std == 0.0:
            std = 1.0

        X_train_scaled = ((X_train - mean) / std).astype(np.float32)
        X_val_scaled = ((X_val - mean) / std).astype(np.float32)
        X_test_scaled = ((X_test - mean) / std).astype(np.float32)

        stats = {
            "scaler_type": "zscore",
            "mean": mean,
            "std": std,
            "train_samples_fitted": int(X_train.shape[0]),
        }
    elif scaler_type == "minmax":
        min_val = float(np.min(X_train))
        max_val = float(np.max(X_train))
        denom = (max_val - min_val) if max_val > min_val else 1.0

        X_train_scaled = ((X_train - min_val) / denom).astype(np.float32)
        X_val_scaled = ((X_val - min_val) / denom).astype(np.float32)
        X_test_scaled = ((X_test - min_val) / denom).astype(np.float32)

        stats = {
            "scaler_type": "minmax",
            "min": min_val,
            "max": max_val,
            "train_samples_fitted": int(X_train.shape[0]),
        }
    else:
        raise ValueError(f"Unsupported scaler_type: {scaler_type}")

    stats["config_hash"] = compute_object_hash(stats)

    if scaler_save_path is not None:
        scaler_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_save_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved scaler parameters to {scaler_save_path}")

    return X_train_scaled, X_val_scaled, X_test_scaled, stats
