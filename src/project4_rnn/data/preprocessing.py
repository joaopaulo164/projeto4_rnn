"""Preprocessing, binary label transformation, and data leakage-free scaling."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from project4_rnn.constants import (
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    ORIGINAL_NEGATIVE_CLASSES,
    ORIGINAL_POSITIVE_CLASSES,
)
from project4_rnn.reproducibility import compute_object_hash

logger = logging.getLogger("project4_rnn.data.preprocessing")


def binarize_labels(
    labels: np.ndarray,
    negative_classes: tuple[int, ...] = ORIGINAL_NEGATIVE_CLASSES,
    positive_classes: tuple[int, ...] = ORIGINAL_POSITIVE_CLASSES,
) -> np.ndarray:
    """Map original multiclass labels (1 to 5) to binary labels (0 and 1).

    Class 1 -> Binary 0 (Negative class)
    Classes 2, 3, 4, 5 -> Binary 1 (Positive class)

    Args:
        labels: Original integer labels array.
        negative_classes: Tuple of class integers to map to 0.
        positive_classes: Tuple of class integers to map to 1.

    Returns:
        Binary int64 array of shape (N,) containing 0 and 1.

    Raises:
        ValueError: If labels contain values outside expected classes.
    """
    binary_labels = np.zeros(len(labels), dtype=np.int64)

    for i, label in enumerate(labels):
        val = int(label)
        if val in negative_classes:
            binary_labels[i] = LABEL_NEGATIVE
        elif val in positive_classes:
            binary_labels[i] = LABEL_POSITIVE
        else:
            raise ValueError(
                f"Unexpected class label {val} at index {i}. "
                f"Expected negative in {negative_classes}, positive in {positive_classes}"
            )

    return binary_labels


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
