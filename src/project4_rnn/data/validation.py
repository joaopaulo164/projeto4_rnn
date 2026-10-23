"""Data quality validation, integrity checks, and dataset manifest generation."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from project4_rnn.constants import (
    SEQUENCE_LENGTH,
)
from project4_rnn.reproducibility import compute_object_hash

logger = logging.getLogger("project4_rnn.data.validation")


def validate_dataset_arrays(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Execute rigorous data quality and consistency checks across all partitions.

    Args:
        X_train: Training time series array (400, 140).
        y_train: Training binary labels array (400,).
        X_val: Validation time series array (100, 140).
        y_val: Validation binary labels array (100,).
        X_test: Test time series array (4500, 140).
        y_test: Test binary labels array (4500,).

    Returns:
        Validation report dictionary with metrics and check statuses.

    Raises:
        ValueError: If any integrity constraint is violated.
    """
    partitions = {
        "train": (X_train, y_train, 400),
        "val": (X_val, y_val, 100),
        "test": (X_test, y_test, 4500),
    }

    report: dict[str, Any] = {"partitions": {}, "overall_checks": {}}

    for name, (X, y, expected_count) in partitions.items():
        if X.shape[0] != expected_count:
            raise ValueError(
                f"Partition '{name}' sample count mismatch: {X.shape[0]} != {expected_count}"
            )
        if X.shape[1] != SEQUENCE_LENGTH:
            raise ValueError(
                f"Partition '{name}' sequence length mismatch: {X.shape[1]} != {SEQUENCE_LENGTH}"
            )
        if len(y) != expected_count:
            raise ValueError(
                f"Partition '{name}' label count mismatch: {len(y)} != {expected_count}"
            )

        # Check NaN and Inf
        nan_count = int(np.isnan(X).sum())
        inf_count = int(np.isinf(X).sum())
        if nan_count > 0:
            raise ValueError(f"Found {nan_count} NaN values in partition '{name}'")
        if inf_count > 0:
            raise ValueError(f"Found {inf_count} Inf values in partition '{name}'")

        # Check constant series
        series_stds = np.std(X, axis=1)
        constant_series_count = int(np.sum(series_stds == 0.0))

        # Check label domain
        unique_labels = np.unique(y).tolist()
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError(f"Invalid binary labels in partition '{name}': {unique_labels}")

        class_counts = {
            int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True), strict=False)
        }

        report["partitions"][name] = {
            "num_samples": int(X.shape[0]),
            "sequence_length": int(X.shape[1]),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "constant_series_count": constant_series_count,
            "class_distribution": class_counts,
            "mean_signal_value": float(np.mean(X)),
            "std_signal_value": float(np.std(X)),
            "min_signal_value": float(np.min(X)),
            "max_signal_value": float(np.max(X)),
        }

    # Count exact duplicates across total dataset
    X_all = np.vstack([X_train, X_val, X_test])
    unique_rows = np.unique(X_all, axis=0)
    exact_duplicates = int(X_all.shape[0] - unique_rows.shape[0])

    report["overall_checks"] = {
        "total_samples": int(X_all.shape[0]),
        "total_unique_series": int(unique_rows.shape[0]),
        "exact_duplicate_series": exact_duplicates,
        "is_valid": True,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    report["overall_hash"] = compute_object_hash(report)

    logger.info(
        f"Data validation passed. Total samples={X_all.shape[0]}, "
        f"Duplicates={exact_duplicates}, Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}"
    )
    return report


def save_dataset_manifest(
    report: dict[str, Any],
    manifest_path: Path,
) -> None:
    """Save dataset validation report as a formal manifest JSON.

    Args:
        report: Dataset validation report dictionary.
        manifest_path: Target destination path for dataset_manifest.json.
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved dataset manifest to {manifest_path}")
