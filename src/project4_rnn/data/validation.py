"""Data contract validation, integrity checks, and dataset manifest generation for ElectricDevices."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from project4_rnn.constants import (
    NUM_CLASSES,
    OFFICIAL_CLASSES,
    SEQUENCE_LENGTH,
)
from project4_rnn.reproducibility import compute_object_hash

logger = logging.getLogger("project4_rnn.data.validation")


class DatasetContractError(Exception):
    """Exception raised when a dataset fails structural, checksum, or dimensional contract requirements."""

    def __init__(
        self,
        message: str,
        expected: Any = None,
        observed: Any = None,
        affected_file: str | Path | None = None,
        expected_checksum: str | None = None,
        observed_checksum: str | None = None,
        suggested_action: str | None = None,
    ) -> None:
        self.message = message
        self.expected = expected
        self.observed = observed
        self.affected_file = str(affected_file) if affected_file is not None else None
        self.expected_checksum = expected_checksum
        self.observed_checksum = observed_checksum
        self.suggested_action = suggested_action

        details = [f"DatasetContractError: {message}"]
        if affected_file is not None:
            details.append(f"  Affected File: {affected_file}")
        if expected is not None or observed is not None:
            details.append(f"  Expected: {expected} | Observed: {observed}")
        if expected_checksum is not None or observed_checksum is not None:
            details.append(
                f"  Expected Checksum: {expected_checksum} | Observed Checksum: {observed_checksum}"
            )
        if suggested_action is not None:
            details.append(f"  Suggested Action: {suggested_action}")

        super().__init__("\n".join(details))


def validate_dataset_arrays(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    expected_train_count: int,
    expected_val_count: int,
    expected_test_count: int = 7711,
) -> dict[str, Any]:
    """Execute rigorous data quality, domain, and consistency checks across partitions.

    Args:
        X_train: Training time series array.
        y_train: Training official label array.
        X_val: Validation time series array.
        y_val: Validation official label array.
        X_test: Test time series array.
        y_test: Test official label array.
        expected_train_count: Expected training samples count.
        expected_val_count: Expected validation samples count.
        expected_test_count: Expected test samples count (7711).

    Returns:
        Validation report dictionary with metrics and check statuses.

    Raises:
        DatasetContractError: If any integrity constraint or contract requirement is violated.
    """
    partitions = {
        "train": (X_train, y_train, expected_train_count),
        "val": (X_val, y_val, expected_val_count),
        "test": (X_test, y_test, expected_test_count),
    }

    report: dict[str, Any] = {"partitions": {}, "overall_checks": {}}

    for name, (X, y, expected_count) in partitions.items():
        if X.ndim != 2:
            raise DatasetContractError(
                message=f"Partition '{name}' must have 2 dimensions (samples, sequence_length), got ndim={X.ndim}",
                expected=2,
                observed=X.ndim,
                suggested_action="Verify dataset loading logic to ensure 2D array representation.",
            )

        if X.shape[0] != expected_count:
            raise DatasetContractError(
                message=f"Partition '{name}' sample count mismatch",
                expected=expected_count,
                observed=X.shape[0],
                suggested_action="Ensure partition sizes match official split specifications.",
            )

        if X.shape[1] != SEQUENCE_LENGTH:
            raise DatasetContractError(
                message=f"Partition '{name}' sequence length mismatch",
                expected=SEQUENCE_LENGTH,
                observed=X.shape[1],
                suggested_action="Ensure ElectricDevices benchmark length 96 is loaded without truncation/padding.",
            )

        if len(y) != expected_count:
            raise DatasetContractError(
                message=f"Partition '{name}' label count mismatch",
                expected=expected_count,
                observed=len(y),
                suggested_action="Verify labels array matches sample rows.",
            )

        # Check NaN and Inf
        nan_count = int(np.isnan(X).sum())
        inf_count = int(np.isinf(X).sum())
        if nan_count > 0:
            raise DatasetContractError(
                message=f"Found {nan_count} NaN values in partition '{name}'",
                expected=0,
                observed=nan_count,
                suggested_action="Sanitize or clean dataset to remove NaN values.",
            )
        if inf_count > 0:
            raise DatasetContractError(
                message=f"Found {inf_count} Inf values in partition '{name}'",
                expected=0,
                observed=inf_count,
                suggested_action="Sanitize dataset to eliminate infinite values.",
            )

        # Check for empty series (length 0 or zero variance across entire series)
        series_stds = np.std(X, axis=1)
        constant_series_count = int(np.sum(series_stds == 0.0))

        # Check label domain (must be within official classes 1..7 or mapped 0..6)
        unique_labels = sorted([int(v) for v in np.unique(y)])
        is_official = set(unique_labels).issubset(set(OFFICIAL_CLASSES))
        is_internal = set(unique_labels).issubset(set(range(NUM_CLASSES)))
        if not (is_official or is_internal):
            raise DatasetContractError(
                message=f"Invalid label domain in partition '{name}': {unique_labels}",
                expected=list(OFFICIAL_CLASSES),
                observed=unique_labels,
                suggested_action="Ensure class labels are within official {1..7} or mapped {0..6}.",
            )

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
    }
    # The hash covers only deterministic content; the timestamp is added afterwards.
    report["overall_hash"] = compute_object_hash(report)
    report["overall_checks"]["timestamp"] = datetime.now(UTC).isoformat()

    logger.info(
        f"Data validation passed. Total samples={X_all.shape[0]}, "
        f"Duplicates={exact_duplicates}, Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}"
    )
    return report


def save_dataset_manifest(
    manifest_data: dict[str, Any],
    manifest_path: Path,
) -> None:
    """Save dataset contract validation report as a formal manifest JSON.

    Args:
        manifest_data: Dataset manifest metadata and validation report dictionary.
        manifest_path: Target destination path for dataset_manifest.json.
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info(f"Saved dataset manifest to {manifest_path}")
