"""Load raw ElectricDevices dataset files into NumPy arrays with strict contract validation."""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from project4_rnn.constants import (
    DATASET_NAME,
    DATASET_SOURCE,
    DATASET_VARIANT,
    NUM_CLASSES,
    NUM_FEATURES,
    OFFICIAL_CLASSES,
    OFFICIAL_TEST_SAMPLES,
    OFFICIAL_TRAIN_SAMPLES,
    SEQUENCE_LENGTH,
    TEST_FILE_NAME,
    TEST_MD5,
    TRAIN_FILE_NAME,
    TRAIN_MD5,
)
from project4_rnn.data.validation import DatasetContractError, save_dataset_manifest
from project4_rnn.reproducibility import compute_file_hash, compute_object_hash

logger = logging.getLogger("project4_rnn.data.loader")


def _parse_ts_file(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse standard UCR/UEA .ts file format.

    Args:
        file_path: Path to .ts file.

    Returns:
        Tuple of (X, y) as numpy arrays.
    """
    data_lines: list[list[float]] = []
    labels_list: list[int] = []

    with open(file_path, encoding="utf-8", errors="ignore") as f:
        in_data = False
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            if line_str.lower().startswith("@data"):
                in_data = True
                continue
            if in_data:
                # Handle colon separation for dimensions or label in UEA .ts format
                if ":" in line_str:
                    parts = line_str.split(":")
                    series_part = parts[0].strip()
                    label_part = parts[1].strip() if len(parts) > 1 else ""
                    vals = [float(v.strip()) for v in series_part.split(",") if v.strip()]
                    label = int(float(label_part.strip("'\"")))
                else:
                    parts = [p.strip() for p in line_str.split(",") if p.strip()]
                    if not parts:
                        continue
                    # Last column is class label
                    label = int(float(parts[-1].strip("'\"")))
                    vals = [float(p) for p in parts[:-1]]

                data_lines.append(vals)
                labels_list.append(label)

    X = np.array(data_lines, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    return X, y


def _parse_txt_file(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse standard space/tab/comma-separated TXT file from UCR archive.

    In standard UCR .txt format, the first column is the class label and the remaining
    columns are time series values.

    Args:
        file_path: Path to .txt file.

    Returns:
        Tuple of (X, y) as numpy arrays.
    """
    data = np.loadtxt(file_path, dtype=np.float32)
    if data.ndim != 2:
        raise DatasetContractError(
            message=f"Expected 2D array from {file_path}, got shape {data.shape}",
            expected=2,
            observed=data.ndim,
            affected_file=file_path,
        )

    # In UCR UnivData format, first column is label
    labels = data[:, 0].astype(np.int64)
    features = data[:, 1:].astype(np.float32)
    return features, labels


def _parse_arff_file(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse standard ARFF format file.

    Args:
        file_path: Path to .arff file.

    Returns:
        Tuple of (X, y) as numpy arrays.
    """
    data_lines: list[list[float]] = []
    labels_list: list[int] = []

    with open(file_path, encoding="utf-8", errors="ignore") as f:
        in_data = False
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith("%"):
                continue
            if line_str.lower().startswith("@data"):
                in_data = True
                continue
            if in_data:
                parts = [p.strip() for p in line_str.split(",") if p.strip()]
                if not parts:
                    continue
                try:
                    label = int(float(parts[-1].strip("'\"")))
                    features = [float(p) for p in parts[:-1]]
                except ValueError:
                    label = int(float(parts[0].strip("'\"")))
                    features = [float(p) for p in parts[1:]]

                labels_list.append(label)
                data_lines.append(features)

    X = np.array(data_lines, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    return X, y


def load_raw_electricdevices(
    raw_dir: Path,
    verify_checksums: bool = True,
    processed_dir: Path | None = Path("data/processed"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load ElectricDevices official train and test partitions with strict contract validation.

    Args:
        raw_dir: Directory where raw dataset files reside.
        verify_checksums: If True, asserts MD5 checksums against official benchmarks.
        processed_dir: Directory where dataset_manifest.json will be saved.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test, manifest_dict).

    Raises:
        FileNotFoundError: If raw files are missing.
        DatasetContractError: If any contract condition (dimensions, checksums, classes, values) fails.
    """
    raw_dir = Path(raw_dir)

    train_ts = raw_dir / TRAIN_FILE_NAME
    test_ts = raw_dir / TEST_FILE_NAME
    train_txt = raw_dir / "ElectricDevices_TRAIN.txt"
    test_txt = raw_dir / "ElectricDevices_TEST.txt"
    train_arff = raw_dir / "ElectricDevices_TRAIN.arff"
    test_arff = raw_dir / "ElectricDevices_TEST.arff"

    train_file: Path | None = None
    test_file: Path | None = None
    file_format = "unknown"

    if train_ts.is_file() and test_ts.is_file():
        train_file = train_ts
        test_file = test_ts
        file_format = "ts"
    elif train_txt.is_file() and test_txt.is_file():
        train_file = train_txt
        test_file = test_txt
        file_format = "txt"
    elif train_arff.is_file() and test_arff.is_file():
        train_file = train_arff
        test_file = test_arff
        file_format = "arff"
    else:
        raise FileNotFoundError(
            f"Could not find ElectricDevices train and test files (.ts, .txt, or .arff) in {raw_dir}."
        )

    # 1. Verify and calculate checksums
    train_sha256 = compute_file_hash(train_file, "sha256")
    test_sha256 = compute_file_hash(test_file, "sha256")
    train_md5 = compute_file_hash(train_file, "md5")
    test_md5 = compute_file_hash(test_file, "md5")

    if verify_checksums and file_format == "ts":
        if train_md5 != TRAIN_MD5:
            raise DatasetContractError(
                message=f"MD5 checksum mismatch for training file {train_file.name}",
                expected=TRAIN_MD5,
                observed=train_md5,
                affected_file=train_file,
                expected_checksum=TRAIN_MD5,
                observed_checksum=train_md5,
                suggested_action="Ensure the official ElectricDevices_TRAIN.ts benchmark file is downloaded without corruption.",
            )
        if test_md5 != TEST_MD5:
            raise DatasetContractError(
                message=f"MD5 checksum mismatch for test file {test_file.name}",
                expected=TEST_MD5,
                observed=test_md5,
                affected_file=test_file,
                expected_checksum=TEST_MD5,
                observed_checksum=test_md5,
                suggested_action="Ensure the official ElectricDevices_TEST.ts benchmark file is downloaded without corruption.",
            )

    # 2. Parse files into arrays
    if file_format == "ts":
        X_train, y_train = _parse_ts_file(train_file)
        X_test, y_test = _parse_ts_file(test_file)
    elif file_format == "txt":
        X_train, y_train = _parse_txt_file(train_file)
        X_test, y_test = _parse_txt_file(test_file)
    elif file_format == "arff":
        X_train, y_train = _parse_arff_file(train_file)
        X_test, y_test = _parse_arff_file(test_file)
    else:
        raise DatasetContractError(
            message=f"Unsupported file format: {file_format}",
            affected_file=train_file,
            suggested_action="Provide files in .ts, .txt, or .arff format.",
        )

    # 3. Contract Checks: Sample counts
    if X_train.shape[0] != OFFICIAL_TRAIN_SAMPLES:
        raise DatasetContractError(
            message="Official training sample count violation",
            expected=OFFICIAL_TRAIN_SAMPLES,
            observed=X_train.shape[0],
            affected_file=train_file,
            suggested_action=f"Ensure training file has exactly {OFFICIAL_TRAIN_SAMPLES} samples.",
        )

    if X_test.shape[0] != OFFICIAL_TEST_SAMPLES:
        raise DatasetContractError(
            message="Official test sample count violation",
            expected=OFFICIAL_TEST_SAMPLES,
            observed=X_test.shape[0],
            affected_file=test_file,
            suggested_action=f"Ensure test file has exactly {OFFICIAL_TEST_SAMPLES} samples.",
        )

    # 4. Contract Checks: Sequence Length (must be 96, reject 720 or arbitrary lengths)
    if X_train.shape[1] != SEQUENCE_LENGTH:
        raise DatasetContractError(
            message=f"Sequence length mismatch for training partition (found {X_train.shape[1]}, expected {SEQUENCE_LENGTH})",
            expected=SEQUENCE_LENGTH,
            observed=X_train.shape[1],
            affected_file=train_file,
            suggested_action="Use the official benchmark ElectricDevices version with sequence length 96 (do not use raw 720).",
        )

    if X_test.shape[1] != SEQUENCE_LENGTH:
        raise DatasetContractError(
            message=f"Sequence length mismatch for test partition (found {X_test.shape[1]}, expected {SEQUENCE_LENGTH})",
            expected=SEQUENCE_LENGTH,
            observed=X_test.shape[1],
            affected_file=test_file,
            suggested_action="Use the official benchmark ElectricDevices version with sequence length 96 (do not use raw 720).",
        )

    # 5. Contract Checks: Number of classes (must be exactly 7)
    train_classes = sorted([int(c) for c in np.unique(y_train)])
    test_classes = sorted([int(c) for c in np.unique(y_test)])
    all_classes = sorted(set(train_classes).union(set(test_classes)))

    if len(all_classes) != NUM_CLASSES or set(all_classes) != set(OFFICIAL_CLASSES):
        raise DatasetContractError(
            message=f"Dataset class domain violation (found classes {all_classes}, expected {list(OFFICIAL_CLASSES)})",
            expected=list(OFFICIAL_CLASSES),
            observed=all_classes,
            affected_file=train_file,
            suggested_action="Ensure all 7 official classes (1 through 7) are represented in the dataset.",
        )

    # 6. Contract Checks: Absence of NaN and Inf
    nan_train = int(np.isnan(X_train).sum())
    nan_test = int(np.isnan(X_test).sum())
    inf_train = int(np.isinf(X_train).sum())
    inf_test = int(np.isinf(X_test).sum())

    if nan_train > 0 or nan_test > 0:
        raise DatasetContractError(
            message=f"Found NaN values in dataset (train: {nan_train}, test: {nan_test})",
            expected=0,
            observed=nan_train + nan_test,
            suggested_action="Ensure dataset contains clean, non-NaN floating point numbers.",
        )

    if inf_train > 0 or inf_test > 0:
        raise DatasetContractError(
            message=f"Found infinite values in dataset (train: {inf_train}, test: {inf_test})",
            expected=0,
            observed=inf_train + inf_test,
            suggested_action="Ensure dataset contains bounded floating point numbers without +/- Inf.",
        )

    # 7. Distribution of classes
    train_dist = {
        int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True), strict=False)
    }
    test_dist = {
        int(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True), strict=False)
    }

    manifest_data = {
        "dataset_name": DATASET_NAME,
        "variant": DATASET_VARIANT,
        "source": DATASET_SOURCE,
        "contract_status": "PASSED",
        "official_split_preserved": True,
        "files": {
            "train": {
                "name": train_file.name,
                "md5": train_md5,
                "sha256": train_sha256,
                "samples": int(X_train.shape[0]),
                "sequence_length": int(X_train.shape[1]),
                "num_classes": len(train_classes),
                "classes": train_classes,
                "class_distribution": train_dist,
            },
            "test": {
                "name": test_file.name,
                "md5": test_md5,
                "sha256": test_sha256,
                "samples": int(X_test.shape[0]),
                "sequence_length": int(X_test.shape[1]),
                "num_classes": len(test_classes),
                "classes": test_classes,
                "class_distribution": test_dist,
            },
        },
        "total_samples": int(X_train.shape[0] + X_test.shape[0]),
        "sequence_length": SEQUENCE_LENGTH,
        "num_classes": NUM_CLASSES,
        "num_dimensions": NUM_FEATURES,
        "loaded_timestamp": datetime.now(UTC).isoformat(),
    }
    manifest_data["manifest_hash"] = compute_object_hash(manifest_data)

    if processed_dir is not None:
        save_dataset_manifest(manifest_data, Path(processed_dir) / "dataset_manifest.json")

    logger.info(
        f"Loaded {DATASET_NAME} Contract Validated: "
        f"Train shape={X_train.shape}, Test shape={X_test.shape}, Classes={all_classes}"
    )

    return X_train, y_train, X_test, y_test, manifest_data
