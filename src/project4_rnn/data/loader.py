"""Load raw ECG5000 dataset files into NumPy arrays."""

import logging
from pathlib import Path

import numpy as np

from project4_rnn.constants import (
    OFFICIAL_DEV_SAMPLES,
    OFFICIAL_TEST_SAMPLES,
    SEQUENCE_LENGTH,
)

logger = logging.getLogger("project4_rnn.data.loader")


def _parse_txt_file(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse standard space/tab-separated TXT file from UCR archive.

    In UCR format, the first column is the class label and the remaining
    columns are time series values.

    Args:
        file_path: Path to .txt file.

    Returns:
        Tuple of (X, y) as numpy arrays.
    """
    data = np.loadtxt(file_path, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array from {file_path}, got shape {data.shape}")

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
                # In ARFF, target is typically the last attribute or first
                try:
                    # Try last column as label
                    label = int(float(parts[-1].strip("'\"")))
                    features = [float(p) for p in parts[:-1]]
                except ValueError:
                    # Try first column as label
                    label = int(float(parts[0].strip("'\"")))
                    features = [float(p) for p in parts[1:]]

                labels_list.append(label)
                data_lines.append(features)

    X = np.array(data_lines, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    return X, y


def load_raw_ecg5000(
    raw_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load development (500 samples) and test (4500 samples) partitions from raw files.

    Args:
        raw_dir: Directory where raw files reside.

    Returns:
        Tuple of (X_dev, y_dev, X_test, y_test).

    Raises:
        FileNotFoundError: If raw files are missing.
        ValueError: If sample counts or sequence lengths do not match official specifications.
    """
    raw_dir = Path(raw_dir)

    train_txt = raw_dir / "ECG5000_TRAIN.txt"
    test_txt = raw_dir / "ECG5000_TEST.txt"
    train_arff = raw_dir / "ECG5000_TRAIN.arff"
    test_arff = raw_dir / "ECG5000_TEST.arff"

    if train_txt.is_file() and test_txt.is_file():
        logger.info(f"Loading TXT dataset from {raw_dir}")
        X_dev, y_dev = _parse_txt_file(train_txt)
        X_test, y_test = _parse_txt_file(test_txt)
    elif train_arff.is_file() and test_arff.is_file():
        logger.info(f"Loading ARFF dataset from {raw_dir}")
        X_dev, y_dev = _parse_arff_file(train_arff)
        X_test, y_test = _parse_arff_file(test_arff)
    else:
        raise FileNotFoundError(
            f"Could not find ECG5000_TRAIN and ECG5000_TEST (.txt or .arff) in {raw_dir}."
        )

    # Validate official dimensions
    if X_dev.shape[0] != OFFICIAL_DEV_SAMPLES:
        raise ValueError(f"Expected {OFFICIAL_DEV_SAMPLES} dev samples, got {X_dev.shape[0]}")
    if X_test.shape[0] != OFFICIAL_TEST_SAMPLES:
        raise ValueError(f"Expected {OFFICIAL_TEST_SAMPLES} test samples, got {X_test.shape[0]}")
    if X_dev.shape[1] != SEQUENCE_LENGTH or X_test.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(
            f"Expected sequence length {SEQUENCE_LENGTH}, got dev {X_dev.shape[1]} and test {X_test.shape[1]}"
        )

    logger.info(
        f"Loaded ECG5000: Dev shape={X_dev.shape}, Test shape={X_test.shape}, "
        f"Dev classes={np.unique(y_dev)}, Test classes={np.unique(y_test)}"
    )
    return X_dev, y_dev, X_test, y_test
