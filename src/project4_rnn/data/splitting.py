"""Stratified, reproducible development split generation and index persistence for ElectricDevices."""

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from project4_rnn.constants import (
    SPLIT_SEED,
    VAL_SPLIT_RATIO,
)
from project4_rnn.reproducibility import compute_object_hash

logger = logging.getLogger("project4_rnn.data.splitting")


def create_or_load_split(
    y_train_official: np.ndarray,
    split_indices_path: Path,
    seed: int = SPLIT_SEED,
    val_ratio: float = VAL_SPLIT_RATIO,
    train_size: int | None = None,
    val_size: int | None = None,
    force_recreate: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate or load deterministic stratified train/val indices from the official training set.

    Args:
        y_train_official: Array of official labels from the training partition.
        split_indices_path: Path to split_indices.json file.
        seed: Random seed for StratifiedShuffleSplit (default 42).
        val_ratio: Validation fraction (default 0.20 = 20%).
        train_size: Optional explicit training samples count.
        val_size: Optional explicit validation samples count.
        force_recreate: If True, regenerates split even if file exists.

    Returns:
        Tuple of (train_indices, val_indices) as numpy integer arrays.

    Raises:
        ValueError: If sample counts or partitions violate specifications.
    """
    split_indices_path = Path(split_indices_path)
    total_samples = len(y_train_official)

    if val_size is None and train_size is None:
        computed_val_size = int(round(total_samples * val_ratio))
        computed_train_size = total_samples - computed_val_size
    elif val_size is not None and train_size is not None:
        computed_train_size = train_size
        computed_val_size = val_size
    elif val_size is not None:
        computed_val_size = val_size
        computed_train_size = total_samples - val_size
    else:
        assert train_size is not None
        computed_train_size = train_size
        computed_val_size = total_samples - train_size

    if split_indices_path.is_file() and not force_recreate:
        logger.info(f"Loading existing split indices from {split_indices_path}")
        with open(split_indices_path, encoding="utf-8") as f:
            data = json.load(f)
        train_idx = np.array(data["train_indices"], dtype=np.int64)
        val_idx = np.array(data["val_indices"], dtype=np.int64)

        _validate_split_indices(
            train_idx, val_idx, total_samples, computed_train_size, computed_val_size
        )
        return train_idx, val_idx

    logger.info(
        f"Generating new stratified split with seed={seed}, total={total_samples}, "
        f"train={computed_train_size}, val={computed_val_size}"
    )

    if computed_train_size + computed_val_size != total_samples:
        raise ValueError(
            f"Sum of train ({computed_train_size}) and val ({computed_val_size}) must equal total ({total_samples})"
        )

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=computed_val_size,
        train_size=computed_train_size,
        random_state=seed,
    )

    train_idx, val_idx = next(sss.split(np.zeros(total_samples), y_train_official))
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)

    _validate_split_indices(
        train_idx, val_idx, total_samples, computed_train_size, computed_val_size
    )

    # Class distributions
    train_labels = y_train_official[train_idx]
    val_labels = y_train_official[val_idx]

    split_payload = {
        "split_seed": seed,
        "total_official_train_samples": total_samples,
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "train_class_distribution": {
            str(k): int(v)
            for k, v in zip(*np.unique(train_labels, return_counts=True), strict=False)
        },
        "val_class_distribution": {
            str(k): int(v) for k, v in zip(*np.unique(val_labels, return_counts=True), strict=False)
        },
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
    }
    split_payload["split_hash"] = compute_object_hash(split_payload)

    split_indices_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_indices_path, "w", encoding="utf-8") as f:
        json.dump(split_payload, f, indent=2)

    logger.info(f"Saved stratified split indices to {split_indices_path}")
    return train_idx, val_idx


def _validate_split_indices(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    total_samples: int,
    expected_train: int,
    expected_val: int,
) -> None:
    """Validate non-overlapping and count constraints of split indices."""
    if len(train_idx) != expected_train:
        raise ValueError(f"Expected {expected_train} train indices, got {len(train_idx)}")
    if len(val_idx) != expected_val:
        raise ValueError(f"Expected {expected_val} val indices, got {len(val_idx)}")

    overlap = set(train_idx).intersection(set(val_idx))
    if overlap:
        raise ValueError(
            f"Data leakage detected: {len(overlap)} overlapping indices between train and val: {overlap}"
        )

    combined = set(train_idx).union(set(val_idx))
    if len(combined) != total_samples:
        raise ValueError(
            f"Union of indices ({len(combined)}) does not match total samples ({total_samples})"
        )

    if np.max(train_idx) >= total_samples or np.max(val_idx) >= total_samples:
        raise ValueError("Split indices exceed total sample bounds")
    if np.min(train_idx) < 0 or np.min(val_idx) < 0:
        raise ValueError("Negative index found in split indices")
