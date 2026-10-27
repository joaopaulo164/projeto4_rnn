"""Prepared ElectricDevices partitions together with the identity of the dataset and the split."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from project4_rnn.config import ExperimentConfig
from project4_rnn.data.download import download_electricdevices
from project4_rnn.data.hashing import hash_series_matrix
from project4_rnn.data.loader import load_raw_electricdevices
from project4_rnn.data.preprocessing import (
    fit_and_apply_scaler,
    map_official_to_internal_labels,
)
from project4_rnn.data.splitting import create_or_load_split
from project4_rnn.data.validation import save_dataset_manifest, validate_dataset_arrays

logger = logging.getLogger("project4_rnn.data.bundle")


@dataclass
class DataBundle:
    """Training-ready partitions plus the provenance needed to tie results to this data.

    ``X_*`` are float32 (after the optional normalization fitted on train only); ``y_*`` use the
    internal labels 0..6. ``*_official`` keep the untouched official labels 1..7 and
    ``*_series_hashes`` the SHA-256 fingerprints of the raw (pre-normalization) series.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    dev_series_hashes: list[str]
    test_series_hashes: list[str]
    y_dev_official: np.ndarray
    y_test_official: np.ndarray
    dataset_hash: str
    split_hash: str
    split_payload: dict[str, Any]
    validation_report: dict[str, Any]
    manifest: dict[str, Any]


def build_data_bundle(
    cfg: ExperimentConfig,
    force: bool = False,
    download: bool = True,
) -> DataBundle:
    """Load the official files, verify the contract, split by series hash and validate.

    The data contract (checksums, shapes, classes, NaN/Inf), the grouped split and its integrity
    (zero shared hashes, full coverage, seven classes per partition) are all verified here, so no
    separate data-validation pipeline step is needed.

    Args:
        cfg: Resolved configuration (only the ``data`` section is used).
        force: Regenerate the split instead of reusing a verified persisted one.
        download: Acquire the raw files when absent.
    """
    raw_dir = Path(cfg.data.raw_dir)
    processed_dir = Path(cfg.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if download:
        download_electricdevices(raw_dir=raw_dir)

    X_dev_raw, y_dev_official, X_test_raw, y_test_official, manifest = load_raw_electricdevices(
        raw_dir=raw_dir, verify_checksums=True, processed_dir=processed_dir
    )
    dataset_hash = str(manifest["dataset_hash"])

    dev_hashes = hash_series_matrix(X_dev_raw)
    test_hashes = hash_series_matrix(X_test_raw)

    split = create_or_load_split(
        y_train_official=y_dev_official,
        series_hashes=dev_hashes,
        split_indices_path=processed_dir / "split_indices.json",
        dataset_hash=dataset_hash,
        seed=cfg.data.split_seed,
        n_splits=cfg.data.split_n_splits,
        val_ratio=cfg.data.val_split_ratio,
        force_recreate=force,
    )

    X_train = X_dev_raw[split.train_idx]
    X_val = X_dev_raw[split.val_idx]
    X_test = X_test_raw
    y_train_orig = y_dev_official[split.train_idx]
    y_val_orig = y_dev_official[split.val_idx]

    # Validate on the raw float64 values so duplicate counts are not distorted by float32 casts.
    validation_report = validate_dataset_arrays(
        X_train=X_train,
        y_train=y_train_orig,
        X_val=X_val,
        y_val=y_val_orig,
        X_test=X_test,
        y_test=y_test_official,
        expected_train_count=len(split.train_idx),
        expected_val_count=len(split.val_idx),
        expected_test_count=len(y_test_official),
    )

    if cfg.data.normalization.get("enabled", False):
        scaler_type = cfg.data.normalization.get("type", "zscore")
        X_train, X_val, X_test, _ = fit_and_apply_scaler(
            X_train, X_val, X_test, scaler_type=scaler_type
        )

    manifest["validation_report"] = validation_report
    manifest["split_hash"] = split.split_hash
    save_dataset_manifest(manifest, processed_dir / "dataset_manifest.json")

    return DataBundle(
        X_train=X_train.astype(np.float32),
        y_train=map_official_to_internal_labels(y_train_orig),
        X_val=X_val.astype(np.float32),
        y_val=map_official_to_internal_labels(y_val_orig),
        X_test=X_test.astype(np.float32),
        y_test=map_official_to_internal_labels(y_test_official),
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        dev_series_hashes=dev_hashes,
        test_series_hashes=test_hashes,
        y_dev_official=y_dev_official,
        y_test_official=y_test_official,
        dataset_hash=dataset_hash,
        split_hash=split.split_hash,
        split_payload=split.payload,
        validation_report=validation_report,
        manifest=manifest,
    )
