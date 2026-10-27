"""Integration tests for reentrance, idempotency, and selective invalidation."""

from pathlib import Path

import numpy as np
import pytest

from project4_rnn.config import build_experiment_config
from project4_rnn.data.preprocessing import map_official_to_internal_labels
from project4_rnn.experiments.runner import check_run_status, run_single_seed
from project4_rnn.orchestration.invalidation import (
    invalidate_experiment,
)


@pytest.mark.integration
def test_reentrance_and_invalidation_flow(
    synthetic_dev_data: tuple[np.ndarray, np.ndarray],
    temp_artifacts_dir: Path,
) -> None:
    """Verify that completed runs are skipped, and selective invalidation clears status."""
    X_dev, y_dev_orig = synthetic_dev_data
    X_dev = X_dev.astype(np.float32)

    y_dev_mapped = map_official_to_internal_labels(y_dev_orig)
    X_train = X_dev[:160]
    y_train = y_dev_mapped[:160]
    X_val = X_dev[160:200]
    y_val = y_dev_mapped[160:200]

    cfg = build_experiment_config(
        base_config_path=Path("configs/base.yaml"),
        experiment_config_path=Path("configs/phase_1/mlp_32.yaml"),
    )
    cfg.training.max_epochs = 2
    cfg.training.batch_size = 16

    # Run 1: execute fresh
    run_single_seed(
        config=cfg,
        seed=42,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        val_indices=np.arange(160, 200),
        dataset_hash="dataset-hash",
        split_hash="split-hash",
        artifacts_dir=temp_artifacts_dir,
        force=False,
    )

    run_dir = temp_artifacts_dir / "runs" / "E01_MLP_32" / "42"
    status_code, _ = check_run_status(run_dir, cfg.compute_hash(), "split-hash", "dataset-hash")
    assert status_code == "SKIP"

    # Invalidate experiment
    invalidate_experiment("E01_MLP_32", artifacts_dir=temp_artifacts_dir)
    assert not (run_dir / "status.json").exists()

    status_code_after, _ = check_run_status(
        run_dir, cfg.compute_hash(), "split-hash", "dataset-hash"
    )
    assert status_code_after == "RERUN"
