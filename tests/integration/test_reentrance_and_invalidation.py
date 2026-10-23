"""Integration tests for reentrance, idempotency, and selective invalidation."""

from pathlib import Path

import numpy as np
import pytest

from project4_rnn.config import build_experiment_config
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
    from project4_rnn.data.preprocessing import binarize_labels

    y_dev_bin = binarize_labels(y_dev_orig)
    X_train = X_dev[:400]
    y_train = y_dev_bin[:400]
    X_val = X_dev[400:500]
    y_val = y_dev_bin[400:500]

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
        val_indices=np.arange(400, 500),
        artifacts_dir=temp_artifacts_dir,
        force=False,
    )

    run_dir = temp_artifacts_dir / "runs" / "E01_MLP_32" / "42"
    status_code, _ = check_run_status(run_dir, cfg.compute_hash())
    assert status_code == "SKIP"

    # Invalidate experiment
    invalidate_experiment("E01_MLP_32", artifacts_dir=temp_artifacts_dir)
    assert not (run_dir / "status.json").exists()

    status_code_after, _ = check_run_status(run_dir, cfg.compute_hash())
    assert status_code_after == "RERUN"
