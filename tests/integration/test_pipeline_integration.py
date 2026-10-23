"""Integration tests for end-to-end training runs, checkpointing, and artifact persistence."""

from pathlib import Path

import numpy as np
import pytest

from project4_rnn.config import build_experiment_config
from project4_rnn.experiments.runner import run_single_seed


@pytest.mark.integration
def test_single_seed_run_artifacts_creation(
    synthetic_dev_data: tuple[np.ndarray, np.ndarray],
    temp_artifacts_dir: Path,
) -> None:
    """Verify that executing a single seed generates all required artifact files."""
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

    run_single_seed(
        config=cfg,
        seed=42,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        val_indices=np.arange(400, 500),
        artifacts_dir=temp_artifacts_dir,
        force=True,
    )

    run_dir = temp_artifacts_dir / "runs" / "E01_MLP_32" / "42"
    assert (run_dir / "status.json").is_file()
    assert (run_dir / "config_resolved.yaml").is_file()
    assert (run_dir / "config_hash.txt").is_file()
    assert (run_dir / "environment.json").is_file()
    assert (run_dir / "dataset_summary.json").is_file()
    assert (run_dir / "split_summary.json").is_file()
    assert (run_dir / "model_summary.txt").is_file()
    assert (run_dir / "parameter_count.json").is_file()
    assert (run_dir / "training_history.csv").is_file()
    assert (run_dir / "validation_metrics.json").is_file()
    assert (run_dir / "predictions_validation.csv").is_file()
    assert (run_dir / "probabilities_validation.csv").is_file()
    assert (run_dir / "best_checkpoint.pt").is_file()
    assert (run_dir / "training_curves.png").is_file()
    assert (run_dir / "validation_confusion_matrix.png").is_file()
    assert (run_dir / "execution_metadata.json").is_file()

    # Re-running with force=False should trigger SKIP
    from project4_rnn.experiments.runner import check_run_status

    status_code, existing = check_run_status(run_dir, cfg.compute_hash())
    assert status_code == "SKIP"
    assert existing is not None
