"""Experiment runner orchestrating individual seed training runs and artifact generation for ElectricDevices."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from project4_rnn.config import ExperimentConfig
from project4_rnn.constants import (
    INTERNAL_TO_OFFICIAL_LABEL,
    OFFICIAL_TO_INTERNAL_LABEL,
)
from project4_rnn.data.dataset import create_dataloaders
from project4_rnn.evaluation.plots import (
    plot_confusion_matrix,
    plot_training_curves,
)
from project4_rnn.evaluation.predictions import save_prediction_artifacts
from project4_rnn.logging_config import setup_logger
from project4_rnn.models.factory import build_model
from project4_rnn.reproducibility import (
    get_environment_info,
    set_seed,
)
from project4_rnn.training.trainer import Trainer

logger = logging.getLogger("project4_rnn.experiments.runner")

REQUIRED_RUN_ARTIFACTS = [
    "status.json",
    "config_resolved.yaml",
    "config_hash.txt",
    "environment.json",
    "dataset_summary.json",
    "split_summary.json",
    "label_mapping.json",
    "model_summary.txt",
    "parameter_count.json",
    "training_history.csv",
    "validation_metrics.json",
    "validation_metrics_by_class.csv",
    "predictions_validation.csv",
    "probabilities_validation.csv",
    "best_checkpoint.pt",
    "training_curves.png",
    "validation_confusion_matrix_raw.png",
    "validation_confusion_matrix_normalized.png",
    "execution_metadata.json",
]


def check_run_status(
    run_dir: Path,
    expected_config_hash: str,
) -> tuple[str, dict[str, Any] | None]:
    """Check whether a specific seed run is completed, invalid, or pending.

    Returns:
        Tuple of (status_code, existing_summary_dict) where status_code is 'SKIP', 'RERUN', or 'PENDING'.
    """
    if not run_dir.is_dir():
        return "PENDING", None

    # Check if all required artifact files exist
    for artifact in REQUIRED_RUN_ARTIFACTS:
        if not (run_dir / artifact).is_file():
            return "RERUN", None

    # Check config hash match
    hash_file = run_dir / "config_hash.txt"
    with open(hash_file, encoding="utf-8") as f:
        stored_hash = f.read().strip()

    if stored_hash != expected_config_hash:
        return "RERUN", None

    # Check status.json
    status_file = run_dir / "status.json"
    try:
        with open(status_file, encoding="utf-8") as f:
            status_data = json.load(f)
        if status_data.get("status") == "COMPLETED":
            metadata_file = run_dir / "execution_metadata.json"
            summary = None
            if metadata_file.is_file():
                with open(metadata_file, encoding="utf-8") as mf:
                    summary = json.load(mf)
            return "SKIP", summary
    except Exception as status_err:
        logger.debug(f"Could not read existing run status in {run_dir}: {status_err}")

    return "RERUN", None


def run_single_seed(
    config: ExperimentConfig,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_indices: np.ndarray,
    artifacts_dir: Path = Path("artifacts"),
    force: bool = False,
) -> dict[str, Any]:
    """Execute training and evaluation for a single seed with full multiclass artifact generation.

    Args:
        config: Resolved ExperimentConfig.
        seed: Random seed.
        X_train: Training features.
        y_train: Training mapped labels (0..6).
        X_val: Validation features.
        y_val: Validation mapped labels (0..6).
        val_indices: Original indices of validation samples in development partition.
        artifacts_dir: Root directory for artifacts.
        force: If True, forces execution even if completed artifacts exist.

    Returns:
        Dictionary containing run summary and metrics.
    """
    run_dir = artifacts_dir / "runs" / config.experiment_id / str(seed)
    config_hash = config.compute_hash()

    if not force:
        status_code, existing_summary = check_run_status(run_dir, config_hash)
        if status_code == "SKIP" and existing_summary is not None:
            logger.info(f"Skipping completed run for {config.experiment_id} (Seed {seed}) [SKIP]")
            return existing_summary

    run_dir.mkdir(parents=True, exist_ok=True)

    # Dedicated run logger
    run_logger = setup_logger(
        name=f"{config.experiment_id}_{seed}",
        log_file=run_dir / "run.log",
    )
    run_logger.info(f"Starting run {config.experiment_id} | Seed {seed}")

    start_timestamp = datetime.now(UTC)

    # 1. Set global deterministic seed
    set_seed(seed=seed, deterministic=True)

    # 2. Save resolved config and hash
    config.save_yaml(run_dir / "config_resolved.yaml")
    with open(run_dir / "config_hash.txt", "w", encoding="utf-8") as f:
        f.write(config_hash)

    # 3. Save environment info
    env_info = get_environment_info()
    with open(run_dir / "environment.json", "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)

    # 4. Save dataset summary and split summary
    dataset_summary = {
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "sequence_length": int(X_train.shape[1]),
        "num_classes": config.model.num_classes,
        "train_class_counts": {
            str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True), strict=False)
        },
        "val_class_counts": {
            str(k): int(v) for k, v in zip(*np.unique(y_val, return_counts=True), strict=False)
        },
    }
    with open(run_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2)

    split_summary = {
        "split_seed": config.data.split_seed,
        "val_indices_count": len(val_indices),
        "val_indices": val_indices.tolist(),
    }
    with open(run_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)

    # Save label mapping
    label_map_payload = {
        "official_to_internal": OFFICIAL_TO_INTERNAL_LABEL,
        "internal_to_official": INTERNAL_TO_OFFICIAL_LABEL,
    }
    with open(run_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_map_payload, f, indent=2)

    # 5. Build model and record summary
    is_rnn = config.model.architecture in ("gru", "lstm")
    model = build_model(config, validate_params=True)

    with open(run_dir / "model_summary.txt", "w", encoding="utf-8") as f:
        f.write(model.get_model_summary())

    param_count_payload = {
        "total_parameters": model.count_parameters(),
        "trainable_parameters": model.count_trainable_parameters(),
        "parameter_breakdown": model.get_parameter_breakdown(),
    }
    with open(run_dir / "parameter_count.json", "w", encoding="utf-8") as f:
        json.dump(param_count_payload, f, indent=2)

    # 6. Create DataLoaders
    train_loader, val_loader, _ = create_dataloaders(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=config.training.batch_size,
        seed=seed,
        reshape_for_rnn=is_rnn,
        num_workers=config.system.num_workers,
    )

    # 7. Execute Training
    trainer = Trainer(
        model=model,
        config=config,
        seed=seed,
        checkpoint_dir=run_dir,
    )
    run_summary = trainer.fit(train_loader=train_loader, val_loader=val_loader)

    # 8. Evaluate best checkpoint on validation set
    val_metrics, val_preds, val_probs, inf_time = trainer.evaluate(val_loader)
    with open(run_dir / "validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)

    # Save per-class metrics CSV
    per_class_dict = val_metrics.get("per_class_metrics", {})
    if per_class_dict:
        class_df = pd.DataFrame(per_class_dict).T
        class_df.to_csv(run_dir / "validation_metrics_by_class.csv")
    else:
        pd.DataFrame().to_csv(run_dir / "validation_metrics_by_class.csv")

    # 9. Save predictions and probabilities
    save_prediction_artifacts(
        predictions_csv_path=run_dir / "predictions_validation.csv",
        probabilities_csv_path=run_dir / "probabilities_validation.csv",
        sample_indices=val_indices,
        y_true=y_val,
        y_pred=val_preds,
        y_prob=val_probs,
        experiment_id=config.experiment_id,
        phase=config.phase,
        seed=seed,
        checkpoint_name="best_checkpoint.pt",
    )

    # 10. Generate run plots
    history_df = trainer.history.to_dataframe()
    plot_training_curves(
        history_df=history_df,
        output_path=run_dir / "training_curves.png",
        experiment_id=config.experiment_id,
        seed=seed,
    )
    plot_confusion_matrix(
        y_true=y_val,
        y_pred=val_preds,
        output_path=run_dir / "validation_confusion_matrix_raw.png",
        title=f"Val Confusion Matrix Raw ({config.experiment_id} - Seed {seed})",
        normalize=False,
    )
    plot_confusion_matrix(
        y_true=y_val,
        y_pred=val_preds,
        output_path=run_dir / "validation_confusion_matrix_normalized.png",
        title=f"Val Confusion Matrix Normalized ({config.experiment_id} - Seed {seed})",
        normalize=True,
    )

    end_timestamp = datetime.now(UTC)
    duration = (end_timestamp - start_timestamp).total_seconds()

    # 11. Execution metadata
    exec_metadata = {
        "experiment_id": config.experiment_id,
        "phase": config.phase,
        "seed": seed,
        "start_time": start_timestamp.isoformat(),
        "end_time": end_timestamp.isoformat(),
        "total_duration_sec": duration,
        "best_epoch": run_summary["best_epoch"],
        "total_epochs": run_summary["total_epochs"],
        "best_val_loss": run_summary["best_val_loss"],
        "total_parameters": model.count_parameters(),
        "trainable_parameters": model.count_trainable_parameters(),
        "validation_metrics": val_metrics,
        "config_hash": config_hash,
    }
    with open(run_dir / "execution_metadata.json", "w", encoding="utf-8") as f:
        json.dump(exec_metadata, f, indent=2)

    # 12. Status marker
    status_payload = {
        "status": "COMPLETED",
        "timestamp": end_timestamp.isoformat(),
        "config_hash": config_hash,
    }
    with open(run_dir / "status.json", "w", encoding="utf-8") as f:
        json.dump(status_payload, f, indent=2)

    run_logger.info(
        f"Completed run {config.experiment_id} | Seed {seed} | "
        f"Val MacroF1: {val_metrics['macro_f1']:.4f} | Val BalAcc: {val_metrics['balanced_accuracy']:.4f}"
    )
    return exec_metadata


def run_experiment_all_seeds(
    config: ExperimentConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_indices: np.ndarray,
    artifacts_dir: Path = Path("artifacts"),
    force: bool = False,
) -> list[dict[str, Any]]:
    """Run all 5 configured seeds for an experiment.

    Args:
        config: Resolved ExperimentConfig.
        X_train: Training features array.
        y_train: Training mapped labels array (0..6).
        X_val: Validation features array.
        y_val: Validation mapped labels array (0..6).
        val_indices: Validation sample index array.
        artifacts_dir: Artifacts root directory.
        force: Force re-execution.

    Returns:
        List of execution summary dictionaries across all seeds.
    """
    logger.info(f"Running all {len(config.training.train_seeds)} seeds for {config.experiment_id}")
    summaries: list[dict[str, Any]] = []

    for seed in config.training.train_seeds:
        run_sum = run_single_seed(
            config=config,
            seed=seed,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            val_indices=val_indices,
            artifacts_dir=artifacts_dir,
            force=force,
        )
        summaries.append(run_sum)

    return summaries
