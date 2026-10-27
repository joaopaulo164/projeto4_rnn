"""Builders of fake run directories used by consolidation, provenance and readiness tests."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from project4_rnn.reproducibility import compute_file_hash

DATASET_HASH = "dataset-hash"
SPLIT_HASH = "split-hash"


def make_run_metadata(
    experiment_id: str,
    seed: int,
    phase: int = 1,
    total_epochs: int = 4,
    mean_epoch_time: float = 0.5,
    val_loss: float = 0.5,
    macro_f1: float = 0.6,
    total_parameters: int = 3335,
) -> dict[str, Any]:
    """A complete, internally consistent execution_metadata document."""
    return {
        "experiment_id": experiment_id,
        "phase": phase,
        "seed": seed,
        "total_duration_sec": mean_epoch_time * total_epochs + 1.0,
        "total_training_time_sec": mean_epoch_time * total_epochs,
        "mean_epoch_time_sec": mean_epoch_time,
        "inference_time_sec": 0.01,
        "best_epoch": 2,
        "total_epochs": total_epochs,
        "best_val_loss": val_loss,
        "total_parameters": total_parameters,
        "trainable_parameters": total_parameters,
        "validation_metrics": {
            "macro_f1": macro_f1,
            "balanced_accuracy": macro_f1 + 0.01,
            "accuracy": macro_f1 + 0.02,
            "loss": val_loss,
        },
        "dataset_hash": DATASET_HASH,
        "split_hash": SPLIT_HASH,
    }


def write_run(
    artifacts_dir: Path,
    metadata: dict[str, Any],
    with_checkpoint: bool = True,
    status_split_hash: str | None = SPLIT_HASH,
) -> Path:
    """Persist a fake run directory: metadata, history, status and a dummy checkpoint."""
    run_dir = artifacts_dir / "runs" / metadata["experiment_id"] / str(metadata["seed"])
    run_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(metadata["total_epochs"])
    history = pd.DataFrame(
        {
            "epoch": range(1, epochs + 1),
            "epoch_time_sec": [metadata["mean_epoch_time_sec"]] * epochs,
        }
    )
    history.to_csv(run_dir / "training_history.csv", index=False)
    if with_checkpoint:
        checkpoint = run_dir / "best_checkpoint.pt"
        checkpoint.write_bytes(f"checkpoint-{metadata['seed']}".encode())
        metadata = {
            **metadata,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": compute_file_hash(checkpoint, "sha256"),
        }
    with open(run_dir / "execution_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    status: dict[str, Any] = {"status": "COMPLETED"}
    if status_split_hash is not None:
        status["split_hash"] = status_split_hash
        status["dataset_hash"] = DATASET_HASH
    with open(run_dir / "status.json", "w", encoding="utf-8") as f:
        json.dump(status, f)
    return run_dir
