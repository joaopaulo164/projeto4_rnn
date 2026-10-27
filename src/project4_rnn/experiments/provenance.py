"""Provenance of the final model selection and the final test evaluation.

The selected seed is derived from validation data only (lowest validation loss among the seeds of
the selected experiment, ties broken by the lowest seed). It is never defaulted. Every identity
that results depend on (checkpoint SHA-256, configuration, dataset and split hashes) is recorded
so a result can be traced back to exactly the artifacts that produced it.
"""

import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from project4_rnn.config import ExperimentConfig
from project4_rnn.experiments.timing import IncompleteRunError, find_numeric_problems
from project4_rnn.reproducibility import compute_file_hash

SEED_SELECTION_CRITERION = (
    "lowest best_val_loss on the validation partition among the seeds of the selected "
    "experiment; ties broken by the lowest seed (official test set not used)"
)

FINAL_SELECTION_REQUIRED_FIELDS: tuple[str, ...] = (
    "experiment_id",
    "architecture",
    "hidden_size",
    "pooling",
    "selected_seed",
    "checkpoint",
    "checkpoint_sha256",
    "best_epoch",
    "best_val_loss",
    "val_macro_f1",
    "val_balanced_accuracy",
    "seed_selection_criterion",
    "resolved_config",
    "config_hash",
    "dataset_hash",
    "split_hash",
    "timestamp",
)

FINAL_TEST_REQUIRED_FIELDS: tuple[str, ...] = (
    "experiment_id",
    "seed",
    "checkpoint",
    "checkpoint_sha256",
    "best_epoch",
    "dataset_hash",
    "split_hash",
    "timestamp",
)


class ProvenanceError(ValueError):
    """Raised when a provenance document is incomplete or inconsistent."""


class StaleArtifactError(RuntimeError):
    """Raised when an artifact was produced with another dataset or split than the current one."""


def check_identity(
    artifact_name: str,
    payload: Mapping[str, Any],
    expected_dataset_hash: str | None,
    expected_split_hash: str | None,
) -> None:
    """Raise StaleArtifactError if the artifact does not match the current dataset/split."""
    for key, expected in (
        ("split_hash", expected_split_hash),
        ("dataset_hash", expected_dataset_hash),
    ):
        if expected is not None and payload.get(key) != expected:
            raise StaleArtifactError(
                f"{artifact_name} is stale: {key} is {payload.get(key)!r} but the current value "
                f"is {expected!r}. It was produced with a different split/dataset; run "
                "'clean-artifacts --generated-only' and regenerate it."
            )


def select_best_seed(
    experiment_dir: Path,
    expected_split_hash: str,
    expected_dataset_hash: str | None = None,
) -> dict[str, Any]:
    """Choose the seed of an experiment with the lowest validation loss.

    Args:
        experiment_dir: ``artifacts/runs/<experiment_id>``.
        expected_split_hash: Split hash every seed must have been trained on.
        expected_dataset_hash: Optional dataset hash every seed must match.

    Returns:
        Dictionary with the chosen seed, checkpoint path, SHA-256 and validation figures.

    Raises:
        FileNotFoundError: If no seed directory with metadata exists.
        IncompleteRunError: If a seed lacks a required field.
        StaleArtifactError: If a seed was trained with another split or dataset.
    """
    experiment_dir = Path(experiment_dir)
    candidates: list[dict[str, Any]] = []
    problems = []
    for seed_dir in sorted(p for p in experiment_dir.iterdir() if p.is_dir()):
        run_id = f"{experiment_dir.name}/seed_{seed_dir.name}"
        meta_path = seed_dir / "execution_metadata.json"
        if not meta_path.is_file():
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        problems.extend(
            find_numeric_problems(
                meta, run_id, ("seed", "best_epoch", "best_val_loss"), allow_negative=False
            )
        )
        metrics = meta.get("validation_metrics") or {}
        problems.extend(find_numeric_problems(metrics, run_id, ("macro_f1", "balanced_accuracy")))
        if problems:
            continue
        check_identity(run_id, meta, expected_dataset_hash, expected_split_hash)
        checkpoint = seed_dir / "best_checkpoint.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint missing for {run_id}: {checkpoint}")
        candidates.append(
            {
                "seed": int(meta["seed"]),
                "best_epoch": int(meta["best_epoch"]),
                "best_val_loss": float(meta["best_val_loss"]),
                "val_macro_f1": float(metrics["macro_f1"]),
                "val_balanced_accuracy": float(metrics["balanced_accuracy"]),
                "checkpoint": checkpoint,
                "checkpoint_sha256": compute_file_hash(checkpoint, "sha256"),
            }
        )
    if problems:
        raise IncompleteRunError(problems)
    if not candidates:
        raise FileNotFoundError(f"No completed seed runs found in {experiment_dir}")
    return min(candidates, key=lambda c: (c["best_val_loss"], c["seed"]))


def build_final_selection_provenance(
    config: ExperimentConfig,
    seed_info: Mapping[str, Any],
    dataset_hash: str,
    split_hash: str,
) -> dict[str, Any]:
    """Provenance block of final_model_selection.json."""
    return {
        "experiment_id": config.experiment_id,
        "architecture": config.model.architecture,
        "hidden_size": config.model.hidden_size,
        "pooling": config.model.pooling,
        "selected_seed": int(seed_info["seed"]),
        "checkpoint": str(seed_info["checkpoint"]),
        "checkpoint_sha256": str(seed_info["checkpoint_sha256"]),
        "best_epoch": int(seed_info["best_epoch"]),
        "best_val_loss": float(seed_info["best_val_loss"]),
        "val_macro_f1": float(seed_info["val_macro_f1"]),
        "val_balanced_accuracy": float(seed_info["val_balanced_accuracy"]),
        "seed_selection_criterion": SEED_SELECTION_CRITERION,
        "resolved_config": config.to_dict(),
        "config_hash": config.compute_hash(),
        "dataset_hash": dataset_hash,
        "split_hash": split_hash,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def build_final_test_provenance(selection: Mapping[str, Any]) -> dict[str, Any]:
    """Provenance fields repeated in final_test_metrics.json (copied from the selection)."""
    validate_final_selection(selection)
    return {
        "experiment_id": selection["experiment_id"],
        "seed": selection["selected_seed"],
        "checkpoint": selection["checkpoint"],
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "best_epoch": selection["best_epoch"],
        "dataset_hash": selection["dataset_hash"],
        "split_hash": selection["split_hash"],
        "timestamp": datetime.now(UTC).isoformat(),
    }


def _missing(
    payload: Mapping[str, Any],
    required: tuple[str, ...],
    nullable: tuple[str, ...] = (),
) -> list[str]:
    return [k for k in required if k not in payload or (payload[k] is None and k not in nullable)]


def validate_final_selection(payload: Mapping[str, Any]) -> None:
    """Raise ProvenanceError if final_model_selection.json lacks a provenance field."""
    # A dense model may legitimately have no pooling; the key must still be present.
    missing = _missing(payload, FINAL_SELECTION_REQUIRED_FIELDS, nullable=("pooling",))
    if missing:
        raise ProvenanceError(f"final_model_selection is missing provenance fields: {missing}")
    for key in ("best_val_loss", "val_macro_f1", "val_balanced_accuracy"):
        if not math.isfinite(float(payload[key])):
            raise ProvenanceError(f"final_model_selection field '{key}' is not finite")
    if len(str(payload["checkpoint_sha256"])) != 64:
        raise ProvenanceError("checkpoint_sha256 must be a 64-character SHA-256 hex digest")


def validate_final_test_metrics(payload: Mapping[str, Any]) -> None:
    """Raise ProvenanceError if final_test_metrics.json lacks a provenance field."""
    missing = _missing(payload, FINAL_TEST_REQUIRED_FIELDS)
    if missing:
        raise ProvenanceError(f"final test metrics are missing provenance fields: {missing}")


def verify_checkpoint_integrity(selection: Mapping[str, Any]) -> None:
    """Confirm the checkpoint on disk still has the recorded SHA-256 before it is evaluated."""
    path = Path(str(selection["checkpoint"]))
    if not path.is_file():
        raise FileNotFoundError(f"Selected checkpoint not found: {path}")
    observed = compute_file_hash(path, "sha256")
    if observed != selection["checkpoint_sha256"]:
        raise ProvenanceError(
            f"Checkpoint {path} changed after selection: recorded SHA-256 "
            f"{selection['checkpoint_sha256']} but observed {observed}"
        )
