"""Final model selection across all experimental configurations based on validation performance for ElectricDevices."""

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from project4_rnn.constants import PARSIMONY_THRESHOLD
from project4_rnn.experiments.pooling_selection import REQUIRED_SUMMARY_FIELDS
from project4_rnn.experiments.timing import IncompleteRunError, find_numeric_problems
from project4_rnn.reproducibility import compute_object_hash

logger = logging.getLogger("project4_rnn.experiments.model_selection")


def select_final_model(
    all_summaries: list[dict[str, Any]],
    output_path: Path,
    dataset_hash: str,
    split_hash: str,
    parsimony_threshold: float = PARSIMONY_THRESHOLD,
    provenance_builder: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Select winning architecture across all 8 configurations strictly on validation set.

    Selection hierarchy:
        1. Higher mean validation Macro F1 score.
        2. Higher mean validation balanced accuracy (1st tie-breaker).
        3. Lower validation Macro F1 standard deviation (2nd tie-breaker).
        4. Lower mean training time in seconds (3rd tie-breaker).
        5. Parsimony rule: if difference in Macro F1 <= parsimony_threshold (0.005),
           prefer the simpler model (fewer parameters).

    Args:
        all_summaries: List of summary dicts for all completed experiments.
        output_path: Destination path for final_model_selection.json.
        dataset_hash: Identity hash of the dataset the runs used.
        split_hash: Identity hash of the train/validation split the runs used.
        parsimony_threshold: Parsimony margin (default 0.005 = 0.5% Macro F1).
        provenance_builder: Optional callable receiving the selected experiment id and returning
            the provenance block (seed, checkpoint, SHA-256, config, hashes, timestamp) that is
            merged into the persisted document.

    Returns:
        Dictionary containing winning model selection and ranking.

    Raises:
        ValueError: If no summaries are provided.
        IncompleteRunError: If a summary lacks a required field (never defaulted to zero).
    """
    if not all_summaries:
        raise ValueError("Cannot select final model: empty summaries list.")

    problems = []
    candidates: list[dict[str, Any]] = []
    for s in all_summaries:
        exp_id = s["experiment_id"]
        problems.extend(find_numeric_problems(s, exp_id, ("phase", *REQUIRED_SUMMARY_FIELDS), True))
        if problems:
            continue
        macro_f1_mean = float(s["val_macro_f1_mean"])
        macro_f1_std = float(s["val_macro_f1_std"])
        candidates.append(
            {
                "experiment_id": exp_id,
                "phase": int(s["phase"]),
                "val_macro_f1_mean": macro_f1_mean,
                "val_macro_f1_std": macro_f1_std,
                "val_f1_mean": macro_f1_mean,
                "val_f1_std": macro_f1_std,
                "val_balanced_accuracy_mean": float(s["val_balanced_accuracy_mean"]),
                "val_balanced_accuracy_std": float(s["val_balanced_accuracy_std"]),
                "val_loss_mean": float(s["val_loss_mean"]),
                "mean_training_time_sec": float(s["total_training_time_sec_mean"]),
                "total_parameters": int(s["total_parameters"]),
            }
        )
    if problems:
        raise IncompleteRunError(problems)

    # Sort strictly by:
    # 1. -val_macro_f1_mean
    # 2. -val_balanced_accuracy_mean
    # 3. val_macro_f1_std (lower is better)
    # 4. mean_training_time_sec (lower is better)
    # 5. total_parameters (lower is better)
    ranked = sorted(
        candidates,
        key=lambda c: (
            -c["val_macro_f1_mean"],
            -c["val_balanced_accuracy_mean"],
            c["val_macro_f1_std"],
            c["mean_training_time_sec"],
            c["total_parameters"],
        ),
    )

    top_candidate = ranked[0]
    selected_candidate = top_candidate
    rule_applied = (
        "Direct validation performance ranking (Macro F1 -> Bal Acc -> Std -> Time -> Params)"
    )
    parsimony_triggered = False
    f1_diff = 0.0

    # Check parsimony against simpler alternatives (fewer parameters)
    simpler = [c for c in ranked if c["total_parameters"] < top_candidate["total_parameters"]]
    if simpler:
        best_simpler = simpler[0]
        f1_diff = top_candidate["val_macro_f1_mean"] - best_simpler["val_macro_f1_mean"]
        if f1_diff <= parsimony_threshold:
            selected_candidate = best_simpler
            parsimony_triggered = True
            rule_applied = (
                f"Parsimony principle applied: lead of {top_candidate['experiment_id']} ({f1_diff:.5f} Macro F1) "
                f"is within threshold ({parsimony_threshold:.5f}). "
                f"Selected more parsimonious model: {best_simpler['experiment_id']}."
            )
            logger.info(rule_applied)

    logger.info(
        f"Selected final model: '{selected_candidate['experiment_id']}' "
        f"(Val Macro F1: {selected_candidate['val_macro_f1_mean']:.4f})"
    )

    selection_payload: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "rule_applied": rule_applied,
        "parsimony_threshold": parsimony_threshold,
        "parsimony_triggered": parsimony_triggered,
        "top_f1_difference": f1_diff,
        "selected_experiment_id": selected_candidate["experiment_id"],
        "selected_candidate_metrics": selected_candidate,
        "all_candidates_ranking": ranked,
        "dataset_hash": dataset_hash,
        "split_hash": split_hash,
    }
    if provenance_builder is not None:
        provenance = provenance_builder(str(selected_candidate["experiment_id"]))
        selection_payload.update(provenance)
    selection_payload["selection_hash"] = compute_object_hash(selection_payload)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selection_payload, f, indent=2)

    logger.info(f"Saved final model selection to {output_path}")
    return selection_payload
