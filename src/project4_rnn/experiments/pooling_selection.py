"""Automatic pooling strategy selection based on Phase 2 validation results and parsimony rule for ElectricDevices."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from project4_rnn.constants import PARSIMONY_THRESHOLD
from project4_rnn.experiments.provenance import check_identity
from project4_rnn.experiments.timing import IncompleteRunError, find_numeric_problems
from project4_rnn.reproducibility import compute_object_hash

logger = logging.getLogger("project4_rnn.experiments.pooling_selection")

REQUIRED_SUMMARY_FIELDS: tuple[str, ...] = (
    "val_macro_f1_mean",
    "val_macro_f1_std",
    "val_balanced_accuracy_mean",
    "val_balanced_accuracy_std",
    "val_loss_mean",
    "total_training_time_sec_mean",
    "total_parameters",
)


def select_best_pooling(
    phase2_summaries: list[dict[str, Any]],
    output_path: Path,
    dataset_hash: str,
    split_hash: str,
    parsimony_threshold: float = PARSIMONY_THRESHOLD,
) -> dict[str, Any]:
    """Execute pre-defined decision rule to select winning temporal pooling strategy from Phase 2.

    Selection hierarchy:
        1. Higher mean validation Macro F1 score.
        2. Higher mean validation balanced accuracy (1st tie-breaker).
        3. Lower validation Macro F1 standard deviation (2nd tie-breaker).
        4. Lower mean training time in seconds (3rd tie-breaker).
        5. Parsimony rule: if difference in Macro F1 <= parsimony_threshold (0.005),
           prefer the simpler model (fewer parameters).

    Args:
        phase2_summaries: List of summary dictionaries for Phase 2 experiments (E02-E05).
        output_path: Target path for pooling_selection.json.
        dataset_hash: Identity hash of the dataset the Phase 2 runs used.
        split_hash: Identity hash of the train/validation split the Phase 2 runs used.
        parsimony_threshold: Parsimony margin (default 0.005 = 0.5% Macro F1).

    Returns:
        Dictionary containing the full selection rationale, ranking, and chosen pooling.

    Raises:
        ValueError: If phase2_summaries is empty or has no Phase 2 candidates.
        IncompleteRunError: If a candidate summary lacks a required field (never defaulted).
    """
    if not phase2_summaries:
        raise ValueError("Cannot select pooling: no Phase 2 experiment summaries provided.")

    candidates: list[dict[str, Any]] = []
    pooling_map = {
        "E02_GRU_32_LAST": "last",
        "E03_GRU_32_MEAN": "mean",
        "E04_GRU_32_MAX": "max",
        "E05_GRU_32_MEAN_MAX": "mean_max",
    }

    problems = []
    for s in phase2_summaries:
        exp_id = s["experiment_id"]
        if exp_id not in pooling_map:
            continue
        problems.extend(find_numeric_problems(s, exp_id, REQUIRED_SUMMARY_FIELDS, True))
        if problems:
            continue
        pooling_strat = pooling_map[exp_id]
        macro_f1_mean = float(s["val_macro_f1_mean"])
        macro_f1_std = float(s["val_macro_f1_std"])
        cand = {
            "experiment_id": exp_id,
            "pooling_strategy": pooling_strat,
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
        candidates.append(cand)
    if problems:
        raise IncompleteRunError(problems)

    if not candidates:
        raise ValueError("No valid Phase 2 candidate experiments found in summaries.")

    # Sort strictly by hierarchical criteria:
    # 1. -val_macro_f1_mean
    # 2. -val_balanced_accuracy_mean
    # 3. val_macro_f1_std (lower is better)
    # 4. mean_training_time_sec (lower is better)
    # 5. total_parameters (lower is better)
    sorted_candidates = sorted(
        candidates,
        key=lambda c: (
            -c["val_macro_f1_mean"],
            -c["val_balanced_accuracy_mean"],
            c["val_macro_f1_std"],
            c["mean_training_time_sec"],
            c["total_parameters"],
        ),
    )

    top_candidate = sorted_candidates[0]
    selected_candidate = top_candidate
    rule_applied = "Direct performance ranking (Macro F1 -> Bal Acc -> Std -> Time -> Params)"
    parsimony_triggered = False
    f1_diff = 0.0

    # Apply parsimony threshold if top candidate is complex (mean_max) and simpler alternative exists
    if top_candidate["pooling_strategy"] == "mean_max":
        simpler_candidates = [
            c
            for c in sorted_candidates
            if c["total_parameters"] < top_candidate["total_parameters"]
        ]
        if simpler_candidates:
            best_simpler = simpler_candidates[0]
            f1_diff = top_candidate["val_macro_f1_mean"] - best_simpler["val_macro_f1_mean"]
            if f1_diff <= parsimony_threshold:
                selected_candidate = best_simpler
                parsimony_triggered = True
                rule_applied = (
                    f"Parsimony principle applied: mean_max lead of {f1_diff:.5f} Macro F1 is within "
                    f"threshold ({parsimony_threshold:.5f}). Selected simpler alternative: {best_simpler['pooling_strategy']}."
                )
                logger.info(rule_applied)

    logger.info(
        f"Selected winning pooling: '{selected_candidate['pooling_strategy']}' "
        f"from experiment '{selected_candidate['experiment_id']}'"
    )

    selection_payload: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "rule_applied": rule_applied,
        "parsimony_threshold": parsimony_threshold,
        "parsimony_triggered": parsimony_triggered,
        "top_f1_difference": f1_diff,
        "selected_experiment_id": selected_candidate["experiment_id"],
        "selected_pooling": selected_candidate["pooling_strategy"],
        "selected_candidate_metrics": selected_candidate,
        "all_candidates_ranking": sorted_candidates,
        "dataset_hash": dataset_hash,
        "split_hash": split_hash,
    }
    selection_payload["selection_hash"] = compute_object_hash(selection_payload)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selection_payload, f, indent=2)

    logger.info(f"Saved pooling selection decision to {output_path}")
    return selection_payload


def load_selected_pooling(
    selection_path: Path,
    expected_split_hash: str | None = None,
    expected_dataset_hash: str | None = None,
) -> str:
    """Load previously selected pooling strategy from JSON file.

    Args:
        selection_path: Path to pooling_selection.json.
        expected_split_hash: When given, the selection must have been made on this split.
        expected_dataset_hash: When given, the selection must have been made on this dataset.

    Returns:
        Selected pooling string ('last', 'mean', 'max', 'mean_max').

    Raises:
        FileNotFoundError: If pooling_selection.json is missing.
        StaleArtifactError: If the selection belongs to another split or dataset.
    """
    selection_path = Path(selection_path)
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"Required pooling selection file not found: {selection_path}. "
            "Phase 2 must be executed and consolidated before proceeding to Phase 3/4."
        )
    with open(selection_path, encoding="utf-8") as f:
        data = json.load(f)
    check_identity("pooling_selection.json", data, expected_dataset_hash, expected_split_hash)
    pooling = str(data["selected_pooling"])
    logger.info(f"Loaded selected pooling '{pooling}' from {selection_path}")
    return pooling
