"""Automatic pooling strategy selection based on Phase 2 validation results and parsimony rule."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from project4_rnn.constants import PARSIMONY_THRESHOLD
from project4_rnn.reproducibility import compute_object_hash

logger = logging.getLogger("project4_rnn.experiments.pooling_selection")


def select_best_pooling(
    phase2_summaries: list[dict[str, Any]],
    output_path: Path,
    parsimony_threshold: float = PARSIMONY_THRESHOLD,
    data_hash: str | None = None,
) -> dict[str, Any]:
    """Execute pre-defined decision rule to select winning temporal pooling strategy from Phase 2.

    Selection hierarchy:
        1. Higher mean validation balanced accuracy.
        2. Higher mean validation F1 score (1st tie-breaker).
        3. Lower validation balanced accuracy standard deviation (2nd tie-breaker).
        4. Lower mean training time in seconds (3rd tie-breaker).
        5. Parsimony rule: if difference in bal acc <= parsimony_threshold (0.005),
           prefer the simpler model (fewer parameters).

    Args:
        phase2_summaries: List of summary dictionaries for Phase 2 experiments (E02-E05).
        output_path: Target path for pooling_selection.json.
        parsimony_threshold: Parsimony margin (default 0.005 = 0.5%).
        data_hash: Optional SHA-256 hash of processed dataset.

    Returns:
        Dictionary containing the full selection rationale, ranking, and chosen pooling.

    Raises:
        ValueError: If phase2_summaries is empty or missing required candidate data.
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

    for s in phase2_summaries:
        exp_id = s["experiment_id"]
        if exp_id not in pooling_map:
            continue
        pooling_strat = pooling_map[exp_id]
        cand = {
            "experiment_id": exp_id,
            "pooling_strategy": pooling_strat,
            "val_balanced_accuracy_mean": float(s.get("val_balanced_accuracy_mean", 0.0)),
            "val_balanced_accuracy_std": float(s.get("val_balanced_accuracy_std", 0.0)),
            "val_f1_mean": float(s.get("val_f1_mean", 0.0)),
            "val_f1_std": float(s.get("val_f1_std", 0.0)),
            "val_loss_mean": float(s.get("val_loss_mean", 0.0)),
            "mean_training_time_sec": float(s.get("total_training_time_sec_mean", 0.0)),
            "total_parameters": int(s.get("total_parameters", 3426)),
            "is_parsimonious": (s.get("total_parameters", 3426) == 3426),
        }
        candidates.append(cand)

    if not candidates:
        raise ValueError("No valid Phase 2 candidate experiments found in summaries.")

    # Sort strictly by hierarchical criteria:
    # 1. -val_balanced_accuracy_mean
    # 2. -val_f1_mean
    # 3. val_balanced_accuracy_std (lower is better)
    # 4. mean_training_time_sec (lower is better)
    # 5. total_parameters (lower is better)
    sorted_candidates = sorted(
        candidates,
        key=lambda c: (
            -c["val_balanced_accuracy_mean"],
            -c["val_f1_mean"],
            c["val_balanced_accuracy_std"],
            c["mean_training_time_sec"],
            c["total_parameters"],
        ),
    )

    top_candidate = sorted_candidates[0]
    selected_candidate = top_candidate
    rule_applied = "Direct performance ranking (Bal Acc -> F1 -> Std -> Time -> Params)"

    # Apply parsimony threshold if top candidate is complex (mean_max) and simpler alternative exists
    if top_candidate["pooling_strategy"] == "mean_max":
        # Find best simpler candidate (parameters == 3426)
        simpler_candidates = [
            c
            for c in sorted_candidates
            if c["total_parameters"] < top_candidate["total_parameters"]
        ]
        if simpler_candidates:
            best_simpler = simpler_candidates[0]
            diff = (
                top_candidate["val_balanced_accuracy_mean"]
                - best_simpler["val_balanced_accuracy_mean"]
            )
            if diff < parsimony_threshold:
                selected_candidate = best_simpler
                rule_applied = (
                    f"Parsimony principle applied: mean_max lead of {diff:.5f} is within "
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
        "selected_experiment_id": selected_candidate["experiment_id"],
        "selected_pooling": selected_candidate["pooling_strategy"],
        "selected_candidate_metrics": selected_candidate,
        "all_candidates_ranking": sorted_candidates,
        "data_hash": data_hash or "",
    }
    selection_payload["selection_hash"] = compute_object_hash(selection_payload)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selection_payload, f, indent=2)

    logger.info(f"Saved pooling selection decision to {output_path}")
    return selection_payload


def load_selected_pooling(selection_path: Path) -> str:
    """Load previously selected pooling strategy from JSON file.

    Args:
        selection_path: Path to pooling_selection.json.

    Returns:
        Selected pooling string ('last', 'mean', 'max', 'mean_max').

    Raises:
        FileNotFoundError: If pooling_selection.json is missing.
    """
    selection_path = Path(selection_path)
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"Required pooling selection file not found: {selection_path}. "
            "Phase 2 must be executed and consolidated before proceeding to Phase 3/4."
        )
    with open(selection_path, encoding="utf-8") as f:
        data = json.load(f)
    pooling = str(data["selected_pooling"])
    logger.info(f"Loaded selected pooling '{pooling}' from {selection_path}")
    return pooling
