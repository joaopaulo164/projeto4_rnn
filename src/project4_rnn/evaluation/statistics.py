"""Statistical aggregation and summary across experimental seeds."""

import logging
from typing import Any

import numpy as np
import pandas as pd

from project4_rnn.experiments.timing import (
    TIMING_FIELDS,
    IncompleteRunError,
    RunProblem,
    find_numeric_problems,
    find_timing_problems,
)

logger = logging.getLogger("project4_rnn.evaluation.statistics")

REQUIRED_RUN_FIELDS: tuple[str, ...] = (
    "phase",
    "seed",
    "best_epoch",
    "total_epochs",
    "total_parameters",
    "trainable_parameters",
    "best_val_loss",
)

REQUIRED_VALIDATION_METRICS: tuple[str, ...] = ("macro_f1", "balanced_accuracy", "loss")


def find_run_problems(run: dict[str, Any], run_id: str) -> list[RunProblem]:
    """All defects of a run summary that would corrupt aggregation (never defaulted to zero)."""
    problems = find_numeric_problems(run, run_id, REQUIRED_RUN_FIELDS)
    problems.extend(find_timing_problems(run, run_id))
    metrics = run.get("validation_metrics")
    if not isinstance(metrics, dict):
        problems.append(RunProblem(run_id, "validation_metrics", "is missing or not a mapping"))
    else:
        problems.extend(
            find_numeric_problems(metrics, run_id, REQUIRED_VALIDATION_METRICS, allow_negative=True)
        )
    return problems


def summarize_seed_runs(
    experiment_id: str,
    phase: int,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate metrics across seeds for a single experiment configuration.

    Args:
        experiment_id: Experiment ID string (e.g. 'E01_MLP_32').
        phase: Phase integer.
        runs: List of dictionary summaries for each seed run.

    Returns:
        Consolidated summary dictionary with mean, std, min, max per metric.

    Raises:
        IncompleteRunError: If any run lacks a required field or has an invalid value. Missing
            values are never replaced by zero.
    """
    if not runs:
        return {
            "experiment_id": experiment_id,
            "phase": phase,
            "completed_runs": 0,
            "failed_runs": 0,
        }

    problems: list[RunProblem] = []
    for r in runs:
        run_id = f"{experiment_id}/seed_{r.get('seed', '?')}"
        problems.extend(find_run_problems(r, run_id))
    if problems:
        raise IncompleteRunError(problems)

    completed_runs = len(runs)

    # Flatten metric dictionaries
    flat_records: list[dict[str, float]] = []
    for r in runs:
        row: dict[str, float] = {
            "seed": float(r["seed"]),
            "best_epoch": float(r["best_epoch"]),
            "total_epochs": float(r["total_epochs"]),
            "total_parameters": float(r["total_parameters"]),
            "trainable_parameters": float(r["trainable_parameters"]),
            "best_val_loss": float(r["best_val_loss"]),
        }
        for name in TIMING_FIELDS:
            row[name] = float(r[name])
        for k, v in r["validation_metrics"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                row[f"val_{k}"] = float(v)
        flat_records.append(row)

    df = pd.DataFrame(flat_records)

    summary: dict[str, Any] = {
        "experiment_id": experiment_id,
        "phase": phase,
        "completed_runs": completed_runs,
        "failed_runs": 0,
        "total_parameters": int(df["total_parameters"].iloc[0]) if not df.empty else 0,
        "trainable_parameters": int(df["trainable_parameters"].iloc[0]) if not df.empty else 0,
    }

    numeric_cols = [c for c in df.columns if c not in ("seed",)]
    for col in numeric_cols:
        values = np.asarray(df[col].dropna().to_numpy(), dtype=np.float64)
        if len(values) > 0:
            summary[f"{col}_mean"] = float(np.mean(values))
            summary[f"{col}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            summary[f"{col}_min"] = float(np.min(values))
            summary[f"{col}_max"] = float(np.max(values))

    return summary


def create_summary_dataframe(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    """Create structured summary DataFrame from list of aggregated experiment summaries."""
    return pd.DataFrame(summaries)
