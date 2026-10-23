"""Statistical aggregation and summary across experimental seeds."""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("project4_rnn.evaluation.statistics")


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
    """
    if not runs:
        return {
            "experiment_id": experiment_id,
            "phase": phase,
            "completed_runs": 0,
            "failed_runs": 0,
        }

    completed_runs = len(runs)

    # Flatten metric dictionaries
    flat_records: list[dict[str, float]] = []
    for r in runs:
        row: dict[str, float] = {
            "seed": float(r["seed"]),
            "best_epoch": float(r["best_epoch"]),
            "total_epochs": float(r.get("total_epochs", 0)),
            "total_training_time_sec": float(r.get("total_training_time_sec", 0.0)),
            "mean_epoch_time_sec": float(r.get("mean_epoch_time_sec", 0.0)),
            "inference_time_sec": float(r.get("inference_time_sec", 0.0)),
            "total_parameters": float(r.get("total_parameters", 0)),
            "trainable_parameters": float(r.get("trainable_parameters", 0)),
            "best_val_loss": float(r.get("best_val_loss", 0.0)),
        }
        val_m = r.get("validation_metrics", {})
        for k, v in val_m.items():
            if isinstance(v, (int, float)) and v is not None:
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
