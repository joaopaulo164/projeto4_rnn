"""Strict validation of per-run timing fields and incomplete-run bookkeeping.

A missing, non-numeric, negative or non-finite timing field is never replaced by zero. A legitimate
measured value of exactly 0.0 is accepted. Problems are collected with the identity of the run so
the consolidation can be marked incomplete and halted.
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

TIMING_FIELDS: tuple[str, ...] = (
    "total_training_time_sec",
    "mean_epoch_time_sec",
    "inference_time_sec",
)

CONSOLIDATION_STATUS_FILE = "consolidation_status.json"

# Absolute tolerance (seconds) when comparing wall-clock aggregates.
TIMING_ABS_TOLERANCE = 1e-3
# Relative tolerance of mean_epoch_time_sec * total_epochs vs total_training_time_sec.
TIMING_REL_TOLERANCE = 1e-6


@dataclass(frozen=True)
class RunProblem:
    """A defect found in a single run's persisted summary."""

    run_id: str
    field: str
    problem: str

    def __str__(self) -> str:
        return f"{self.run_id}: field '{self.field}' {self.problem}"


class IncompleteRunError(RuntimeError):
    """Raised when one or more runs have missing or invalid summary fields."""

    def __init__(self, problems: Sequence[RunProblem]) -> None:
        self.problems = list(problems)
        runs = sorted({p.run_id for p in self.problems})
        details = "; ".join(str(p) for p in self.problems)
        super().__init__(
            f"Incomplete or invalid run summaries in {len(runs)} run(s) {runs}: {details}. "
            "Official consolidation halted; rerun the affected runs."
        )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def find_numeric_problems(
    run: Mapping[str, Any],
    run_id: str,
    fields: Sequence[str],
    allow_negative: bool = False,
) -> list[RunProblem]:
    """Report absent, non-numeric, non-finite (and optionally negative) fields of a run summary."""
    problems: list[RunProblem] = []
    for name in fields:
        if name not in run or run[name] is None:
            problems.append(RunProblem(run_id, name, "is missing"))
            continue
        value = run[name]
        if not _is_number(value):
            problems.append(
                RunProblem(
                    run_id, name, f"has invalid type {type(value).__name__} (expected number)"
                )
            )
            continue
        if not math.isfinite(float(value)):
            problems.append(RunProblem(run_id, name, f"is not finite ({value})"))
            continue
        if not allow_negative and float(value) < 0:
            problems.append(RunProblem(run_id, name, f"is negative ({value})"))
    return problems


def find_timing_problems(run: Mapping[str, Any], run_id: str) -> list[RunProblem]:
    """Timing-field defects of a run summary, including internal consistency."""
    problems = find_numeric_problems(run, run_id, TIMING_FIELDS)
    if problems:
        return problems

    total = float(run["total_training_time_sec"])
    mean_epoch = float(run["mean_epoch_time_sec"])
    epochs = run.get("total_epochs")
    if isinstance(epochs, (int, float)) and not isinstance(epochs, bool) and epochs > 0:
        expected_total = mean_epoch * float(epochs)
        if abs(expected_total - total) > TIMING_ABS_TOLERANCE + TIMING_REL_TOLERANCE * total:
            problems.append(
                RunProblem(
                    run_id,
                    "mean_epoch_time_sec",
                    f"inconsistent with total_training_time_sec ({mean_epoch} x {epochs} epochs "
                    f"!= {total})",
                )
            )
    return problems


def validate_timing_fields(run: Mapping[str, Any], run_id: str) -> dict[str, float]:
    """Return the validated timing fields of a run.

    Raises:
        IncompleteRunError: If any timing field is missing, invalid, or inconsistent.
    """
    problems = find_timing_problems(run, run_id)
    if problems:
        raise IncompleteRunError(problems)
    return {name: float(run[name]) for name in TIMING_FIELDS}


def find_history_consistency_problems(
    run: Mapping[str, Any],
    history: pd.DataFrame,
    run_id: str,
) -> list[RunProblem]:
    """Cross-check the summary timings against training_history.csv."""
    problems: list[RunProblem] = []
    if "epoch_time_sec" not in history.columns:
        return [RunProblem(run_id, "epoch_time_sec", "is missing from training_history.csv")]

    epochs = run.get("total_epochs")
    if (
        isinstance(epochs, (int, float))
        and not isinstance(epochs, bool)
        and int(epochs) != len(history)
    ):
        problems.append(
            RunProblem(
                run_id,
                "total_epochs",
                f"({epochs}) differs from training_history.csv rows ({len(history)})",
            )
        )

    total = run.get("total_training_time_sec")
    if isinstance(total, (int, float)) and not isinstance(total, bool):
        epoch_sum = float(history["epoch_time_sec"].sum())
        if epoch_sum > float(total) + TIMING_ABS_TOLERANCE:
            problems.append(
                RunProblem(
                    run_id,
                    "total_training_time_sec",
                    f"({total}) is smaller than the sum of epoch times ({epoch_sum})",
                )
            )
    return problems


def write_consolidation_status(
    summary_dir: Path,
    complete: bool,
    problems: Sequence[RunProblem] = (),
    runs_consolidated: int = 0,
) -> Path:
    """Persist whether the last official consolidation was complete."""
    summary_dir = Path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    path = summary_dir / CONSOLIDATION_STATUS_FILE
    payload = {
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "timestamp": datetime.now(UTC).isoformat(),
        "runs_consolidated": runs_consolidated,
        "problems": [{"run": p.run_id, "field": p.field, "problem": p.problem} for p in problems],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
