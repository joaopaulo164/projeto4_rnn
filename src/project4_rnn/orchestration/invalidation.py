"""Selective cache invalidation for experiments, phases, and downstream dependencies.

Besides explicit invalidation, results are tied to the dataset/split identity: an artifact whose
recorded ``split_hash`` differs from (or lacks) the current split hash is *stale* and must never be
reused. Stale results can be listed and flagged as obsolete without deleting them.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("project4_rnn.orchestration.invalidation")

OBSOLETE_MARKER_FILE = "OBSOLETE_ARTIFACTS.json"

# Summary files that depend on trained runs and therefore on the split.
SPLIT_DEPENDENT_SUMMARY_FILES: tuple[str, ...] = (
    "pooling_selection.json",
    "final_model_selection.json",
    "final_test_metrics.json",
    "final_test_metrics_no_exact_overlap.json",
    "final_test_metrics_without_label_conflicts.json",
)

# Derived summary files removed whenever any experiment is invalidated.
DERIVED_SUMMARY_FILES: tuple[str, ...] = (
    "experiments_summary.csv",
    "experiments_summary.json",
    "metrics_by_seed.csv",
    "validation_ranking.csv",
    "pooling_ablation.csv",
    "consolidation_status.json",
    "final_model_selection.json",
    "final_test_metrics.json",
    "final_test_metrics_no_exact_overlap.json",
    "final_test_metrics_without_label_conflicts.json",
    "final_evaluation_sensitivity_comparison.csv",
    "final_evaluation_sensitivity_comparison.md",
    "article_evidence_catalog.md",
)


def invalidate_experiment(
    experiment_id: str,
    artifacts_dir: Path = Path("artifacts"),
) -> None:
    """Invalidate all artifacts for a specific experiment and dependent summary files."""
    artifacts_dir = Path(artifacts_dir)
    exp_dir = artifacts_dir / "runs" / experiment_id

    if exp_dir.is_dir():
        logger.info(f"Invalidating run artifacts in {exp_dir}")
        for status_file in exp_dir.glob("**/status.json"):
            status_file.unlink(missing_ok=True)

    summary_dir = artifacts_dir / "summary"
    for stale_file in DERIVED_SUMMARY_FILES:
        (summary_dir / stale_file).unlink(missing_ok=True)

    logger.info(f"Invalidated downstream artifacts for {experiment_id}")


def invalidate_pooling_and_downstream(
    artifacts_dir: Path = Path("artifacts"),
) -> None:
    """Invalidate Phase 3, Phase 4, and all final summary artifacts when pooling changes."""
    artifacts_dir = Path(artifacts_dir)
    logger.info(
        "Invalidating pooling selection and all downstream phases (Phase 3, Phase 4, Final Eval)"
    )

    summary_dir = artifacts_dir / "summary"
    (summary_dir / "pooling_selection.json").unlink(missing_ok=True)

    downstream_experiments = [
        "E06_LSTM_32_BEST_POOLING",
        "E07_GRU_64_BEST_POOLING",
        "E08_LSTM_64_BEST_POOLING",
    ]
    for exp_id in downstream_experiments:
        invalidate_experiment(exp_id, artifacts_dir)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def find_stale_runs(artifacts_dir: Path, split_hash: str) -> list[Path]:
    """Run directories whose status.json lacks the current split hash (trained on another split)."""
    runs_dir = Path(artifacts_dir) / "runs"
    stale: list[Path] = []
    if not runs_dir.is_dir():
        return stale
    for status_file in sorted(runs_dir.glob("*/*/status.json")):
        status = _read_json(status_file)
        if status is None or status.get("split_hash") != split_hash:
            stale.append(status_file.parent)
    return stale


def find_stale_summaries(artifacts_dir: Path, split_hash: str) -> list[Path]:
    """Split-dependent summary files that lack the current split hash."""
    summary_dir = Path(artifacts_dir) / "summary"
    stale: list[Path] = []
    for name in SPLIT_DEPENDENT_SUMMARY_FILES:
        path = summary_dir / name
        if not path.is_file():
            continue
        payload = _read_json(path)
        if payload is None or payload.get("split_hash") != split_hash:
            stale.append(path)
    return stale


def mark_artifacts_obsolete(artifacts_dir: Path, split_hash: str, reason: str) -> Path | None:
    """Flag incompatible results as obsolete without deleting them.

    Writes ``OBSOLETE_ARTIFACTS.json`` listing stale runs and summaries. Returns the marker path,
    or None when nothing is stale.
    """
    artifacts_dir = Path(artifacts_dir)
    runs = find_stale_runs(artifacts_dir, split_hash)
    summaries = find_stale_summaries(artifacts_dir, split_hash)
    if not runs and not summaries:
        return None
    marker = artifacts_dir / OBSOLETE_MARKER_FILE
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "current_split_hash": split_hash,
        "reason": reason,
        "obsolete_runs": [str(p) for p in runs],
        "obsolete_summaries": [str(p) for p in summaries],
        "action": "Run 'clean-artifacts --generated-only' before the next full experiment.",
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    with open(marker, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.warning(
        f"{len(runs)} run(s) and {len(summaries)} summary file(s) are incompatible with the "
        f"current split and were flagged obsolete in {marker}"
    )
    return marker
