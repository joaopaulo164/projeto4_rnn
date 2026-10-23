"""Selective cache invalidation for experiments, phases, and downstream dependencies."""

import logging
from pathlib import Path

logger = logging.getLogger("project4_rnn.orchestration.invalidation")


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

    # Invalidate summaries that depend on experiments
    summary_dir = artifacts_dir / "summary"
    for stale_file in [
        "experiments_summary.csv",
        "experiments_summary.json",
        "metrics_by_seed.csv",
        "validation_ranking.csv",
        "final_model_selection.json",
        "final_test_metrics.json",
        "article_evidence_catalog.md",
    ]:
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
    (summary_dir / "final_model_selection.json").unlink(missing_ok=True)
    (summary_dir / "final_test_metrics.json").unlink(missing_ok=True)

    # Invalidate Phase 3 and Phase 4 runs
    downstream_experiments = [
        "E06_LSTM_32_BEST_POOLING",
        "E07_GRU_64_BEST_POOLING",
        "E08_LSTM_64_BEST_POOLING",
    ]
    for exp_id in downstream_experiments:
        invalidate_experiment(exp_id, artifacts_dir)
