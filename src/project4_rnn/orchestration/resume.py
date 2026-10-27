"""Pipeline reentrance and execution resume utilities."""

import logging
from pathlib import Path

from project4_rnn.orchestration.dependency_graph import PipelineStep, resolve_output_path

logger = logging.getLogger("project4_rnn.orchestration.resume")


def is_step_completed(
    step: PipelineStep,
    artifacts_dir: Path = Path("artifacts"),
) -> bool:
    """Check if all expected output artifacts for a step exist and are non-empty.

    ``artifacts/...`` outputs are resolved under ``artifacts_dir``; other paths (``data/...``)
    are used as declared.
    """
    if not step.expected_outputs:
        return False

    for out_path in step.expected_outputs:
        p = resolve_output_path(out_path, artifacts_dir)
        if not p.is_file() or p.stat().st_size == 0:
            return False

    return True


def evaluate_resumption_state(
    steps: list[PipelineStep],
    artifacts_dir: Path = Path("artifacts"),
) -> tuple[list[str], list[str]]:
    """Partition steps into completed and pending for resuming pipeline execution.

    Returns:
        Tuple of (completed_step_ids, pending_step_ids).
    """
    completed: list[str] = []
    pending: list[str] = []

    for step in steps:
        if is_step_completed(step, artifacts_dir):
            completed.append(step.step_id)
        else:
            pending.append(step.step_id)

    logger.info(
        f"Resumption state: {len(completed)} completed steps, {len(pending)} pending steps."
    )
    return completed, pending
