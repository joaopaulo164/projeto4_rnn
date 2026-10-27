"""Full end-to-end pipeline orchestrator for reproducible scientific study."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from project4_rnn.audit.duplicates import run_duplicate_audit
from project4_rnn.config import build_experiment_config
from project4_rnn.orchestration.dependency_graph import (
    DependencyGraph,
    build_standard_pipeline_graph,
)
from project4_rnn.orchestration.execution_plan import (
    ExecutionPlan,
    build_execution_plan,
)
from project4_rnn.orchestration.invalidation import mark_artifacts_obsolete
from project4_rnn.orchestration.manifest import PipelineManifest
from project4_rnn.orchestration.phase_runner import (
    prepare_and_load_data,
    run_final_evaluation,
    run_phase_1,
    run_phase_2,
    run_phase_3,
    run_phase_4,
    run_sensitivity_analysis,
)

logger = logging.getLogger("project4_rnn.orchestration.pipeline")


class StepFailedError(RuntimeError):
    """Raised when a pipeline step failed and execution must stop."""


def run_full_pipeline(
    base_config_path: Path = Path("configs/base.yaml"),
    artifacts_dir: Path = Path("artifacts"),
    dry_run: bool = False,
    resume: bool = False,
    force: bool = False,
    from_phase: int | None = None,
    until_phase: int | None = None,
    fail_fast: bool = True,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    """Execute the complete scientific study pipeline.

    Args:
        base_config_path: Path to configs/base.yaml.
        artifacts_dir: Target directory for runs, summaries, and audits.
        dry_run: If True, prints the execution plan and exits. A dry run trains nothing, opens no
            checkpoint, evaluates no test data, and writes no artifact.
        resume: If True, reuses existing completed valid artifacts.
        force: If True, forces re-execution of all steps.
        from_phase: Optional starting phase (1-4).
        until_phase: Optional ending phase (1-4).
        fail_fast: If True, halts immediately on first step failure.
        continue_on_error: If True, continues independent steps on failure.

    Returns:
        Pipeline execution result dictionary.
    """
    artifacts_dir = Path(artifacts_dir)

    # 1. Build and validate the execution plan (pure: reads configuration only)
    plan: ExecutionPlan = build_execution_plan(
        dry_run=dry_run,
        resume=resume,
        force=force,
        from_phase=from_phase,
        until_phase=until_phase,
        fail_fast=fail_fast,
        continue_on_error=continue_on_error,
        artifacts_dir=artifacts_dir,
        base_config_path=base_config_path,
    )

    if dry_run:
        plan.print_plan()
        logger.info("Dry run completed successfully. No training or evaluation was executed.")
        return {"status": "DRY_RUN_COMPLETED", "plan": plan.to_dict()}

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 2. Initialize provenance manifest
    manifest = PipelineManifest(artifacts_dir=artifacts_dir)
    base_cfg = build_experiment_config(base_config_path)
    manifest.save_initial_state(
        resolved_config=base_cfg.to_dict(),
        execution_plan=plan.to_dict(),
    )
    logger.info(f"========== INITIATING PIPELINE RUN ({manifest.run_id}) ==========")

    dag: DependencyGraph = build_standard_pipeline_graph()
    failed: set[str] = set()
    state: dict[str, Any] = {"manifest_data": None, "final_test_metrics": None}

    def upstream(step_id: str) -> set[str]:
        found: set[str] = set()
        for dep in dag.get_step(step_id).dependencies:
            found.add(dep)
            found.update(upstream(dep))
        return found

    def execute(step_id: str, action: Callable[[], Any]) -> Any:
        """Run a step unless an upstream step failed; record the outcome in the manifest."""
        blocked = sorted(upstream(step_id) & failed)
        if blocked:
            logger.warning(f"Skipping '{step_id}': upstream step(s) failed: {blocked}")
            failed.add(step_id)
            return None
        try:
            result = action()
        except Exception as exc:
            logger.error(f"Step '{step_id}' failed: {exc}", exc_info=True)
            manifest.record_step_failed(step_id, str(exc))
            failed.add(step_id)
            if fail_fast and not continue_on_error:
                raise StepFailedError(f"Step '{step_id}' failed: {exc}") from exc
            return None
        manifest.record_step_completed(step_id)
        return result

    def data_step() -> None:
        bundle = prepare_and_load_data(
            base_config_path=base_config_path, force=force, artifacts_dir=artifacts_dir
        )
        state["manifest_data"] = bundle.manifest
        mark_artifacts_obsolete(
            artifacts_dir,
            bundle.split_hash,
            reason="runs/summaries lack the current hash-grouped split hash",
        )

    def phase_step(runner: Callable[..., Any]) -> Callable[[], Any]:
        def _run() -> Any:
            return runner(
                base_config_path=base_config_path, artifacts_dir=artifacts_dir, force=force
            )

        return _run

    def final_step() -> None:
        final_res = run_final_evaluation(
            base_config_path=base_config_path, artifacts_dir=artifacts_dir
        )
        state["final_test_metrics"] = final_res.get("test_metrics")

    try:
        execute("data_preparation", data_step)
        execute(
            "duplicate_audit",
            lambda: run_duplicate_audit(base_config_path, artifacts_dir / "audits"),
        )

        phase_runners = ((1, run_phase_1), (2, run_phase_2), (3, run_phase_3), (4, run_phase_4))
        for phase, runner in phase_runners:
            if not should_execute_phase(phase, from_phase, until_phase):
                continue
            execute(f"phase_{phase}", phase_step(runner))
            if phase == 2:
                if "phase_2" in failed:
                    failed.add("pooling_selection")
                else:
                    manifest.record_step_completed("pooling_selection")

        if until_phase is None or until_phase >= 4:
            execute("final_test_evaluation", final_step)
            if "final_test_evaluation" not in failed:
                for done in ("consolidation", "final_model_selection", "compliance_audit"):
                    manifest.record_step_completed(done)
            execute(
                "sensitivity_evaluation",
                lambda: run_sensitivity_analysis(base_config_path, artifacts_dir),
            )
            if "sensitivity_evaluation" not in failed:
                manifest.record_step_completed("evidence_generation")

        status = "FAILED" if failed else "COMPLETED"
        report_path = manifest.finalize(
            status=status,
            dataset_manifest=state["manifest_data"],
            final_test_metrics=state["final_test_metrics"],
        )
        logger.info(f"========== PIPELINE RUN {status} ({manifest.run_id}) ==========")
        return {
            "status": status,
            "pipeline_run_id": manifest.run_id,
            "report_path": str(report_path),
            "failed_steps": sorted(failed),
            "final_test_metrics": state["final_test_metrics"],
        }
    except StepFailedError:
        manifest.finalize(status="FAILED", dataset_manifest=state["manifest_data"])
        raise


def should_execute_phase(
    phase: int,
    from_phase: int | None,
    until_phase: int | None,
) -> bool:
    """Check if a specific phase is within requested execution bounds."""
    if from_phase is not None and phase < from_phase:
        return False
    if until_phase is not None and phase > until_phase:
        return False
    return True
