"""Full end-to-end pipeline orchestrator for reproducible scientific study."""

import logging
from pathlib import Path
from typing import Any

from project4_rnn.config import build_experiment_config
from project4_rnn.orchestration.execution_plan import (
    ExecutionPlan,
    build_execution_plan,
)
from project4_rnn.orchestration.manifest import PipelineManifest
from project4_rnn.orchestration.phase_runner import (
    prepare_and_load_data,
    run_final_evaluation,
    run_phase_1,
    run_phase_2,
    run_phase_3,
    run_phase_4,
)

logger = logging.getLogger("project4_rnn.orchestration.pipeline")


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
    """Execute complete 20-step scientific study pipeline.

    Args:
        base_config_path: Path to configs/base.yaml.
        artifacts_dir: Target directory for runs, summaries, and audits.
        dry_run: If True, prints execution plan and exits without running.
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
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build and validate execution plan
    plan: ExecutionPlan = build_execution_plan(
        dry_run=dry_run,
        resume=resume,
        force=force,
        from_phase=from_phase,
        until_phase=until_phase,
        fail_fast=fail_fast,
        continue_on_error=continue_on_error,
        artifacts_dir=artifacts_dir,
    )

    if dry_run:
        plan.print_plan()
        logger.info("Dry run completed successfully. No training was executed.")
        return {"status": "DRY_RUN_COMPLETED", "plan": plan.to_dict()}

    # 2. Initialize Provenance Manifest
    manifest = PipelineManifest(artifacts_dir=artifacts_dir)
    base_cfg = build_experiment_config(base_config_path)
    manifest.save_initial_state(
        resolved_config=base_cfg.to_dict(),
        execution_plan=plan.to_dict(),
    )

    logger.info(f"========== INITIATING PIPELINE RUN ({manifest.run_id}) ==========")

    val_report: dict[str, Any] | None = None
    final_test_metrics: dict[str, Any] | None = None

    try:
        # Step 1 & 2: Environment and Data Preparation
        logger.info("[Step 1-6] Environment validation & Data preparation...")
        _, _, _, _, _, _, _, val_report = prepare_and_load_data(
            base_config_path=base_config_path,
            force=force,
        )
        manifest.record_step_completed("data_preparation")
        manifest.record_step_completed("data_validation")

        # Step 7: Phase 1 (MLP-32 Baseline)
        if should_execute_phase(1, from_phase, until_phase):
            logger.info("[Step 7] Executing Phase 1: MLP-32 Baseline...")
            run_phase_1(base_config_path=base_config_path, artifacts_dir=artifacts_dir, force=force)
            manifest.record_step_completed("phase_1")

        # Step 8-10: Phase 2 & Pooling Selection
        if should_execute_phase(2, from_phase, until_phase):
            logger.info(
                "[Step 8-10] Executing Phase 2: GRU Pooling Ablation & Automatic Selection..."
            )
            run_phase_2(base_config_path=base_config_path, artifacts_dir=artifacts_dir, force=force)
            manifest.record_step_completed("phase_2")
            manifest.record_step_completed("pooling_selection")

        # Step 11: Phase 3 (LSTM-32 with Winning Pooling)
        if should_execute_phase(3, from_phase, until_phase):
            logger.info("[Step 11] Executing Phase 3: LSTM-32 with Best Pooling...")
            run_phase_3(base_config_path=base_config_path, artifacts_dir=artifacts_dir, force=force)
            manifest.record_step_completed("phase_3")

        # Step 12: Phase 4 (GRU-64 and LSTM-64 with Winning Pooling)
        if should_execute_phase(4, from_phase, until_phase):
            logger.info("[Step 12] Executing Phase 4: Capacity Scaling (GRU-64 & LSTM-64)...")
            run_phase_4(base_config_path=base_config_path, artifacts_dir=artifacts_dir, force=force)
            manifest.record_step_completed("phase_4")

        # Step 13-20: Final Evaluation, Consolidation, Evidence & Audit (if until_phase allows)
        if until_phase is None or until_phase >= 4:
            logger.info(
                "[Step 13-20] Executing Final Model Selection, Official Test Evaluation & Audit..."
            )
            final_res = run_final_evaluation(
                base_config_path=base_config_path,
                artifacts_dir=artifacts_dir,
            )
            final_test_metrics = final_res.get("test_metrics")
            manifest.record_step_completed("consolidation")
            manifest.record_step_completed("final_model_selection")
            manifest.record_step_completed("final_test_evaluation")
            manifest.record_step_completed("evidence_generation")
            manifest.record_step_completed("compliance_audit")

        # Finalize manifest
        report_path = manifest.finalize(
            status="COMPLETED",
            dataset_manifest=val_report,
            final_test_metrics=final_test_metrics,
        )

        logger.info(f"========== PIPELINE RUN SUCCESSFUL ({manifest.run_id}) ==========")
        return {
            "status": "COMPLETED",
            "pipeline_run_id": manifest.run_id,
            "report_path": str(report_path),
            "final_test_metrics": final_test_metrics,
        }

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        manifest.record_step_failed("pipeline_execution", str(e))
        manifest.finalize(status="FAILED", dataset_manifest=val_report)
        if fail_fast:
            raise
        return {"status": "FAILED", "error": str(e), "pipeline_run_id": manifest.run_id}


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
