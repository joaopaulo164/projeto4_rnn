"""Execution plan generation, CLI flag validation, and dry-run display."""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from project4_rnn.orchestration.dependency_graph import (
    DependencyGraph,
    build_standard_pipeline_graph,
)

logger = logging.getLogger("project4_rnn.orchestration.execution_plan")


@dataclass
class PlanStepItem:
    """Item in the formatted execution plan."""

    step_id: str
    phase: int | None
    description: str
    dependencies: list[str]
    expected_outputs: list[str]
    action: str  # 'EXECUTE', 'SKIP', 'RERUN'


@dataclass
class ExecutionPlan:
    """Full execution plan for a pipeline invocation."""

    steps: list[PlanStepItem]
    dry_run: bool = False
    resume: bool = False
    force: bool = False
    from_phase: int | None = None
    until_phase: int | None = None
    fail_fast: bool = True
    continue_on_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def print_plan(self) -> None:
        """Format and print execution plan to console."""
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION PLAN")
        print("=" * 80)
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE EXECUTION'}")
        print(f"Resume: {self.resume} | Force: {self.force}")
        print(f"Phase bounds: From={self.from_phase} -> Until={self.until_phase}")
        print(
            f"Error handling: Fail-fast={self.fail_fast} | Continue-on-error={self.continue_on_error}"
        )
        print("-" * 80)
        print(f"{'Step ID':<26} {'Phase':<8} {'Action':<10} {'Description'}")
        print("-" * 80)

        for step in self.steps:
            phase_str = str(step.phase) if step.phase is not None else "-"
            print(f"{step.step_id:<26} {phase_str:<8} {step.action:<10} {step.description}")

        print("=" * 80 + "\n")


def validate_execution_flags(
    resume: bool = False,
    force: bool = False,
    from_phase: int | None = None,
    until_phase: int | None = None,
    fail_fast: bool = True,
    continue_on_error: bool = False,
) -> None:
    """Validate mutual compatibility of pipeline execution options.

    Raises:
        ValueError: If mutually exclusive or invalid options are specified.
    """
    if resume and force:
        raise ValueError("Incompatible options: --resume and --force cannot be used together.")

    if fail_fast and continue_on_error:
        raise ValueError(
            "Incompatible options: --fail-fast and --continue-on-error cannot be used together."
        )

    if from_phase is not None and from_phase not in (1, 2, 3, 4):
        raise ValueError(f"Invalid --from-phase: {from_phase}. Must be 1, 2, 3, or 4.")

    if until_phase is not None and until_phase not in (1, 2, 3, 4):
        raise ValueError(f"Invalid --until-phase: {until_phase}. Must be 1, 2, 3, or 4.")

    if from_phase is not None and until_phase is not None and from_phase > until_phase:
        raise ValueError(
            f"Invalid phase range: --from-phase ({from_phase}) is greater than --until-phase ({until_phase})."
        )


def build_execution_plan(
    dag: DependencyGraph | None = None,
    dry_run: bool = False,
    resume: bool = False,
    force: bool = False,
    from_phase: int | None = None,
    until_phase: int | None = None,
    fail_fast: bool = True,
    continue_on_error: bool = False,
    artifacts_dir: Path = Path("artifacts"),
) -> ExecutionPlan:
    """Construct an ordered ExecutionPlan based on dependency graph and runtime flags."""
    validate_execution_flags(
        resume=resume,
        force=force,
        from_phase=from_phase,
        until_phase=until_phase,
        fail_fast=fail_fast,
        continue_on_error=continue_on_error,
    )

    if dag is None:
        dag = build_standard_pipeline_graph()

    ordered_step_ids = dag.get_topological_order()
    plan_items: list[PlanStepItem] = []

    for step_id in ordered_step_ids:
        step = dag.get_step(step_id)

        # Filter by phase bounds if step has a phase
        if step.phase is not None:
            if from_phase is not None and step.phase < from_phase:
                continue
            if until_phase is not None and step.phase > until_phase:
                continue

        # Determine action (EXECUTE, SKIP, RERUN)
        action = "EXECUTE"
        if force:
            action = "RERUN"
        elif resume:
            # Check if all expected outputs exist
            outputs_exist = all((Path(p).is_file()) for p in step.expected_outputs)
            if outputs_exist and step.expected_outputs:
                action = "SKIP"

        item = PlanStepItem(
            step_id=step.step_id,
            phase=step.phase,
            description=step.description,
            dependencies=step.dependencies,
            expected_outputs=step.expected_outputs,
            action=action,
        )
        plan_items.append(item)

    return ExecutionPlan(
        steps=plan_items,
        dry_run=dry_run,
        resume=resume,
        force=force,
        from_phase=from_phase,
        until_phase=until_phase,
        fail_fast=fail_fast,
        continue_on_error=continue_on_error,
    )
