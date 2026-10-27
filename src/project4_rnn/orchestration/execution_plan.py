"""Execution plan generation, CLI flag validation, and dry-run display."""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from project4_rnn.config import build_experiment_config
from project4_rnn.experiments.registry import EXPERIMENT_REGISTRY
from project4_rnn.orchestration.dependency_graph import (
    DependencyGraph,
    build_standard_pipeline_graph,
    resolve_output_path,
)

logger = logging.getLogger("project4_rnn.orchestration.execution_plan")

FUTURE_EVALUATIONS: tuple[str, ...] = (
    "final_test_evaluation: frozen winning model on the full official test set (7,711)",
    "sensitivity_evaluation: test set without exact-overlap series (descriptive only)",
    "sensitivity_evaluation: test set without label-conflict groups (descriptive only)",
)


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
    experiments: list[dict[str, Any]] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    planned_runs: int = 0
    data_info: dict[str, Any] = field(default_factory=dict)
    pooling_selection: dict[str, Any] = field(default_factory=dict)
    future_evaluations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def print_plan(self) -> None:
        """Format and print execution plan to console."""
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION PLAN")
        print("=" * 80)
        print(f"Mode: {'DRY RUN (nothing is trained or evaluated)' if self.dry_run else 'LIVE'}")
        print(f"Resume: {self.resume} | Force: {self.force}")
        print(f"Phase bounds: From={self.from_phase} -> Until={self.until_phase}")
        print(
            f"Error handling: Fail-fast={self.fail_fast} | Continue-on-error={self.continue_on_error}"
        )
        print("-" * 80)
        print(
            f"Configurations planned: {len(self.experiments)} | Seeds: {self.seeds} | "
            f"Future training runs: {self.planned_runs}"
        )
        for exp in self.experiments:
            print(
                f"  - {exp['experiment_id']:<26} phase {exp['phase']} "
                f"{exp['architecture']}-{exp['hidden_size']} pooling={exp['pooling']}"
            )
        print("-" * 80)
        print("Data / split:")
        for key, value in self.data_info.items():
            print(f"  {key}: {value}")
        print(f"Pooling selection: {self.pooling_selection.get('state', 'n/a')}")
        print("-" * 80)
        print(f"{'Step ID':<26} {'Phase':<8} {'Action':<10} {'Description'}")
        print("-" * 80)

        for step in self.steps:
            phase_str = str(step.phase) if step.phase is not None else "-"
            print(f"{step.step_id:<26} {phase_str:<8} {step.action:<10} {step.description}")
            for dep in step.dependencies:
                print(f"{'':<26} {'':<8} {'':<10}   depends on: {dep}")

        print("-" * 80)
        print("Future evaluations (not executed):")
        for item in self.future_evaluations:
            print(f"  - {item}")
        print("Expected artifacts:")
        for step in self.steps:
            for out in step.expected_outputs:
                print(f"  - {out}")
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


def describe_data_state(base_config_path: Path) -> dict[str, Any]:
    """Describe the data and split without loading the dataset or touching any checkpoint.

    Reads only the configuration and, when present, the persisted split document.
    """
    cfg = build_experiment_config(base_config_path)
    split_path = Path(cfg.data.processed_dir) / "split_indices.json"
    info: dict[str, Any] = {
        "split_method": cfg.data.split_method,
        "split_seed": cfg.data.split_seed,
        "split_n_splits": cfg.data.split_n_splits,
        "group_by_series_hash": cfg.data.group_by_series_hash,
        "series_hash": "SHA-256 over 96 little-endian float64 values (label and index excluded)",
        "split_file": str(split_path),
    }
    if not split_path.is_file():
        info["split_state"] = "not generated yet (created by data_preparation)"
        return info
    try:
        with open(split_path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        info["split_state"] = f"unreadable: {exc}"
        return info
    if payload.get("method") != "stratified_group_kfold_by_series_hash":
        info["split_state"] = "OBSOLETE (not produced by the hash-grouped method); will be rejected"
        return info
    info["split_state"] = "hash-grouped split present"
    info["split_hash"] = payload.get("split_hash")
    info["dataset_hash"] = payload.get("dataset_hash")
    info["train_samples"] = payload.get("train_samples")
    info["val_samples"] = payload.get("val_samples")
    info["shared_hashes_train_val"] = payload.get("shared_hashes_train_val")
    info["selected_fold"] = payload.get("selected_fold")
    return info


def describe_pooling_state(artifacts_dir: Path) -> dict[str, Any]:
    """Describe whether a pooling selection exists (only the file's presence is checked)."""
    path = Path(artifacts_dir) / "summary" / "pooling_selection.json"
    if path.is_file():
        return {"state": f"existing selection file {path} (validated against the split at use)"}
    return {"state": "pending: selected automatically after Phase 2 (E06-E08 use best_pooling)"}


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
    base_config_path: Path = Path("configs/base.yaml"),
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
        elif resume and step.expected_outputs:
            outputs_exist = all(
                resolve_output_path(p, artifacts_dir).is_file() for p in step.expected_outputs
            )
            if outputs_exist:
                action = "SKIP"

        plan_items.append(
            PlanStepItem(
                step_id=step.step_id,
                phase=step.phase,
                description=step.description,
                dependencies=step.dependencies,
                expected_outputs=[
                    str(resolve_output_path(p, artifacts_dir)).replace("\\", "/")
                    for p in step.expected_outputs
                ],
                action=action,
            )
        )

    cfg = build_experiment_config(base_config_path)
    seeds = list(cfg.training.train_seeds)
    experiments = [
        {
            "experiment_id": spec.experiment_id,
            "phase": spec.phase,
            "architecture": spec.architecture,
            "hidden_size": spec.hidden_size,
            "pooling": spec.pooling,
            "config_file": spec.config_file,
        }
        for spec in EXPERIMENT_REGISTRY.values()
        if (from_phase is None or spec.phase >= from_phase)
        and (until_phase is None or spec.phase <= until_phase)
    ]

    return ExecutionPlan(
        steps=plan_items,
        dry_run=dry_run,
        resume=resume,
        force=force,
        from_phase=from_phase,
        until_phase=until_phase,
        fail_fast=fail_fast,
        continue_on_error=continue_on_error,
        experiments=experiments,
        seeds=seeds,
        planned_runs=len(experiments) * len(seeds),
        data_info=describe_data_state(base_config_path),
        pooling_selection=describe_pooling_state(artifacts_dir),
        future_evaluations=list(FUTURE_EVALUATIONS),
    )
