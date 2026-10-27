"""Tests for the CLI surface, flag validation and the purity of the dry-run."""

import json
from pathlib import Path

import pytest

from project4_rnn import cli
from project4_rnn.orchestration.dependency_graph import build_standard_pipeline_graph
from project4_rnn.orchestration.execution_plan import (
    FUTURE_EVALUATIONS,
    build_execution_plan,
    validate_execution_flags,
)
from project4_rnn.orchestration.manifest import PipelineManifest
from project4_rnn.orchestration.pipeline import run_full_pipeline, should_execute_phase

pytestmark = pytest.mark.unit

BASE_CONFIG = Path("configs/base.yaml")


def test_parser_exposes_all_commands() -> None:
    parser = cli.build_parser()
    for argv in (
        ["audit-duplicates"],
        ["audit-readiness", "--run-quality"],
        ["run-full-experiment", "--dry-run"],
        ["smoke-test"],
        ["clean-artifacts", "--generated-only"],
        ["prepare-data", "--force"],
        ["validate-data"],
        ["run-phase", "--phase", "2"],
        ["evaluate-sensitivity"],
        ["audit"],
    ):
        args = parser.parse_args(argv)
        assert callable(args.func)


def test_dry_run_flag_defaults_and_resume_flags() -> None:
    args = cli.build_parser().parse_args(
        ["run-full-experiment", "--config", "configs/base.yaml", "--resume", "--fail-fast"]
    )
    assert args.resume and args.fail_fast and not args.dry_run and not args.force


def test_clean_artifacts_requires_generated_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main_returning(["clean-artifacts", "--artifacts-dir", str(tmp_path)])
    assert code == 1
    assert "Refusing" in capsys.readouterr().out


def test_clean_artifacts_generated_only_in_temp_dir(tmp_path: Path) -> None:
    artifacts, processed = tmp_path / "artifacts", tmp_path / "processed"
    (artifacts / "runs").mkdir(parents=True)
    (artifacts / "corrections").mkdir()
    processed.mkdir()
    (processed / "split_indices.json").write_text("{}", encoding="utf-8")
    argv = [
        "clean-artifacts",
        "--generated-only",
        "--artifacts-dir",
        str(artifacts),
        "--processed-dir",
        str(processed),
    ]
    assert cli.main_returning(argv + ["--dry-run"]) == 0
    assert (artifacts / "runs").exists()
    assert cli.main_returning(argv) == 0
    assert not (artifacts / "runs").exists()
    assert not (processed / "split_indices.json").exists()
    assert (artifacts / "corrections").exists()


def test_clean_artifacts_refuses_project_root(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    argv = ["clean-artifacts", "--generated-only", "--artifacts-dir", str(tmp_path)]
    assert cli.main_returning(argv + ["--processed-dir", str(tmp_path / "p")]) == 1


def test_validate_execution_flags_rejects_incompatible_options() -> None:
    validate_execution_flags()
    with pytest.raises(ValueError, match="--resume and --force"):
        validate_execution_flags(resume=True, force=True)
    with pytest.raises(ValueError, match="--fail-fast and --continue-on-error"):
        validate_execution_flags(fail_fast=True, continue_on_error=True)
    with pytest.raises(ValueError, match="--from-phase"):
        validate_execution_flags(from_phase=7)
    with pytest.raises(ValueError, match="--until-phase"):
        validate_execution_flags(until_phase=0)
    with pytest.raises(ValueError, match="greater than"):
        validate_execution_flags(from_phase=3, until_phase=2)


def test_continue_on_error_alone_is_accepted() -> None:
    validate_execution_flags(fail_fast=False, continue_on_error=True)


def test_dag_has_no_data_validation_and_is_topologically_ordered() -> None:
    dag = build_standard_pipeline_graph()
    order = dag.get_topological_order()
    assert "data_validation" not in order
    assert order.index("data_preparation") < order.index("duplicate_audit")
    assert order.index("phase_2") < order.index("phase_3")
    assert order.index("final_test_evaluation") < order.index("sensitivity_evaluation")


def test_execution_plan_has_eight_configurations_and_forty_runs(tmp_path: Path) -> None:
    plan = build_execution_plan(dry_run=True, artifacts_dir=tmp_path, base_config_path=BASE_CONFIG)
    assert len(plan.experiments) == 8
    assert plan.seeds == [7, 21, 42, 84, 168]
    assert plan.planned_runs == 40
    assert plan.future_evaluations == list(FUTURE_EVALUATIONS)
    assert all(s.action == "EXECUTE" for s in plan.steps)
    assert plan.pooling_selection["state"].startswith("pending")
    assert plan.to_dict()["planned_runs"] == 40


def test_execution_plan_respects_phase_bounds_and_force(tmp_path: Path) -> None:
    plan = build_execution_plan(
        force=True,
        from_phase=2,
        until_phase=3,
        artifacts_dir=tmp_path,
        base_config_path=BASE_CONFIG,
    )
    assert {e["phase"] for e in plan.experiments} == {2, 3}
    assert all(s.action == "RERUN" for s in plan.steps)


def test_execution_plan_resume_skips_completed_steps(tmp_path: Path) -> None:
    dag = build_standard_pipeline_graph()
    step = dag.get_step("duplicate_audit")
    for out in step.expected_outputs:
        target = tmp_path / out.removeprefix("artifacts/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x", encoding="utf-8")
    plan = build_execution_plan(resume=True, artifacts_dir=tmp_path, base_config_path=BASE_CONFIG)
    actions = {s.step_id: s.action for s in plan.steps}
    assert actions["duplicate_audit"] == "SKIP"
    assert actions["phase_1"] == "EXECUTE"


def test_dry_run_writes_nothing_and_trains_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifacts = tmp_path / "never_created"
    result = run_full_pipeline(base_config_path=BASE_CONFIG, artifacts_dir=artifacts, dry_run=True)
    assert result["status"] == "DRY_RUN_COMPLETED"
    assert not artifacts.exists()
    assert result["plan"]["planned_runs"] == 40
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "Future evaluations (not executed)" in out
    assert "data_validation" not in out


def test_cli_dry_run_exit_code_and_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main_returning(
        [
            "run-full-experiment",
            "--config",
            str(BASE_CONFIG),
            "--artifacts-dir",
            str(tmp_path / "a"),
            "--dry-run",
        ]
    )
    assert code == 0
    assert "No training, checkpoint or test evaluation" in capsys.readouterr().out
    assert not (tmp_path / "a").exists()


def test_cli_rejects_incompatible_flags(tmp_path: Path) -> None:
    argv = [
        "run-full-experiment",
        "--config",
        str(BASE_CONFIG),
        "--artifacts-dir",
        str(tmp_path),
        "--dry-run",
        "--resume",
        "--force",
    ]
    assert cli.main_returning(argv) == 1


def test_should_execute_phase_bounds() -> None:
    assert should_execute_phase(2, None, None)
    assert not should_execute_phase(1, 2, None)
    assert not should_execute_phase(4, None, 3)


def test_manifest_report_requires_all_metrics(tmp_path: Path) -> None:
    manifest = PipelineManifest("unit_run", artifacts_dir=tmp_path)
    manifest.save_initial_state({"a": 1}, {"steps": []})
    manifest.record_step_completed("x")
    manifest.record_step_failed("y", "boom")
    metrics = {
        "macro_f1": 0.5,
        "balanced_accuracy": 0.5,
        "accuracy": 0.5,
        "macro_precision": 0.5,
        "macro_recall": 0.5,
        "loss": 0.5,
    }
    report = manifest.finalize(
        status="FAILED", dataset_manifest={"k": "v"}, final_test_metrics=metrics
    )
    text = report.read_text(encoding="utf-8")
    assert "boom" in text and "Macro F1" in text
    status = json.loads((manifest.run_dir / "pipeline_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "FAILED" and status["failed_steps_count"] == 1

    incomplete = PipelineManifest("unit_run_2", artifacts_dir=tmp_path)
    with pytest.raises(KeyError, match="macro_recall"):
        incomplete.finalize(
            final_test_metrics={k: v for k, v in metrics.items() if k != "macro_recall"}
        )


def test_run_phase_rejects_invalid_number(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main_returning(
        [
            "run-phase",
            "--phase",
            "9",
            "--config",
            str(BASE_CONFIG),
            "--artifacts-dir",
            str(tmp_path),
        ]
    )
    assert code == 1
    assert "Invalid phase" in capsys.readouterr().out
