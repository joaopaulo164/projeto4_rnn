"""Orchestration tests of the live pipeline path with every heavy step replaced by a stub.

No data is loaded and nothing is trained: the stubs only record the order in which the pipeline
calls its steps and let a chosen step fail.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from project4_rnn.orchestration import pipeline
from project4_rnn.orchestration.pipeline import StepFailedError, run_full_pipeline

pytestmark = pytest.mark.unit

BASE_CONFIG = Path("configs/base.yaml")


class Recorder:
    """Stub factory recording calls; ``fail_on`` names the step that raises."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[str] = []
        self.fail_on = fail_on

    def step(self, name: str, result: Any = None) -> Any:
        def _run(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            if name == self.fail_on:
                raise RuntimeError(f"{name} exploded")
            return result

        return _run


def _install(monkeypatch: pytest.MonkeyPatch, rec: Recorder) -> None:
    bundle = SimpleNamespace(manifest={"dataset": "fixture"}, split_hash="split-x")
    monkeypatch.setattr(pipeline, "prepare_and_load_data", rec.step("data_preparation", bundle))
    monkeypatch.setattr(pipeline, "mark_artifacts_obsolete", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "run_duplicate_audit", rec.step("duplicate_audit", {}))
    monkeypatch.setattr(pipeline, "run_phase_1", rec.step("phase_1"))
    monkeypatch.setattr(pipeline, "run_phase_2", rec.step("phase_2"))
    monkeypatch.setattr(pipeline, "run_phase_3", rec.step("phase_3"))
    monkeypatch.setattr(pipeline, "run_phase_4", rec.step("phase_4"))
    full = dict.fromkeys(
        ("macro_f1", "balanced_accuracy", "accuracy", "macro_precision", "macro_recall", "loss"),
        0.1,
    )
    metrics = {"test_metrics": full}
    monkeypatch.setattr(
        pipeline, "run_final_evaluation", rec.step("final_test_evaluation", metrics)
    )
    monkeypatch.setattr(pipeline, "run_sensitivity_analysis", rec.step("sensitivity_evaluation"))


def test_successful_run_calls_steps_in_dependency_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = Recorder()
    _install(monkeypatch, rec)
    result = run_full_pipeline(base_config_path=BASE_CONFIG, artifacts_dir=tmp_path)
    assert result["status"] == "COMPLETED"
    assert rec.calls == [
        "data_preparation",
        "duplicate_audit",
        "phase_1",
        "phase_2",
        "phase_3",
        "phase_4",
        "final_test_evaluation",
        "sensitivity_evaluation",
    ]
    run_dirs = list((tmp_path / "pipeline_runs").iterdir())
    assert len(run_dirs) == 1
    completed = json.loads((run_dirs[0] / "completed_steps.json").read_text(encoding="utf-8"))
    assert "pooling_selection" in completed and "evidence_generation" in completed


def test_fail_fast_stops_at_first_failing_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = Recorder(fail_on="phase_2")
    _install(monkeypatch, rec)
    with pytest.raises(StepFailedError, match="phase_2"):
        run_full_pipeline(base_config_path=BASE_CONFIG, artifacts_dir=tmp_path, fail_fast=True)
    assert rec.calls[-1] == "phase_2"
    assert "phase_3" not in rec.calls and "final_test_evaluation" not in rec.calls
    status = next((tmp_path / "pipeline_runs").iterdir()) / "pipeline_status.json"
    assert json.loads(status.read_text(encoding="utf-8"))["status"] == "FAILED"


def test_continue_on_error_blocks_only_dependent_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = Recorder(fail_on="phase_1")
    _install(monkeypatch, rec)
    result = run_full_pipeline(
        base_config_path=BASE_CONFIG,
        artifacts_dir=tmp_path,
        fail_fast=False,
        continue_on_error=True,
    )
    assert result["status"] == "FAILED"
    assert "phase_1" in result["failed_steps"]
    assert "final_test_evaluation" in result["failed_steps"]
    assert "final_test_evaluation" not in rec.calls
    assert "duplicate_audit" in rec.calls


def test_phase_bounds_skip_phases_and_final_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = Recorder()
    _install(monkeypatch, rec)
    run_full_pipeline(base_config_path=BASE_CONFIG, artifacts_dir=tmp_path, until_phase=2)
    assert "phase_3" not in rec.calls and "final_test_evaluation" not in rec.calls
    rec_two = Recorder()
    _install(monkeypatch, rec_two)
    run_full_pipeline(base_config_path=BASE_CONFIG, artifacts_dir=tmp_path / "b", from_phase=3)
    assert "phase_1" not in rec_two.calls and "phase_3" in rec_two.calls
