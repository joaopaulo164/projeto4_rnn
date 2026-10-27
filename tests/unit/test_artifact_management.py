"""Tests for safe artifact cleaning, obsolescence marking and step resumption."""

import json
from pathlib import Path

import pytest

from project4_rnn.audit.artifacts import (
    clean_generated_artifacts,
    plan_clean_generated_artifacts,
    verify_file_exists_and_non_empty,
    verify_json_validity,
)
from project4_rnn.orchestration.dependency_graph import PipelineStep
from project4_rnn.orchestration.invalidation import (
    OBSOLETE_MARKER_FILE,
    find_stale_runs,
    find_stale_summaries,
    invalidate_experiment,
    invalidate_pooling_and_downstream,
    mark_artifacts_obsolete,
)
from project4_rnn.orchestration.resume import evaluate_resumption_state, is_step_completed

pytestmark = pytest.mark.unit


def _populate(artifacts: Path, processed: Path) -> None:
    for sub in (
        "runs/E01/1",
        "summary",
        "audits",
        "pipeline_runs",
        "logs",
        "corrections",
        "quality",
    ):
        (artifacts / sub).mkdir(parents=True, exist_ok=True)
        (artifacts / sub / "file.txt").write_text("x", encoding="utf-8")
    (artifacts / "environment.json").write_text("{}", encoding="utf-8")
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "split_indices.json").write_text("{}", encoding="utf-8")
    (processed / ".gitkeep").write_text("", encoding="utf-8")


def test_clean_removes_generated_and_keeps_protected(tmp_path: Path) -> None:
    artifacts, processed, smoke = tmp_path / "artifacts", tmp_path / "processed", tmp_path / "smk"
    _populate(artifacts, processed)
    smoke.mkdir()
    removed = clean_generated_artifacts(artifacts, processed, smoke)
    assert removed
    assert not (artifacts / "runs").exists()
    assert not (artifacts / "summary").exists()
    assert not (artifacts / "environment.json").exists()
    assert not (processed / "split_indices.json").exists()
    assert not smoke.exists()
    assert (processed / ".gitkeep").exists()
    assert (artifacts / "corrections" / "file.txt").exists()
    assert (artifacts / "quality" / "file.txt").exists()


def test_clean_dry_run_deletes_nothing(tmp_path: Path) -> None:
    artifacts, processed = tmp_path / "artifacts", tmp_path / "processed"
    _populate(artifacts, processed)
    planned = clean_generated_artifacts(artifacts, processed, tmp_path / "none", dry_run=True)
    assert planned
    assert (artifacts / "runs" / "E01" / "1" / "file.txt").exists()
    assert planned == [
        str(p) for p in plan_clean_generated_artifacts(artifacts, processed, tmp_path / "none")
    ]


def test_clean_refuses_project_root(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="project root"):
        clean_generated_artifacts(tmp_path, tmp_path / "processed", tmp_path / "smk")


def test_clean_tolerates_missing_directories(tmp_path: Path) -> None:
    assert clean_generated_artifacts(tmp_path / "a", tmp_path / "p", tmp_path / "s") == []


def test_verify_helpers(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert verify_file_exists_and_non_empty(missing)[0] is False
    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    assert "empty" in verify_file_exists_and_non_empty(empty)[1]
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    assert verify_json_validity(bad)[0] is False
    good = tmp_path / "good.json"
    good.write_text('{"a": 1}', encoding="utf-8")
    ok, _, data = verify_json_validity(good)
    assert ok and data == {"a": 1}


def _write_status(artifacts: Path, exp: str, seed: int, split_hash: str | None) -> Path:
    run = artifacts / "runs" / exp / str(seed)
    run.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"status": "COMPLETED"}
    if split_hash is not None:
        payload["split_hash"] = split_hash
    (run / "status.json").write_text(json.dumps(payload), encoding="utf-8")
    return run


def test_stale_runs_and_summaries_are_detected(tmp_path: Path) -> None:
    fresh = _write_status(tmp_path, "E01", 1, "new")
    legacy = _write_status(tmp_path, "E01", 2, None)
    other = _write_status(tmp_path, "E02", 1, "old")
    stale = find_stale_runs(tmp_path, "new")
    assert fresh not in stale
    assert set(stale) == {legacy, other}

    summary = tmp_path / "summary"
    summary.mkdir()
    (summary / "pooling_selection.json").write_text('{"split_hash": "old"}', encoding="utf-8")
    (summary / "final_model_selection.json").write_text('{"split_hash": "new"}', encoding="utf-8")
    (summary / "final_test_metrics.json").write_text("not json", encoding="utf-8")
    names = {p.name for p in find_stale_summaries(tmp_path, "new")}
    assert names == {"pooling_selection.json", "final_test_metrics.json"}


def test_find_stale_runs_without_runs_dir(tmp_path: Path) -> None:
    assert find_stale_runs(tmp_path, "x") == []


def test_mark_obsolete_writes_marker_and_keeps_results(tmp_path: Path) -> None:
    run = _write_status(tmp_path, "E01", 1, "old")
    marker = mark_artifacts_obsolete(tmp_path, "new", "split changed")
    assert marker is not None and marker.name == OBSOLETE_MARKER_FILE
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["current_split_hash"] == "new"
    assert payload["reason"] == "split changed"
    assert str(run) in payload["obsolete_runs"]
    assert (run / "status.json").exists()


def test_mark_obsolete_returns_none_when_everything_is_current(tmp_path: Path) -> None:
    _write_status(tmp_path, "E01", 1, "new")
    assert mark_artifacts_obsolete(tmp_path, "new", "n/a") is None
    assert not (tmp_path / OBSOLETE_MARKER_FILE).exists()


def test_invalidate_experiment_removes_status_and_derived_files(tmp_path: Path) -> None:
    run = _write_status(tmp_path, "E01", 1, "h")
    summary = tmp_path / "summary"
    summary.mkdir()
    (summary / "experiments_summary.csv").write_text("a", encoding="utf-8")
    (summary / "keep.txt").write_text("a", encoding="utf-8")
    invalidate_experiment("E01", tmp_path)
    assert not (run / "status.json").exists()
    assert not (summary / "experiments_summary.csv").exists()
    assert (summary / "keep.txt").exists()


def test_invalidate_pooling_removes_selection_and_downstream_status(tmp_path: Path) -> None:
    run = _write_status(tmp_path, "E06_LSTM_32_BEST_POOLING", 1, "h")
    summary = tmp_path / "summary"
    summary.mkdir()
    (summary / "pooling_selection.json").write_text("{}", encoding="utf-8")
    invalidate_pooling_and_downstream(tmp_path)
    assert not (summary / "pooling_selection.json").exists()
    assert not (run / "status.json").exists()


def _step(outputs: list[str]) -> PipelineStep:
    return PipelineStep(step_id="s", phase=1, description="d", expected_outputs=outputs)


def test_is_step_completed_requires_non_empty_outputs(tmp_path: Path) -> None:
    out = tmp_path / "summary" / "x.json"
    step = _step(["artifacts/summary/x.json"])
    assert is_step_completed(step, tmp_path) is False
    out.parent.mkdir(parents=True)
    out.write_text("", encoding="utf-8")
    assert is_step_completed(step, tmp_path) is False
    out.write_text("{}", encoding="utf-8")
    assert is_step_completed(step, tmp_path) is True
    assert is_step_completed(_step([]), tmp_path) is False


def test_evaluate_resumption_state_partitions_steps(tmp_path: Path) -> None:
    done = tmp_path / "summary" / "a.json"
    done.parent.mkdir(parents=True)
    done.write_text("{}", encoding="utf-8")
    first = PipelineStep(
        step_id="a", phase=1, description="A", expected_outputs=["artifacts/summary/a.json"]
    )
    second = PipelineStep(
        step_id="b",
        phase=1,
        description="B",
        dependencies=["a"],
        expected_outputs=["artifacts/summary/b.json"],
    )
    completed, pending = evaluate_resumption_state([first, second], tmp_path)
    assert completed == ["a"]
    assert pending == ["b"]
