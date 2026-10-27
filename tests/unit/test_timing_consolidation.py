"""Unit tests for strict timing validation and the consolidation of run summaries."""

import json
from pathlib import Path

import pytest
from run_fixtures import make_run_metadata, write_run

from project4_rnn.evaluation.statistics import summarize_seed_runs
from project4_rnn.experiments.consolidation import (
    consolidate_experiments,
    load_all_run_summaries,
)
from project4_rnn.experiments.timing import (
    CONSOLIDATION_STATUS_FILE,
    IncompleteRunError,
    find_timing_problems,
    validate_timing_fields,
    write_consolidation_status,
)


def test_valid_timing_fields_are_returned() -> None:
    """A complete, consistent summary yields its three timing values."""
    run = make_run_metadata("E01_MLP_32", 42)
    timings = validate_timing_fields(run, "E01_MLP_32/seed_42")
    assert timings == {
        "total_training_time_sec": 2.0,
        "mean_epoch_time_sec": 0.5,
        "inference_time_sec": 0.01,
    }


@pytest.mark.parametrize(
    "field", ["total_training_time_sec", "mean_epoch_time_sec", "inference_time_sec"]
)
def test_missing_timing_field_is_an_error_naming_the_run(field: str) -> None:
    """An absent field is never defaulted to zero."""
    run = make_run_metadata("E01_MLP_32", 42)
    del run[field]
    with pytest.raises(IncompleteRunError, match="E01_MLP_32/seed_42") as info:
        validate_timing_fields(run, "E01_MLP_32/seed_42")
    assert field in str(info.value)
    assert info.value.problems[0].field == field


def test_none_value_counts_as_missing() -> None:
    """A null timing is treated as missing."""
    run = make_run_metadata("E01_MLP_32", 42)
    run["inference_time_sec"] = None
    with pytest.raises(IncompleteRunError, match="is missing"):
        validate_timing_fields(run, "r")


def test_legitimate_zero_timing_is_accepted() -> None:
    """A measured value of exactly 0.0 is valid."""
    run = make_run_metadata("E01_MLP_32", 42, mean_epoch_time=0.0)
    run["inference_time_sec"] = 0.0
    assert validate_timing_fields(run, "r")["inference_time_sec"] == 0.0


@pytest.mark.parametrize("bad", ["1.5", [1.0], True, float("nan"), float("inf"), -1.0])
def test_invalid_types_and_values_are_rejected(bad: object) -> None:
    """Strings, lists, booleans, NaN, infinity and negatives are not timings."""
    run = make_run_metadata("E01_MLP_32", 42)
    run["total_training_time_sec"] = bad
    assert find_timing_problems(run, "r")


def test_inconsistent_mean_and_total_is_reported() -> None:
    """mean_epoch_time x epochs must match the total."""
    run = make_run_metadata("E01_MLP_32", 42)
    run["total_training_time_sec"] = 99.0
    problems = find_timing_problems(run, "r")
    assert problems
    assert "inconsistent" in problems[0].problem


def test_summarize_seed_runs_aggregates_timings_across_seeds() -> None:
    """Mean, std, min and max of the timings are computed over the seeds."""
    runs = [
        make_run_metadata("E01_MLP_32", 7, mean_epoch_time=1.0),
        make_run_metadata("E01_MLP_32", 21, mean_epoch_time=2.0),
        make_run_metadata("E01_MLP_32", 42, mean_epoch_time=3.0),
    ]
    summary = summarize_seed_runs("E01_MLP_32", 1, runs)
    assert summary["completed_runs"] == 3
    assert summary["mean_epoch_time_sec_mean"] == pytest.approx(2.0)
    assert summary["mean_epoch_time_sec_std"] == pytest.approx(1.0)
    assert summary["mean_epoch_time_sec_min"] == pytest.approx(1.0)
    assert summary["mean_epoch_time_sec_max"] == pytest.approx(3.0)
    assert summary["total_training_time_sec_mean"] == pytest.approx(8.0)
    assert summary["inference_time_sec_mean"] == pytest.approx(0.01)


def test_summarize_seed_runs_halts_on_a_missing_timing() -> None:
    """One incomplete seed makes the whole experiment summary fail instead of averaging zero."""
    good = make_run_metadata("E01_MLP_32", 7)
    bad = make_run_metadata("E01_MLP_32", 21)
    del bad["mean_epoch_time_sec"]
    with pytest.raises(IncompleteRunError, match="seed_21"):
        summarize_seed_runs("E01_MLP_32", 1, [good, bad])


def test_summarize_with_no_runs_reports_zero_completed() -> None:
    """An experiment without runs is reported as such, without fabricated metrics."""
    assert summarize_seed_runs("E01_MLP_32", 1, []) == {
        "experiment_id": "E01_MLP_32",
        "phase": 1,
        "completed_runs": 0,
        "failed_runs": 0,
    }


def test_consolidation_of_complete_runs(tmp_path: Path) -> None:
    """Valid runs are consolidated, timings included, and the status is COMPLETE."""
    for seed in (7, 21):
        write_run(tmp_path, make_run_metadata("E01_MLP_32", seed, mean_epoch_time=0.5 * seed / 7))
    result = consolidate_experiments(tmp_path)
    summary = result["summary_df"].iloc[0]
    assert summary["completed_runs"] == 2
    assert summary["total_training_time_sec_mean"] > 0
    status = json.loads((tmp_path / "summary" / CONSOLIDATION_STATUS_FILE).read_text("utf-8"))
    assert status["status"] == "COMPLETE"
    assert status["runs_consolidated"] == 2
    seeds = (tmp_path / "summary" / "metrics_by_seed.csv").read_text("utf-8")
    assert "total_training_time_sec" in seeds


def test_consolidation_halts_and_marks_incomplete_when_a_timing_is_missing(
    tmp_path: Path,
) -> None:
    """A missing field stops the official consolidation and names the run; nothing is zeroed."""
    write_run(tmp_path, make_run_metadata("E01_MLP_32", 7))
    broken = make_run_metadata("E01_MLP_32", 21)
    del broken["inference_time_sec"]
    write_run(tmp_path, broken)

    with pytest.raises(IncompleteRunError, match="E01_MLP_32/seed_21"):
        consolidate_experiments(tmp_path)

    status = json.loads((tmp_path / "summary" / CONSOLIDATION_STATUS_FILE).read_text("utf-8"))
    assert status["status"] == "INCOMPLETE"
    assert status["problems"][0]["run"] == "E01_MLP_32/seed_21"
    assert not (tmp_path / "summary" / "experiments_summary.csv").exists()


def test_consolidation_detects_metadata_history_inconsistency(tmp_path: Path) -> None:
    """The summary epoch count must match training_history.csv."""
    run_dir = write_run(tmp_path, make_run_metadata("E01_MLP_32", 7))
    (run_dir / "training_history.csv").write_text(
        "epoch,epoch_time_sec\n1,0.5\n2,0.5\n", encoding="utf-8"
    )
    with pytest.raises(IncompleteRunError, match="total_epochs"):
        consolidate_experiments(tmp_path)


def test_consolidation_requires_epoch_times_in_history(tmp_path: Path) -> None:
    """A history without epoch_time_sec is an inconsistency."""
    run_dir = write_run(tmp_path, make_run_metadata("E01_MLP_32", 7))
    (run_dir / "training_history.csv").write_text(
        "epoch\n" + "\n".join(str(i) for i in range(1, 5)) + "\n", encoding="utf-8"
    )
    with pytest.raises(IncompleteRunError, match="epoch_time_sec"):
        consolidate_experiments(tmp_path)


def test_consolidation_reports_missing_and_unreadable_metadata(tmp_path: Path) -> None:
    """Missing or corrupt metadata files are reported, not skipped."""
    (tmp_path / "runs" / "E01_MLP_32" / "7").mkdir(parents=True)
    corrupt = tmp_path / "runs" / "E01_MLP_32" / "21"
    corrupt.mkdir(parents=True)
    (corrupt / "execution_metadata.json").write_text("{not json", encoding="utf-8")
    _, problems = load_all_run_summaries(tmp_path)
    assert {p.run_id for p in problems} == {"E01_MLP_32/seed_7", "E01_MLP_32/seed_21"}
    with pytest.raises(IncompleteRunError):
        consolidate_experiments(tmp_path)


def test_consolidation_without_runs_returns_empty(tmp_path: Path) -> None:
    """No runs directory means nothing to consolidate."""
    assert consolidate_experiments(tmp_path) == {}


def test_write_consolidation_status_roundtrip(tmp_path: Path) -> None:
    """The status file stores state, counts and problems."""
    path = write_consolidation_status(tmp_path, complete=True, runs_consolidated=40)
    assert json.loads(path.read_text("utf-8"))["runs_consolidated"] == 40
