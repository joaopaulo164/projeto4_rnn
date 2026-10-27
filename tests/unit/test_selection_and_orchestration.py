"""Unit tests for pooling selection rule, parsimony margin, dependency DAG, and execution flags."""

from pathlib import Path

import pytest

from project4_rnn.experiments.model_selection import select_final_model
from project4_rnn.experiments.pooling_selection import (
    load_selected_pooling,
    select_best_pooling,
)
from project4_rnn.experiments.provenance import StaleArtifactError
from project4_rnn.experiments.timing import IncompleteRunError
from project4_rnn.orchestration.dependency_graph import (
    build_standard_pipeline_graph,
    resolve_output_path,
)
from project4_rnn.orchestration.execution_plan import (
    validate_execution_flags,
)

DATASET_HASH = "dataset-hash"
SPLIT_HASH = "split-hash"


def _summary(exp_id: str, f1: float, params: int, **extra: object) -> dict[str, object]:
    summary: dict[str, object] = {
        "experiment_id": exp_id,
        "val_macro_f1_mean": f1,
        "val_macro_f1_std": 0.01,
        "val_balanced_accuracy_mean": f1,
        "val_balanced_accuracy_std": 0.01,
        "val_loss_mean": 0.5,
        "total_training_time_sec_mean": 10.0,
        "total_parameters": params,
    }
    summary.update(extra)
    return summary


def _phase2(mean_max_f1: float) -> list[dict[str, object]]:
    return [
        _summary("E02_GRU_32_LAST", 0.900, 3591),
        _summary("E03_GRU_32_MEAN", 0.940, 3591),
        _summary("E04_GRU_32_MAX", 0.920, 3591),
        _summary("E05_GRU_32_MEAN_MAX", mean_max_f1, 3815),
    ]


def test_parsimony_rule_threshold(tmp_path: Path) -> None:
    """Verify parsimony rule with 0.005 threshold (0.5 percentage points Macro F1)."""
    out_file = tmp_path / "pooling_selection.json"

    res_a = select_best_pooling(
        _phase2(0.943), out_file, DATASET_HASH, SPLIT_HASH, parsimony_threshold=0.005
    )
    assert res_a["selected_pooling"] == "mean"
    assert "Parsimony principle applied" in res_a["rule_applied"]
    assert res_a["split_hash"] == SPLIT_HASH
    assert res_a["dataset_hash"] == DATASET_HASH

    res_b = select_best_pooling(
        _phase2(0.950), out_file, DATASET_HASH, SPLIT_HASH, parsimony_threshold=0.005
    )
    assert res_b["selected_pooling"] == "mean_max"


def test_pooling_selection_rejects_missing_fields(tmp_path: Path) -> None:
    """A missing summary field is reported with the run id and never defaulted to zero."""
    summaries = _phase2(0.95)
    del summaries[1]["total_training_time_sec_mean"]
    with pytest.raises(IncompleteRunError, match="E03_GRU_32_MEAN"):
        select_best_pooling(summaries, tmp_path / "p.json", DATASET_HASH, SPLIT_HASH)
    assert not (tmp_path / "p.json").exists()


def test_pooling_selection_empty_and_unknown(tmp_path: Path) -> None:
    """Empty input and input without Phase 2 candidates raise ValueError."""
    with pytest.raises(ValueError, match="no Phase 2"):
        select_best_pooling([], tmp_path / "p.json", DATASET_HASH, SPLIT_HASH)
    with pytest.raises(ValueError, match="No valid Phase 2"):
        select_best_pooling(
            [_summary("E01_MLP_32", 0.9, 3335)], tmp_path / "p.json", DATASET_HASH, SPLIT_HASH
        )


def test_load_selected_pooling_detects_stale_split(tmp_path: Path) -> None:
    """A pooling selection made on another split is rejected."""
    path = tmp_path / "pooling_selection.json"
    select_best_pooling(_phase2(0.95), path, DATASET_HASH, SPLIT_HASH)
    assert load_selected_pooling(path, SPLIT_HASH, DATASET_HASH) == "mean_max"
    with pytest.raises(StaleArtifactError, match="split_hash"):
        load_selected_pooling(path, "another-split", DATASET_HASH)
    with pytest.raises(FileNotFoundError):
        load_selected_pooling(tmp_path / "missing.json")


def test_final_model_selection(tmp_path: Path) -> None:
    """Verify final model selection on validation performance and recorded identity."""
    out_file = tmp_path / "final_model_selection.json"
    all_summaries = [
        _summary("E01_MLP_32", 0.87, 3335, phase=1),
        _summary("E07_GRU_64_BEST_POOLING", 0.95, 13319, phase=4),
    ]

    res = select_final_model(all_summaries, out_file, DATASET_HASH, SPLIT_HASH)
    assert res["selected_experiment_id"] == "E07_GRU_64_BEST_POOLING"
    assert res["split_hash"] == SPLIT_HASH
    assert out_file.is_file()


def test_final_model_selection_parsimony_and_provenance(tmp_path: Path) -> None:
    """Parsimony prefers the smaller model; the provenance block is merged into the file."""
    summaries = [
        _summary("E01_MLP_32", 0.948, 3335, phase=1),
        _summary("E07_GRU_64_BEST_POOLING", 0.950, 13319, phase=4),
    ]
    res = select_final_model(
        summaries,
        tmp_path / "sel.json",
        DATASET_HASH,
        SPLIT_HASH,
        provenance_builder=lambda exp_id: {"experiment_id": exp_id, "selected_seed": 7},
    )
    assert res["selected_experiment_id"] == "E01_MLP_32"
    assert res["parsimony_triggered"] is True
    assert res["selected_seed"] == 7


def test_final_model_selection_requires_phase(tmp_path: Path) -> None:
    """A missing phase field is an incomplete summary, not a default."""
    summaries = [_summary("E01_MLP_32", 0.9, 3335)]
    with pytest.raises(IncompleteRunError, match="phase"):
        select_final_model(summaries, tmp_path / "sel.json", DATASET_HASH, SPLIT_HASH)
    with pytest.raises(ValueError, match="empty"):
        select_final_model([], tmp_path / "sel.json", DATASET_HASH, SPLIT_HASH)


def test_dependency_graph_topological_sort() -> None:
    """Verify DAG topological ordering respects dependencies."""
    dag = build_standard_pipeline_graph()
    order = dag.get_topological_order()

    assert "data_validation" not in order
    assert order.index("data_preparation") < order.index("phase_1")
    assert order.index("data_preparation") < order.index("phase_2")
    assert order.index("data_preparation") < order.index("duplicate_audit")
    assert order.index("phase_2") < order.index("pooling_selection")
    assert order.index("pooling_selection") < order.index("phase_3")
    assert order.index("pooling_selection") < order.index("phase_4")
    assert order.index("phase_4") < order.index("consolidation")
    assert order.index("consolidation") < order.index("final_model_selection")
    assert order.index("final_model_selection") < order.index("final_test_evaluation")
    assert order.index("final_test_evaluation") < order.index("sensitivity_evaluation")
    assert order.index("duplicate_audit") < order.index("sensitivity_evaluation")
    assert order.index("sensitivity_evaluation") < order.index("evidence_generation")
    assert order.index("evidence_generation") < order.index("compliance_audit")


def test_resolve_output_path_is_relative_to_artifacts_dir(tmp_path: Path) -> None:
    """Logical artifact paths are resolved under the requested artifacts directory."""
    resolved = resolve_output_path("artifacts/summary/pooling_selection.json", tmp_path)
    assert resolved == tmp_path / "summary" / "pooling_selection.json"
    other = resolve_output_path("data/processed/split_indices.json", tmp_path)
    assert other == Path("data/processed/split_indices.json")


def test_execution_flags_validation() -> None:
    """Verify incompatible CLI options raise descriptive ValueError."""
    # Incompatible resume + force
    with pytest.raises(ValueError, match="Incompatible options: --resume and --force"):
        validate_execution_flags(resume=True, force=True)

    # Incompatible fail-fast + continue-on-error
    with pytest.raises(
        ValueError, match="Incompatible options: --fail-fast and --continue-on-error"
    ):
        validate_execution_flags(fail_fast=True, continue_on_error=True)

    # Invalid phase bound
    with pytest.raises(ValueError, match="Invalid phase range"):
        validate_execution_flags(
            from_phase=4, until_phase=2, fail_fast=True, continue_on_error=False
        )
