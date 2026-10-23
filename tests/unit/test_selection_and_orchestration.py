"""Unit tests for pooling selection rule, parsimony margin, dependency DAG, and execution flags."""

from pathlib import Path

import pytest

from project4_rnn.experiments.model_selection import select_final_model
from project4_rnn.experiments.pooling_selection import (
    select_best_pooling,
)
from project4_rnn.orchestration.dependency_graph import (
    build_standard_pipeline_graph,
)
from project4_rnn.orchestration.execution_plan import (
    validate_execution_flags,
)


def test_parsimony_rule_threshold(tmp_path: Path) -> None:
    """Verify parsimony rule with 0.005 threshold (0.5 percentage points)."""
    out_file = tmp_path / "pooling_selection.json"

    # Case A: mean_max lead is 0.003 (< 0.005 threshold) -> Must select simpler model (mean)
    summaries_a = [
        {
            "experiment_id": "E02_GRU_32_LAST",
            "val_balanced_accuracy_mean": 0.900,
            "total_parameters": 3426,
        },
        {
            "experiment_id": "E03_GRU_32_MEAN",
            "val_balanced_accuracy_mean": 0.940,
            "total_parameters": 3426,
        },
        {
            "experiment_id": "E04_GRU_32_MAX",
            "val_balanced_accuracy_mean": 0.920,
            "total_parameters": 3426,
        },
        {
            "experiment_id": "E05_GRU_32_MEAN_MAX",
            "val_balanced_accuracy_mean": 0.943,
            "total_parameters": 3490,
        },
    ]

    res_a = select_best_pooling(summaries_a, output_path=out_file, parsimony_threshold=0.005)
    assert res_a["selected_pooling"] == "mean"
    assert "Parsimony principle applied" in res_a["rule_applied"]

    # Case B: mean_max lead is 0.010 (> 0.005 threshold) -> Must select mean_max
    summaries_b = [
        {
            "experiment_id": "E02_GRU_32_LAST",
            "val_balanced_accuracy_mean": 0.900,
            "total_parameters": 3426,
        },
        {
            "experiment_id": "E03_GRU_32_MEAN",
            "val_balanced_accuracy_mean": 0.940,
            "total_parameters": 3426,
        },
        {
            "experiment_id": "E04_GRU_32_MAX",
            "val_balanced_accuracy_mean": 0.920,
            "total_parameters": 3426,
        },
        {
            "experiment_id": "E05_GRU_32_MEAN_MAX",
            "val_balanced_accuracy_mean": 0.950,
            "total_parameters": 3490,
        },
    ]

    res_b = select_best_pooling(summaries_b, output_path=out_file, parsimony_threshold=0.005)
    assert res_b["selected_pooling"] == "mean_max"


def test_final_model_selection(tmp_path: Path) -> None:
    """Verify final model selection on validation performance."""
    out_file = tmp_path / "final_model_selection.json"
    all_summaries = [
        {
            "experiment_id": "E01_MLP_32",
            "phase": 1,
            "val_balanced_accuracy_mean": 0.88,
            "val_f1_mean": 0.87,
            "total_parameters": 4578,
        },
        {
            "experiment_id": "E07_GRU_64_BEST_POOLING",
            "phase": 4,
            "val_balanced_accuracy_mean": 0.96,
            "val_f1_mean": 0.95,
            "total_parameters": 12994,
        },
    ]

    res = select_final_model(all_summaries, output_path=out_file)
    assert res["selected_experiment_id"] == "E07_GRU_64_BEST_POOLING"
    assert out_file.is_file()


def test_dependency_graph_topological_sort() -> None:
    """Verify DAG topological ordering respects dependencies."""
    dag = build_standard_pipeline_graph()
    order = dag.get_topological_order()

    # Verify order constraints
    assert order.index("data_preparation") < order.index("data_validation")
    assert order.index("data_validation") < order.index("phase_1")
    assert order.index("data_validation") < order.index("phase_2")
    assert order.index("phase_2") < order.index("pooling_selection")
    assert order.index("pooling_selection") < order.index("phase_3")
    assert order.index("pooling_selection") < order.index("phase_4")
    assert order.index("phase_4") < order.index("consolidation")
    assert order.index("consolidation") < order.index("final_model_selection")
    assert order.index("final_model_selection") < order.index("final_test_evaluation")
    assert order.index("final_test_evaluation") < order.index("evidence_generation")
    assert order.index("evidence_generation") < order.index("compliance_audit")


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
