"""Unit tests for the descriptive sensitivity evaluations of the final test set."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from project4_rnn.constants import SENSITIVITY_ARTIFACT_NAMES
from project4_rnn.data.hashing import hash_series_matrix
from project4_rnn.evaluation.final_evaluation import (
    comparison_frame,
    per_class_frame,
    run_sensitivity_evaluations,
    write_primary_artifacts,
)
from project4_rnn.evaluation.metrics import compute_classification_metrics

PROVENANCE = {
    "experiment_id": "E01_MLP_32",
    "seed": 7,
    "checkpoint": "artifacts/runs/E01_MLP_32/7/best_checkpoint.pt",
    "checkpoint_sha256": "a" * 64,
    "best_epoch": 3,
    "dataset_hash": "dataset-hash",
    "split_hash": "split-hash",
    "timestamp": "2000-01-01T00:00:00+00:00",
}


class RecordingEvaluator:
    """Fake frozen model: predicts the true label, except for every fifth sample."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[dict[str, object], np.ndarray, np.ndarray, float]:
        self.calls.append(len(y))
        preds = y.copy()
        preds[::5] = (preds[::5] + 1) % 7
        probs = np.full((len(y), 7), 0.01)
        probs[np.arange(len(y)), preds] = 0.94
        metrics = compute_classification_metrics(y, preds, probs, loss=0.5)
        return metrics, preds, probs, 0.0


def _run(
    tmp_path: Path,
    synthetic_dev_data: tuple[np.ndarray, np.ndarray],
    synthetic_test_data: tuple[np.ndarray, np.ndarray],
    synthetic_dev_hashes: list[str],
) -> tuple[dict[str, dict[str, object]], np.ndarray, RecordingEvaluator, dict[str, object]]:
    _, y_dev = synthetic_dev_data
    X_test, y_test = synthetic_test_data
    y_internal = y_test - 1
    evaluator = RecordingEvaluator()
    full_metrics, full_preds, full_probs, _ = evaluator(X_test, y_internal)
    full = write_primary_artifacts(
        tmp_path, full_metrics, full_preds, full_probs, y_internal, PROVENANCE, phase=1
    )
    docs = run_sensitivity_evaluations(
        evaluate=evaluator,
        X_test=X_test,
        y_test_internal=y_internal,
        y_test_official=y_test,
        dev_hashes=synthetic_dev_hashes,
        dev_labels=y_dev.tolist(),
        test_hashes=hash_series_matrix(X_test),
        full_document=full,
        provenance=PROVENANCE,
        summary_dir=tmp_path,
        phase=1,
    )
    return docs, X_test, evaluator, full


def test_sensitivity_writes_every_artifact_and_never_modifies_the_test_set(
    tmp_path: Path,
    synthetic_dev_data: tuple[np.ndarray, np.ndarray],
    synthetic_test_data: tuple[np.ndarray, np.ndarray],
    synthetic_dev_hashes: list[str],
) -> None:
    """All sensitivity artifacts are produced and the official test arrays are untouched."""
    X_before, y_before = synthetic_test_data[0].copy(), synthetic_test_data[1].copy()
    _run(tmp_path, synthetic_dev_data, synthetic_test_data, synthetic_dev_hashes)
    for name in SENSITIVITY_ARTIFACT_NAMES:
        assert (tmp_path / name).is_file(), name
    np.testing.assert_array_equal(synthetic_test_data[0], X_before)
    np.testing.assert_array_equal(synthetic_test_data[1], y_before)


def test_sensitivity_subsets_remove_the_expected_rows(
    tmp_path: Path,
    synthetic_dev_data: tuple[np.ndarray, np.ndarray],
    synthetic_test_data: tuple[np.ndarray, np.ndarray],
    synthetic_dev_hashes: list[str],
) -> None:
    """Rows sharing a hash with development, or in conflicting groups, are excluded."""
    docs, X_test, evaluator, full = _run(
        tmp_path, synthetic_dev_data, synthetic_test_data, synthetic_dev_hashes
    )
    overlap = set(synthetic_dev_hashes)
    test_hashes = hash_series_matrix(X_test)
    expected_removed = sum(1 for h in test_hashes if h in overlap)
    assert expected_removed > 0
    no_overlap = docs["no_exact_overlap"]
    assert no_overlap["samples_removed"] == expected_removed
    assert no_overlap["samples_kept"] == len(test_hashes) - expected_removed
    assert full["total_samples"] == len(test_hashes)
    assert docs["without_label_conflicts"]["samples_kept"] <= len(test_hashes)
    assert evaluator.calls[0] == len(test_hashes)


def test_sensitivity_documents_are_descriptive_only_and_carry_provenance(
    tmp_path: Path,
    synthetic_dev_data: tuple[np.ndarray, np.ndarray],
    synthetic_test_data: tuple[np.ndarray, np.ndarray],
    synthetic_dev_hashes: list[str],
) -> None:
    """Variants are flagged as never used for selection and repeat the model identity."""
    docs, *_ = _run(tmp_path, synthetic_dev_data, synthetic_test_data, synthetic_dev_hashes)
    for name, doc in docs.items():
        assert doc["used_for_selection"] is False
        assert doc["purpose"] == "sensitivity_analysis_only"
        assert doc["official_test_modified"] is False
        assert doc["checkpoint_sha256"] == PROVENANCE["checkpoint_sha256"]
        assert doc["split_hash"] == PROVENANCE["split_hash"]
        saved = json.loads(
            (tmp_path / f"final_test_metrics_{name}.json").read_text(encoding="utf-8")
        )
        assert saved["evaluation"] == name
        assert saved["seed"] == 7


def test_comparison_table_lists_full_test_first_with_deltas(
    tmp_path: Path,
    synthetic_dev_data: tuple[np.ndarray, np.ndarray],
    synthetic_test_data: tuple[np.ndarray, np.ndarray],
    synthetic_dev_hashes: list[str],
) -> None:
    """The comparison CSV and Markdown contain the full test plus both variants."""
    _run(tmp_path, synthetic_dev_data, synthetic_test_data, synthetic_dev_hashes)
    frame = pd.read_csv(tmp_path / "final_evaluation_sensitivity_comparison.csv")
    assert frame["evaluation"].tolist() == [
        "full_official_test",
        "no_exact_overlap",
        "without_label_conflicts",
    ]
    assert frame.loc[0, "macro_f1_delta_vs_full"] == 0.0
    assert frame.loc[0, "samples_removed"] == 0
    markdown = (tmp_path / "final_evaluation_sensitivity_comparison.md").read_text("utf-8")
    assert "análises descritivas" in markdown
    assert PROVENANCE["checkpoint_sha256"] in markdown


def test_prediction_files_of_a_variant_index_the_original_test_rows(
    tmp_path: Path,
    synthetic_dev_data: tuple[np.ndarray, np.ndarray],
    synthetic_test_data: tuple[np.ndarray, np.ndarray],
    synthetic_dev_hashes: list[str],
) -> None:
    """The variant predictions keep the original test indices of the kept rows."""
    docs, *_ = _run(tmp_path, synthetic_dev_data, synthetic_test_data, synthetic_dev_hashes)
    preds = pd.read_csv(tmp_path / "final_test_predictions_no_exact_overlap.csv")
    assert len(preds) == docs["no_exact_overlap"]["samples_kept"]
    assert preds["sample_id"].is_monotonic_increasing
    assert preds["sample_id"].max() < len(synthetic_test_data[1])


def test_empty_sensitivity_subset_is_an_error(tmp_path: Path) -> None:
    """If a filter removes every test row there is nothing to evaluate."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 96))
    hashes = hash_series_matrix(X)
    labels = np.array([1, 2, 3, 4])
    evaluator = RecordingEvaluator()
    with pytest.raises(ValueError, match="empty"):
        run_sensitivity_evaluations(
            evaluate=evaluator,
            X_test=X,
            y_test_internal=labels - 1,
            y_test_official=labels,
            dev_hashes=hashes,
            dev_labels=labels.tolist(),
            test_hashes=hashes,
            full_document={},
            provenance=PROVENANCE,
            summary_dir=tmp_path,
            phase=1,
        )


def test_per_class_frame_and_comparison_with_missing_metric() -> None:
    """Metrics without per-class data give an empty frame; absent metrics become NaN deltas."""
    assert per_class_frame({}).empty
    full = {"evaluation": "full_official_test", "total_samples": 10, "macro_f1": 0.5}
    variant = {"evaluation": "v", "total_samples": 8}
    frame = comparison_frame([full, variant])
    assert frame.loc[1, "samples_removed"] == 2
    assert np.isnan(frame.loc[1, "macro_f1"])
