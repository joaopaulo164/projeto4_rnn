"""Unit tests for the duplicate and overlap audit."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from project4_rnn.audit.duplicates import (
    build_duplicate_audit,
    build_group_index,
    conflicting_hashes,
    overlap_stats,
    partition_stats,
    render_audit_markdown,
    sensitivity_masks,
    write_audit_artifacts,
)
from project4_rnn.data.splitting import create_grouped_split


def test_internal_duplicate_statistics_with_three_occurrences() -> None:
    """A group of three equal series has excess 2 and involves 3 samples."""
    index = build_group_index(["a", "a", "a", "b", "c", "c"], [1, 1, 1, 2, 3, 3])
    stats = partition_stats(index)
    assert stats["samples"] == 6
    assert stats["unique_series"] == 3
    assert stats["excess_duplicates"] == 3
    assert stats["duplicate_groups"] == 2
    assert stats["samples_involved"] == 5
    assert stats["largest_group"] == 3
    assert stats["conflicting_groups"] == 0


def test_conflicting_group_detection() -> None:
    """Identical series with different labels are flagged, never merged."""
    hashes = ["a", "a", "b", "b"]
    labels = [1, 2, 3, 3]
    assert conflicting_hashes(hashes, labels) == {"a"}
    stats = partition_stats(build_group_index(hashes, labels))
    assert stats["conflicting_groups"] == 1
    assert stats["conflicting_samples"] == 2


def test_cross_partition_pairs_and_label_agreement() -> None:
    """Pairs are the product of group sizes, split into same and different label."""
    a = build_group_index(["x", "x", "y"], [1, 2, 3])
    b = build_group_index(["x", "y", "z"], [1, 3, 4])
    stats = overlap_stats(a, b)
    assert stats["shared_hashes"] == 2
    assert stats["samples_a"] == 3
    assert stats["samples_b"] == 2
    assert stats["pairs"] == 3
    assert stats["pairs_same_label"] == 2
    assert stats["pairs_different_label"] == 1
    assert stats["groups_with_label_conflict"] == 1


def test_group_index_requires_equal_lengths() -> None:
    """Mismatched hashes and labels are rejected."""
    with pytest.raises(ValueError, match="equal length"):
        build_group_index(["a"], [1, 2])


def _audit_fixture(
    synthetic_dev_data: tuple[np.ndarray, np.ndarray],
    synthetic_test_data: tuple[np.ndarray, np.ndarray],
    synthetic_dev_hashes: list[str],
) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    from project4_rnn.data.hashing import hash_series_matrix

    _, y_dev = synthetic_dev_data
    X_test, y_test = synthetic_test_data
    split = create_grouped_split(y_dev, synthetic_dev_hashes, "dataset-hash")
    return build_duplicate_audit(
        dev_hashes=synthetic_dev_hashes,
        dev_labels=y_dev,
        test_hashes=hash_series_matrix(X_test),
        test_labels=y_test,
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        dataset_hash="dataset-hash",
        split_hash=split.split_hash,
        split_method="grouped",
    )


def test_audit_after_grouped_split_has_no_train_validation_overlap(
    synthetic_dev_data: tuple[np.ndarray, np.ndarray],
    synthetic_test_data: tuple[np.ndarray, np.ndarray],
    synthetic_dev_hashes: list[str],
) -> None:
    """With the grouped split no hash is shared between train and validation."""
    audit, tables = _audit_fixture(synthetic_dev_data, synthetic_test_data, synthetic_dev_hashes)
    assert audit["pairwise_overlaps"]["train_vs_validation"]["shared_hashes"] == 0  # type: ignore[index]
    assert audit["partitions"]["train"]["samples"] + audit["partitions"]["validation"][  # type: ignore[index]
        "samples"
    ] == len(synthetic_dev_hashes)
    assert audit["development_vs_test"]["shared_groups"] >= 1  # type: ignore[index]
    assert set(tables) == {
        "duplicate_groups",
        "duplicate_overlap_pairs",
        "duplicate_label_conflicts",
        "duplicate_counts_by_class",
        "duplicate_sample_inventory",
    }
    conflicts = tables["duplicate_label_conflicts"]
    assert len(conflicts) >= 2
    assert (conflicts["n_distinct_labels"] > 1).all()
    inventory = tables["duplicate_sample_inventory"]
    assert len(inventory) == len(synthetic_dev_hashes) + len(synthetic_test_data[1])


def test_class_counts_cover_every_partition_and_class(
    synthetic_dev_data: tuple[np.ndarray, np.ndarray],
    synthetic_test_data: tuple[np.ndarray, np.ndarray],
    synthetic_dev_hashes: list[str],
) -> None:
    """Counts by class exist for each partition, plus the development union."""
    _, tables = _audit_fixture(synthetic_dev_data, synthetic_test_data, synthetic_dev_hashes)
    by_class = tables["duplicate_counts_by_class"]
    assert set(by_class["partition"]) == {"train", "validation", "test", "development"}
    assert set(by_class["official_class"]) == set(range(1, 8))
    dev = by_class[by_class["partition"] == "development"]
    assert dev["samples"].sum() == len(synthetic_dev_hashes)


def test_old_style_index_split_leaks_and_is_detected() -> None:
    """A non-grouped split that separates identical series shows shared hashes."""
    hashes = ["a", "a", "b", "c"]
    labels = np.array([1, 1, 2, 3])
    audit, _ = build_duplicate_audit(
        dev_hashes=hashes,
        dev_labels=labels,
        test_hashes=["z"],
        test_labels=np.array([1]),
        train_idx=np.array([0, 2]),
        val_idx=np.array([1, 3]),
    )
    overlap = audit["pairwise_overlaps"]["train_vs_validation"]
    assert overlap["shared_hashes"] == 1
    assert overlap["pairs_same_label"] == 1


def test_audit_artifacts_are_written(
    tmp_path: Path,
    synthetic_dev_data: tuple[np.ndarray, np.ndarray],
    synthetic_test_data: tuple[np.ndarray, np.ndarray],
    synthetic_dev_hashes: list[str],
) -> None:
    """All required audit files are produced and the markdown renders."""
    audit, tables = _audit_fixture(synthetic_dev_data, synthetic_test_data, synthetic_dev_hashes)
    written = write_audit_artifacts(audit, tables, tmp_path)  # type: ignore[arg-type]
    names = {p.name for p in written}
    assert names == {
        "duplicate_overlap_audit.json",
        "duplicate_overlap_audit.md",
        "duplicate_groups.csv",
        "duplicate_overlap_pairs.csv",
        "duplicate_label_conflicts.csv",
        "duplicate_counts_by_class.csv",
        "duplicate_sample_inventory.csv",
        "official_development_test_overlap.json",
    }
    assert json.loads((tmp_path / "duplicate_overlap_audit.json").read_text("utf-8"))["global"]
    assert "Auditoria de Duplicatas" in render_audit_markdown(audit)


def test_sensitivity_masks_filter_overlap_and_conflicts_without_touching_labels() -> None:
    """Masks drop test rows sharing a dev hash, and rows in conflicting hash groups."""
    dev_hashes = ["a", "b", "c", "c"]
    dev_labels = [1, 2, 3, 4]
    test_hashes = ["a", "d", "c", "e"]
    test_labels = [1, 5, 3, 6]
    masks = sensitivity_masks(dev_hashes, dev_labels, test_hashes, test_labels)
    assert masks["no_exact_overlap"].tolist() == [False, True, False, True]
    assert masks["without_label_conflicts"].tolist() == [True, True, False, True]
    assert test_labels == [1, 5, 3, 6]
