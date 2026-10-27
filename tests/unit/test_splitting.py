"""Unit tests for the hash-grouped, stratified train/validation split."""

import json
from pathlib import Path

import numpy as np
import pytest

from project4_rnn.constants import OFFICIAL_CLASSES
from project4_rnn.data.splitting import (
    SplitIntegrityError,
    compute_payload_checksum,
    create_grouped_split,
    create_or_load_split,
    read_split_identity,
    score_fold,
    verify_split_integrity,
)

DATASET_HASH = "dataset-hash"


def _y_and_hashes(
    synthetic_dev_data: tuple[np.ndarray, np.ndarray], hashes: list[str]
) -> tuple[np.ndarray, list[str]]:
    return synthetic_dev_data[1], hashes


def test_grouped_split_is_disjoint_complete_and_leak_free(
    synthetic_dev_data: tuple[np.ndarray, np.ndarray], synthetic_dev_hashes: list[str]
) -> None:
    """Indices are disjoint and complete; no hash crosses train and validation."""
    y, hashes = _y_and_hashes(synthetic_dev_data, synthetic_dev_hashes)
    result = create_grouped_split(y, hashes, DATASET_HASH)

    train, val = set(result.train_idx.tolist()), set(result.val_idx.tolist())
    assert not train & val
    assert train | val == set(range(len(y)))
    assert {hashes[i] for i in train}.isdisjoint({hashes[i] for i in val})
    assert result.payload["shared_hashes_train_val"] == 0
    assert set(y[result.train_idx]) == set(OFFICIAL_CLASSES)
    assert set(y[result.val_idx]) == set(OFFICIAL_CLASSES)
    assert 0.1 < len(val) / len(y) < 0.3


def test_conflicting_groups_stay_whole_and_labels_are_preserved(
    synthetic_dev_data: tuple[np.ndarray, np.ndarray], synthetic_dev_hashes: list[str]
) -> None:
    """Every occurrence of a conflicting hash lands in one partition with its own label."""
    y, hashes = _y_and_hashes(synthetic_dev_data, synthetic_dev_hashes)
    original_y = y.copy()
    result = create_grouped_split(y, hashes, DATASET_HASH)

    by_hash: dict[str, set[int]] = {}
    for i, h in enumerate(hashes):
        by_hash.setdefault(h, set()).add(i)
    conflicting = [h for h, idx in by_hash.items() if len({int(y[i]) for i in idx}) > 1]
    assert conflicting
    val = set(result.val_idx.tolist())
    for h in conflicting:
        members = by_hash[h]
        assert members <= val or not (members & val)
    np.testing.assert_array_equal(y, original_y)
    assert result.payload["conflicting_groups"]["count"] == len(conflicting)


def test_split_is_deterministic(
    synthetic_dev_data: tuple[np.ndarray, np.ndarray], synthetic_dev_hashes: list[str]
) -> None:
    """Two creations with the same inputs give identical indices and split hash."""
    y, hashes = _y_and_hashes(synthetic_dev_data, synthetic_dev_hashes)
    first = create_grouped_split(y, hashes, DATASET_HASH)
    second = create_grouped_split(y, hashes, DATASET_HASH)
    np.testing.assert_array_equal(first.train_idx, second.train_idx)
    np.testing.assert_array_equal(first.val_idx, second.val_idx)
    assert first.split_hash == second.split_hash


def test_fold_selection_depends_only_on_size_and_classes(
    synthetic_dev_data: tuple[np.ndarray, np.ndarray], synthetic_dev_hashes: list[str]
) -> None:
    """The selected fold has the minimal score among the eligible candidates."""
    y, hashes = _y_and_hashes(synthetic_dev_data, synthetic_dev_hashes)
    payload = create_grouped_split(y, hashes, DATASET_HASH).payload
    eligible = [c for c in payload["fold_candidates"] if c["eligible"]]
    best = min(c["score"] for c in eligible)
    assert payload["selected_score"] == pytest.approx(best)
    assert payload["selection_criterion"]


def test_score_fold_prefers_the_official_distribution() -> None:
    """A validation set matching the official class mix scores lower than a skewed one."""
    y = np.repeat(np.arange(1, 8), 20)
    balanced_val = np.arange(0, 140, 5)
    skewed_val = np.arange(0, 28)
    other = lambda val: np.setdiff1d(np.arange(len(y)), val)  # noqa: E731
    good = score_fold(y, other(balanced_val), balanced_val, 0.2)
    bad = score_fold(y, other(skewed_val), skewed_val, 0.2)
    assert good < bad


def test_persistence_and_reuse(
    tmp_path: Path,
    synthetic_dev_data: tuple[np.ndarray, np.ndarray],
    synthetic_dev_hashes: list[str],
) -> None:
    """The split is saved with a checksum and reloaded unchanged."""
    y, hashes = _y_and_hashes(synthetic_dev_data, synthetic_dev_hashes)
    path = tmp_path / "split_indices.json"
    created = create_or_load_split(y, hashes, path, DATASET_HASH)
    document = json.loads(path.read_text("utf-8"))
    assert document["checksum_sha256"] == compute_payload_checksum(document)
    assert document["train_samples"] == len(created.train_idx)
    for key in (
        "method",
        "method_version",
        "split_seed",
        "n_splits",
        "selected_fold",
        "selection_criterion",
        "selected_score",
        "hash_algorithm",
        "hash_dtype",
        "hash_endianness",
        "num_groups",
        "conflicting_groups",
        "train_class_distribution",
        "val_class_distribution",
        "shared_hashes_train_val",
        "generated_at",
    ):
        assert key in document

    reloaded = create_or_load_split(y, hashes, path, DATASET_HASH)
    np.testing.assert_array_equal(created.train_idx, reloaded.train_idx)
    assert reloaded.split_hash == created.split_hash
    assert read_split_identity(path) == created.split_hash


def test_force_recreate_rewrites_the_file(
    tmp_path: Path,
    synthetic_dev_data: tuple[np.ndarray, np.ndarray],
    synthetic_dev_hashes: list[str],
) -> None:
    """force_recreate regenerates even when a valid split exists."""
    y, hashes = _y_and_hashes(synthetic_dev_data, synthetic_dev_hashes)
    path = tmp_path / "split_indices.json"
    first = create_or_load_split(y, hashes, path, DATASET_HASH)
    forced = create_or_load_split(y, hashes, path, DATASET_HASH, force_recreate=True)
    assert forced.split_hash == first.split_hash


def _saved_split(
    tmp_path: Path, synthetic_dev_data: tuple[np.ndarray, np.ndarray], hashes: list[str]
) -> tuple[Path, np.ndarray]:
    y = synthetic_dev_data[1]
    path = tmp_path / "split_indices.json"
    create_or_load_split(y, hashes, path, DATASET_HASH)
    return path, y


def _tamper(path: Path, mutate: "object") -> None:
    document = json.loads(path.read_text("utf-8"))
    mutate(document)  # type: ignore[operator]
    path.write_text(json.dumps(document), encoding="utf-8")


def test_tampered_checksum_is_rejected(
    tmp_path: Path,
    synthetic_dev_data: tuple[np.ndarray, np.ndarray],
    synthetic_dev_hashes: list[str],
) -> None:
    """Editing the split file without refreshing the checksum fails fast."""
    path, y = _saved_split(tmp_path, synthetic_dev_data, synthetic_dev_hashes)
    _tamper(path, lambda d: d.update({"selected_score": 123.0}))
    with pytest.raises(SplitIntegrityError, match="checksum"):
        create_or_load_split(y, synthetic_dev_hashes, path, DATASET_HASH)
    with pytest.raises(SplitIntegrityError, match="checksum"):
        read_split_identity(path)


def test_legacy_split_is_rejected_with_actionable_message(
    tmp_path: Path,
    synthetic_dev_data: tuple[np.ndarray, np.ndarray],
    synthetic_dev_hashes: list[str],
) -> None:
    """The old index-based split file is detected as obsolete."""
    y = synthetic_dev_data[1]
    path = tmp_path / "split_indices.json"
    path.write_text(json.dumps({"train_indices": [0], "val_indices": [1]}), encoding="utf-8")
    with pytest.raises(SplitIntegrityError, match="obsolete") as info:
        create_or_load_split(y, synthetic_dev_hashes, path, DATASET_HASH)
    assert "Suggested action" in str(info.value)
    with pytest.raises(SplitIntegrityError, match="hash-grouped"):
        read_split_identity(path)


def test_split_for_another_dataset_is_rejected(
    tmp_path: Path,
    synthetic_dev_data: tuple[np.ndarray, np.ndarray],
    synthetic_dev_hashes: list[str],
) -> None:
    """A persisted split is tied to the dataset hash it was created for."""
    path, y = _saved_split(tmp_path, synthetic_dev_data, synthetic_dev_hashes)
    with pytest.raises(SplitIntegrityError, match="current dataset"):
        create_or_load_split(y, synthetic_dev_hashes, path, "another-dataset")


def test_split_with_changed_hashes_is_rejected(
    tmp_path: Path,
    synthetic_dev_data: tuple[np.ndarray, np.ndarray],
    synthetic_dev_hashes: list[str],
) -> None:
    """Indices must stay consistent with the series hashes of the current data."""
    path, y = _saved_split(tmp_path, synthetic_dev_data, synthetic_dev_hashes)
    changed = list(synthetic_dev_hashes)
    changed[0] = "f" * 64
    with pytest.raises(SplitIntegrityError, match="consistent with the series hashes"):
        create_or_load_split(y, changed, path, DATASET_HASH)


def test_split_with_other_parameters_is_rejected(
    tmp_path: Path,
    synthetic_dev_data: tuple[np.ndarray, np.ndarray],
    synthetic_dev_hashes: list[str],
) -> None:
    """The persisted seed and n_splits must equal the configuration."""
    path, y = _saved_split(tmp_path, synthetic_dev_data, synthetic_dev_hashes)
    with pytest.raises(SplitIntegrityError, match="split parameters"):
        create_or_load_split(y, synthetic_dev_hashes, path, DATASET_HASH, seed=7)


def test_missing_split_file_has_no_identity(tmp_path: Path) -> None:
    """Reading the identity of a nonexistent split raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_split_identity(tmp_path / "missing.json")


def test_invalid_validation_ratio_is_rejected(
    synthetic_dev_data: tuple[np.ndarray, np.ndarray], synthetic_dev_hashes: list[str]
) -> None:
    """The ratio must equal 1/n_splits for StratifiedGroupKFold."""
    y, hashes = _y_and_hashes(synthetic_dev_data, synthetic_dev_hashes)
    with pytest.raises(ValueError, match="1/n_splits"):
        create_grouped_split(y, hashes, DATASET_HASH, val_ratio=0.3)


# --- fail-fast integrity conditions -------------------------------------------------------


@pytest.fixture
def small_case() -> tuple[np.ndarray, list[str]]:
    """Fourteen samples, two per class, all with distinct hashes."""
    y = np.repeat(np.arange(1, 8), 2)
    hashes = [f"h{i}" for i in range(14)]
    return y, hashes


def test_integrity_accepts_a_valid_split(small_case: tuple[np.ndarray, list[str]]) -> None:
    """One sample of each class on each side, distinct hashes: every condition holds."""
    y, hashes = small_case
    verify_split_integrity(np.arange(0, 14, 2), np.arange(1, 14, 2), y, hashes)


def test_integrity_repeated_indices(small_case: tuple[np.ndarray, list[str]]) -> None:
    """A repeated index inside a partition is reported."""
    y, hashes = small_case
    with pytest.raises(SplitIntegrityError, match="no repeated indices") as info:
        verify_split_integrity(np.array([0, 0, 2]), np.array([3]), y, hashes)
    assert info.value.offending == [0]


def test_integrity_out_of_range(small_case: tuple[np.ndarray, list[str]]) -> None:
    """Indices outside the dataset are reported."""
    y, hashes = small_case
    with pytest.raises(SplitIntegrityError, match="within"):
        verify_split_integrity(np.array([0, 99]), np.array([3]), y, hashes)


def test_integrity_overlapping_indices(small_case: tuple[np.ndarray, list[str]]) -> None:
    """The same index in both partitions is reported."""
    y, hashes = small_case
    with pytest.raises(SplitIntegrityError, match="disjoint") as info:
        verify_split_integrity(np.array([0, 2]), np.array([2, 3]), y, hashes)
    assert info.value.offending == [2]


def test_integrity_incomplete_coverage(small_case: tuple[np.ndarray, list[str]]) -> None:
    """Dropping samples from both partitions is reported."""
    y, hashes = small_case
    with pytest.raises(SplitIntegrityError, match="cover every"):
        verify_split_integrity(np.array([0, 2]), np.array([3, 5]), y, hashes)


def test_integrity_missing_class(small_case: tuple[np.ndarray, list[str]]) -> None:
    """A partition without all seven classes is reported."""
    y, hashes = small_case
    train = np.array([0, 2, 4, 6, 8, 10, 12, 1, 3, 5, 7, 9])
    val = np.array([11, 13])
    with pytest.raises(SplitIntegrityError, match="classes present in the validation") as info:
        verify_split_integrity(train, val, y, hashes)
    assert info.value.observed == [6, 7]


def test_integrity_shared_hash_names_the_offender() -> None:
    """A hash on both sides of the split is reported with the hash itself."""
    y = np.repeat(np.arange(1, 8), 2)
    hashes = [f"h{i}" for i in range(14)]
    hashes[1] = hashes[0]
    train = np.array([0, 2, 4, 6, 8, 10, 12])
    val = np.array([1, 3, 5, 7, 9, 11, 13])
    with pytest.raises(SplitIntegrityError, match="indivisible") as info:
        verify_split_integrity(train, val, y, hashes)
    assert info.value.offending == ["h0"]
    assert info.value.suggested_action
    assert "Expected" in str(info.value)
    assert "Observed" in str(info.value)


def test_integrity_hash_count_mismatch() -> None:
    """One hash per sample is required."""
    with pytest.raises(SplitIntegrityError, match="one series hash per sample"):
        verify_split_integrity(np.array([0]), np.array([1]), np.array([1, 2]), ["a"])


def test_error_message_truncates_long_offender_lists() -> None:
    """Long offending lists are shortened in the message but kept on the exception."""
    error = SplitIntegrityError("c", 0, 30, offending=list(range(30)), suggested_action="fix")
    assert "(+20 more)" in str(error)
    assert len(error.offending) == 30
