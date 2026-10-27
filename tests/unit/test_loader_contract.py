"""Dataset-contract tests for the raw ElectricDevices loader, using generated fixture files."""

from pathlib import Path

import numpy as np
import pytest

from project4_rnn.constants import (
    OFFICIAL_TEST_SAMPLES,
    OFFICIAL_TRAIN_SAMPLES,
    SEQUENCE_LENGTH,
)
from project4_rnn.data.loader import (
    _parse_arff_file,
    _parse_ts_file,
    compute_dataset_hash,
    load_raw_electricdevices,
)
from project4_rnn.data.validation import DatasetContractError

pytestmark = pytest.mark.unit


def _matrix(rows: int, length: int = SEQUENCE_LENGTH, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(rows, length))


def _labels(rows: int, classes: int = 7) -> np.ndarray:
    return (np.arange(rows) % classes + 1).astype(np.int64)


def _write_txt(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    np.savetxt(path, np.column_stack([y, X]), fmt="%.10g")


def _write_pair(
    raw: Path,
    n_train: int = OFFICIAL_TRAIN_SAMPLES,
    n_test: int = OFFICIAL_TEST_SAMPLES,
    length: int = SEQUENCE_LENGTH,
    classes: int = 7,
    X_train: np.ndarray | None = None,
) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    X_tr = X_train if X_train is not None else _matrix(n_train, length, 1)
    _write_txt(raw / "ElectricDevices_TRAIN.txt", X_tr, _labels(len(X_tr), classes))
    _write_txt(
        raw / "ElectricDevices_TEST.txt", _matrix(n_test, length, 2), _labels(n_test, classes)
    )


def test_valid_contract_loads_float64_and_builds_manifest(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_pair(raw)
    X_tr, y_tr, X_te, y_te, manifest = load_raw_electricdevices(
        raw, verify_checksums=True, processed_dir=tmp_path / "processed"
    )
    assert X_tr.shape == (OFFICIAL_TRAIN_SAMPLES, SEQUENCE_LENGTH)
    assert X_te.shape == (OFFICIAL_TEST_SAMPLES, SEQUENCE_LENGTH)
    assert X_tr.dtype == np.float64
    assert set(np.unique(y_tr)) == set(range(1, 8)) == set(np.unique(y_te))
    assert manifest["contract_status"] == "PASSED"
    assert manifest["dataset_hash"] == compute_dataset_hash(manifest)
    assert (tmp_path / "processed" / "dataset_manifest.json").is_file()


def test_dataset_hash_does_not_depend_on_load_time(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_pair(raw, n_train=OFFICIAL_TRAIN_SAMPLES)
    first = load_raw_electricdevices(raw, processed_dir=None)[-1]
    second = load_raw_electricdevices(raw, processed_dir=None)[-1]
    assert first["loaded_timestamp"] != "" and first["dataset_hash"] == second["dataset_hash"]


def test_missing_files_raise_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_electricdevices(tmp_path, processed_dir=None)


def test_wrong_train_count_is_rejected(tmp_path: Path) -> None:
    _write_pair(tmp_path, n_train=100)
    with pytest.raises(DatasetContractError, match="training sample count"):
        load_raw_electricdevices(tmp_path, processed_dir=None)


def test_wrong_test_count_is_rejected(tmp_path: Path) -> None:
    _write_pair(tmp_path, n_test=100)
    with pytest.raises(DatasetContractError, match="test sample count"):
        load_raw_electricdevices(tmp_path, processed_dir=None)


def test_wrong_sequence_length_is_rejected(tmp_path: Path) -> None:
    _write_pair(tmp_path, length=720)
    with pytest.raises(DatasetContractError, match="Sequence length mismatch"):
        load_raw_electricdevices(tmp_path, processed_dir=None)


def test_wrong_class_domain_is_rejected(tmp_path: Path) -> None:
    _write_pair(tmp_path, classes=6)
    with pytest.raises(DatasetContractError, match="class domain"):
        load_raw_electricdevices(tmp_path, processed_dir=None)


def test_nan_is_rejected(tmp_path: Path) -> None:
    X = _matrix(OFFICIAL_TRAIN_SAMPLES, seed=1)
    X[3, 5] = np.nan
    _write_pair(tmp_path, X_train=X)
    with pytest.raises(DatasetContractError, match="NaN"):
        load_raw_electricdevices(tmp_path, processed_dir=None)


def test_infinite_values_are_rejected(tmp_path: Path) -> None:
    X = _matrix(OFFICIAL_TRAIN_SAMPLES, seed=1)
    X[3, 5] = np.inf
    _write_pair(tmp_path, X_train=X)
    with pytest.raises(DatasetContractError, match="infinite"):
        load_raw_electricdevices(tmp_path, processed_dir=None)


def test_ts_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "ElectricDevices_TRAIN.ts").write_text("@data\n1,2,3:1\n", encoding="utf-8")
    (tmp_path / "ElectricDevices_TEST.ts").write_text("@data\n1,2,3:1\n", encoding="utf-8")
    with pytest.raises(DatasetContractError, match="MD5 checksum mismatch"):
        load_raw_electricdevices(tmp_path, verify_checksums=True, processed_dir=None)


def test_parse_ts_supports_colon_and_plain_layouts(tmp_path: Path) -> None:
    path = tmp_path / "x.ts"
    path.write_text(
        "# comment\n@problemName X\n@data\n1.5,2.5,3.5:2\n4,5,6,3\n\n",
        encoding="utf-8",
    )
    X, y = _parse_ts_file(path)
    assert X.tolist() == [[1.5, 2.5, 3.5], [4.0, 5.0, 6.0]]
    assert y.tolist() == [2, 3]


def test_parse_arff_handles_trailing_and_leading_labels(tmp_path: Path) -> None:
    path = tmp_path / "x.arff"
    path.write_text(
        "% c\n@relation r\n@data\n1.0,2.0,3\n'4',5.0,6.0\n",
        encoding="utf-8",
    )
    X, y = _parse_arff_file(path)
    assert y.tolist() == [3, 4]
    assert X.tolist() == [[1.0, 2.0], [5.0, 6.0]]


def test_arff_files_are_discovered(tmp_path: Path) -> None:
    for name in ("TRAIN", "TEST"):
        (tmp_path / f"ElectricDevices_{name}.arff").write_text("@data\n1.0,2.0,3\n", "utf-8")
    with pytest.raises(DatasetContractError, match="training sample count"):
        load_raw_electricdevices(tmp_path, processed_dir=None)
