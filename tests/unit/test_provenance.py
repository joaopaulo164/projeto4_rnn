"""Unit tests for the provenance of the final model selection and the final test metrics."""

import json
from pathlib import Path

import pytest
from run_fixtures import DATASET_HASH, SPLIT_HASH, make_run_metadata, write_run

from project4_rnn.config import ExperimentConfig, build_experiment_config
from project4_rnn.experiments.provenance import (
    FINAL_SELECTION_REQUIRED_FIELDS,
    FINAL_TEST_REQUIRED_FIELDS,
    ProvenanceError,
    StaleArtifactError,
    build_final_selection_provenance,
    build_final_test_provenance,
    check_identity,
    select_best_seed,
    validate_final_selection,
    validate_final_test_metrics,
    verify_checkpoint_integrity,
)
from project4_rnn.experiments.timing import IncompleteRunError
from project4_rnn.reproducibility import compute_file_hash


@pytest.fixture
def config() -> ExperimentConfig:
    return build_experiment_config(Path("configs/base.yaml"), Path("configs/phase_1/mlp_32.yaml"))


def _experiment_with_seeds(tmp_path: Path) -> Path:
    for seed, loss in ((7, 0.40), (21, 0.30), (42, 0.30), (84, 0.50)):
        write_run(tmp_path, make_run_metadata("E01_MLP_32", seed, val_loss=loss))
    return tmp_path / "runs" / "E01_MLP_32"


def test_best_seed_is_lowest_validation_loss_with_lowest_seed_tiebreak(tmp_path: Path) -> None:
    """Seeds 21 and 42 tie on loss; the lowest seed wins. The criterion never uses the test set."""
    best = select_best_seed(_experiment_with_seeds(tmp_path), SPLIT_HASH, DATASET_HASH)
    assert best["seed"] == 21
    assert best["best_val_loss"] == 0.30
    assert best["best_epoch"] == 2
    assert len(best["checkpoint_sha256"]) == 64
    assert best["checkpoint_sha256"] == compute_file_hash(best["checkpoint"], "sha256")


def test_best_seed_is_never_defaulted(tmp_path: Path) -> None:
    """With only seed 7 available, 7 is chosen; there is no implicit default of 42."""
    write_run(tmp_path, make_run_metadata("E01_MLP_32", 7))
    best = select_best_seed(tmp_path / "runs" / "E01_MLP_32", SPLIT_HASH)
    assert best["seed"] == 7


def test_best_seed_rejects_runs_from_another_split(tmp_path: Path) -> None:
    """Metadata trained on a different split is stale."""
    experiment_dir = _experiment_with_seeds(tmp_path)
    with pytest.raises(StaleArtifactError, match="split_hash"):
        select_best_seed(experiment_dir, "another-split")
    with pytest.raises(StaleArtifactError, match="dataset_hash"):
        select_best_seed(experiment_dir, SPLIT_HASH, "another-dataset")


def test_best_seed_reports_incomplete_metadata(tmp_path: Path) -> None:
    """A missing best_val_loss is reported with the run id."""
    metadata = make_run_metadata("E01_MLP_32", 7)
    del metadata["best_val_loss"]
    write_run(tmp_path, metadata)
    with pytest.raises(IncompleteRunError, match="E01_MLP_32/seed_7"):
        select_best_seed(tmp_path / "runs" / "E01_MLP_32", SPLIT_HASH)


def test_best_seed_requires_a_checkpoint_and_runs(tmp_path: Path) -> None:
    """A run without checkpoint, or an experiment without runs, is an error."""
    write_run(tmp_path, make_run_metadata("E01_MLP_32", 7), with_checkpoint=False)
    with pytest.raises(FileNotFoundError, match="Checkpoint missing"):
        select_best_seed(tmp_path / "runs" / "E01_MLP_32", SPLIT_HASH)
    empty = tmp_path / "runs" / "E99"
    empty.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No completed seed runs"):
        select_best_seed(empty, SPLIT_HASH)


def test_selection_provenance_has_every_required_field(
    tmp_path: Path, config: ExperimentConfig
) -> None:
    """The selection block records experiment, model, seed, checkpoint, config and hashes."""
    best = select_best_seed(_experiment_with_seeds(tmp_path), SPLIT_HASH, DATASET_HASH)
    block = build_final_selection_provenance(config, best, DATASET_HASH, SPLIT_HASH)
    for field in FINAL_SELECTION_REQUIRED_FIELDS:
        assert field in block
    assert block["selected_seed"] == 21
    assert block["architecture"] == "mlp"
    assert block["hidden_size"] == 32
    assert block["config_hash"] == config.compute_hash()
    assert block["resolved_config"]["experiment_id"] == "E01_MLP_32"
    assert block["split_hash"] == SPLIT_HASH
    assert block["dataset_hash"] == DATASET_HASH
    assert "test set not used" in block["seed_selection_criterion"]
    validate_final_selection(block)
    json.dumps(block)


def test_selection_validation_rejects_missing_and_invalid_fields(
    tmp_path: Path, config: ExperimentConfig
) -> None:
    """Missing keys, null values (except pooling), NaN metrics and short digests fail."""
    best = select_best_seed(_experiment_with_seeds(tmp_path), SPLIT_HASH, DATASET_HASH)
    block = build_final_selection_provenance(config, best, DATASET_HASH, SPLIT_HASH)

    block["pooling"] = None  # a dense model has no pooling; the key must exist though
    validate_final_selection(block)

    missing = {k: v for k, v in block.items() if k != "checkpoint_sha256"}
    with pytest.raises(ProvenanceError, match="checkpoint_sha256"):
        validate_final_selection(missing)
    with pytest.raises(ProvenanceError, match="pooling"):
        validate_final_selection({k: v for k, v in block.items() if k != "pooling"})
    with pytest.raises(ProvenanceError, match="split_hash"):
        validate_final_selection({**block, "split_hash": None})
    with pytest.raises(ProvenanceError, match="not finite"):
        validate_final_selection({**block, "best_val_loss": float("nan")})
    with pytest.raises(ProvenanceError, match="64-character"):
        validate_final_selection({**block, "checkpoint_sha256": "abc"})


def test_final_test_provenance_repeats_the_selection_identity(
    tmp_path: Path, config: ExperimentConfig
) -> None:
    """The test metrics repeat experiment, seed, checkpoint, SHA-256, epoch and hashes."""
    best = select_best_seed(_experiment_with_seeds(tmp_path), SPLIT_HASH, DATASET_HASH)
    selection = build_final_selection_provenance(config, best, DATASET_HASH, SPLIT_HASH)
    test_block = build_final_test_provenance(selection)
    assert set(test_block) == set(FINAL_TEST_REQUIRED_FIELDS)
    assert test_block["seed"] == selection["selected_seed"]
    assert test_block["checkpoint_sha256"] == selection["checkpoint_sha256"]
    assert test_block["best_epoch"] == selection["best_epoch"]
    validate_final_test_metrics(test_block)
    with pytest.raises(ProvenanceError, match="seed"):
        validate_final_test_metrics({k: v for k, v in test_block.items() if k != "seed"})


def test_checkpoint_integrity_detects_modification(
    tmp_path: Path, config: ExperimentConfig
) -> None:
    """A checkpoint altered after selection no longer matches its recorded SHA-256."""
    best = select_best_seed(_experiment_with_seeds(tmp_path), SPLIT_HASH, DATASET_HASH)
    selection = build_final_selection_provenance(config, best, DATASET_HASH, SPLIT_HASH)
    verify_checkpoint_integrity(selection)
    Path(selection["checkpoint"]).write_bytes(b"tampered")
    with pytest.raises(ProvenanceError, match="changed after selection"):
        verify_checkpoint_integrity(selection)
    Path(selection["checkpoint"]).unlink()
    with pytest.raises(FileNotFoundError):
        verify_checkpoint_integrity(selection)


def test_check_identity_detects_stale_artifacts() -> None:
    """Dataset and split hashes of an artifact must equal the expected ones when given."""
    payload = {"split_hash": "a", "dataset_hash": "b"}
    check_identity("x", payload, "b", "a")
    check_identity("x", payload, None, None)
    with pytest.raises(StaleArtifactError, match="clean-artifacts"):
        check_identity("x", payload, "b", "other")
    with pytest.raises(StaleArtifactError, match="dataset_hash"):
        check_identity("x", {"split_hash": "a"}, "b", "a")
