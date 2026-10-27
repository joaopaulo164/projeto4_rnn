"""Pytest fixtures and configuration for unit, integration, and smoke test suites."""

from pathlib import Path

import numpy as np
import pytest

from project4_rnn.config import ExperimentConfig, build_experiment_config
from project4_rnn.data.hashing import hash_series_matrix
from project4_rnn.data.synthetic import make_synthetic_series, make_synthetic_test_series
from project4_rnn.reproducibility import set_seed


@pytest.fixture(autouse=True)
def _deterministic_seed() -> None:
    """Lock every random number generator before each test."""
    set_seed(42, deterministic=True)


@pytest.fixture
def synthetic_dev_data() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic development series (float64) with official labels 1..7 and duplicate groups."""
    return make_synthetic_series()


@pytest.fixture
def synthetic_test_data(
    synthetic_dev_data: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic test series partially coinciding with the synthetic development series."""
    X_dev, y_dev = synthetic_dev_data
    return make_synthetic_test_series(X_dev, y_dev)


@pytest.fixture
def synthetic_dev_hashes(synthetic_dev_data: tuple[np.ndarray, np.ndarray]) -> list[str]:
    """Series hashes of the synthetic development data."""
    return hash_series_matrix(synthetic_dev_data[0])


@pytest.fixture
def temp_artifacts_dir(tmp_path: Path) -> Path:
    """Provide temporary directory for test artifacts."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return artifacts


@pytest.fixture
def base_config() -> ExperimentConfig:
    """Return base ExperimentConfig instance."""
    return build_experiment_config(Path("configs/base.yaml"))
