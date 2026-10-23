"""Pytest fixtures and configuration for unit, integration, and smoke test suites."""

from pathlib import Path

import numpy as np
import pytest

from project4_rnn.config import ExperimentConfig, build_experiment_config
from project4_rnn.constants import (
    OFFICIAL_DEV_SAMPLES,
    OFFICIAL_TEST_SAMPLES,
    SEQUENCE_LENGTH,
)


@pytest.fixture
def synthetic_dev_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate reproducible synthetic development data (500 samples, 140 length)."""
    np.random.seed(42)
    X = np.random.randn(OFFICIAL_DEV_SAMPLES, SEQUENCE_LENGTH).astype(np.float32)
    # Multiclass labels: class 1 (approx 290), class 2 (approx 180), classes 3-5 (approx 30)
    y_orig = np.random.choice(
        [1, 2, 3, 4, 5], size=OFFICIAL_DEV_SAMPLES, p=[0.58, 0.35, 0.03, 0.02, 0.02]
    )
    return X, y_orig


@pytest.fixture
def synthetic_test_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate reproducible synthetic test data (4500 samples, 140 length)."""
    np.random.seed(42)
    X = np.random.randn(OFFICIAL_TEST_SAMPLES, SEQUENCE_LENGTH).astype(np.float32)
    y_orig = np.random.choice(
        [1, 2, 3, 4, 5], size=OFFICIAL_TEST_SAMPLES, p=[0.58, 0.35, 0.03, 0.02, 0.02]
    )
    return X, y_orig


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
