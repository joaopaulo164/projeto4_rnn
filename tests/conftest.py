"""Pytest fixtures and configuration for unit, integration, and smoke test suites."""

from pathlib import Path

import numpy as np
import pytest

from project4_rnn.config import ExperimentConfig, build_experiment_config
from project4_rnn.constants import (
    OFFICIAL_CLASSES,
    SEQUENCE_LENGTH,
)


@pytest.fixture
def synthetic_dev_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate reproducible synthetic development data (100 samples, 96 length, 7 classes)."""
    np.random.seed(42)
    n_samples = 140
    X = np.random.randn(n_samples, SEQUENCE_LENGTH).astype(np.float32)
    y_orig = np.random.choice(list(OFFICIAL_CLASSES), size=n_samples)
    return X, y_orig


@pytest.fixture
def synthetic_test_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate reproducible synthetic test data (70 samples, 96 length, 7 classes)."""
    np.random.seed(42)
    n_samples = 70
    X = np.random.randn(n_samples, SEQUENCE_LENGTH).astype(np.float32)
    y_orig = np.random.choice(list(OFFICIAL_CLASSES), size=n_samples)
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
