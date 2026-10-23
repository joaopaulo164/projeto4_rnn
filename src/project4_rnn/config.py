"""Declarative YAML configuration management, validation, merging, and hashing."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from project4_rnn.constants import (
    OFFICIAL_DEV_SAMPLES,
    OFFICIAL_TEST_SAMPLES,
    PARSIMONY_THRESHOLD,
    SEQUENCE_LENGTH,
    SPLIT_SEED,
    TRAIN_SEEDS,
)
from project4_rnn.reproducibility import compute_object_hash


@dataclass
class DataConfig:
    """Dataset and split configuration parameters."""

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    dataset_name: str = "ECG5000"
    expected_sequence_length: int = SEQUENCE_LENGTH
    expected_features: int = 1
    expected_dev_samples: int = OFFICIAL_DEV_SAMPLES
    expected_test_samples: int = OFFICIAL_TEST_SAMPLES
    split_seed: int = SPLIT_SEED
    train_samples: int = 400
    val_samples: int = 100
    stratified: bool = True
    binary_mapping: dict[str, list[int]] = field(
        default_factory=lambda: {
            "negative_class": [1],
            "positive_class": [2, 3, 4, 5],
        }
    )
    normalization: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "type": "zscore",
        }
    )


@dataclass
class TrainingConfig:
    """Hyperparameters and training loop configuration."""

    loss: str = "CrossEntropyLoss"
    optimizer: str = "Adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 32
    max_epochs: int = 150
    patience: int = 15
    min_delta: float = 0.0001
    gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    use_class_weights: bool = False
    checkpoint_metric: str = "val_loss"
    checkpoint_mode: str = "min"
    train_seeds: list[int] = field(default_factory=lambda: list(TRAIN_SEEDS))


@dataclass
class SelectionConfig:
    """Decision criteria for pooling and model selection."""

    parsimony_threshold: float = PARSIMONY_THRESHOLD
    primary_metric: str = "val_balanced_accuracy"
    first_tie_breaker: str = "val_f1"
    second_tie_breaker: str = "val_balanced_accuracy_std"
    third_tie_breaker: str = "mean_training_time"
    final_tie_breaker: str = "parameter_count"


@dataclass
class SystemConfig:
    """System-level hardware and runtime settings."""

    device: str = "auto"
    num_workers: int = 0
    artifacts_dir: str = "artifacts"


@dataclass
class ModelConfig:
    """Neural network model architecture settings."""

    architecture: str = "mlp"  # "mlp", "gru", "lstm"
    input_size: int = 140  # 140 for mlp, 1 for rnn
    hidden_size: int = 32  # 32 or 64
    num_layers: int = 1
    bidirectional: bool = False
    batch_first: bool = True
    pooling: str | None = None  # "last", "mean", "max", "mean_max", "best_pooling"
    num_classes: int = 2
    expected_parameters: int | dict[str, int] | None = None


@dataclass
class ExperimentConfig:
    """Full resolved configuration for a specific experiment."""

    project_name: str = "project4_rnn"
    version: str = "1.0.0"
    experiment_id: str = "E01_MLP_32"
    phase: int = 1
    description: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration dataclass to dictionary."""
        return asdict(self)

    def compute_hash(self) -> str:
        """Compute deterministic SHA-256 hash of the configuration dictionary."""
        return compute_object_hash(self.to_dict())

    def save_yaml(self, path: Path) -> None:
        """Save resolved configuration dictionary to a YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def load_yaml(path: Path) -> dict[str, Any]:
    """Safely load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If YAML syntax is invalid.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML configuration format in {path}: expected a mapping.")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries without mutating inputs.

    Args:
        base: Base dictionary.
        override: Override dictionary whose values take precedence.

    Returns:
        New merged dictionary.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def validate_config_dict(cfg: dict[str, Any]) -> None:
    """Validate semantic correctness of configuration parameters.

    Args:
        cfg: Configuration dictionary to validate.

    Raises:
        ValueError: If any parameter is out of bounds or invalid.
    """
    training = cfg.get("training", {})
    if training.get("learning_rate", 0) <= 0:
        raise ValueError(f"learning_rate must be > 0, got {training.get('learning_rate')}")
    if training.get("batch_size", 0) <= 0:
        raise ValueError(f"batch_size must be > 0, got {training.get('batch_size')}")
    if training.get("max_epochs", 0) <= 0:
        raise ValueError(f"max_epochs must be > 0, got {training.get('max_epochs')}")
    if training.get("patience", 0) < 1:
        raise ValueError(f"patience must be >= 1, got {training.get('patience')}")
    if training.get("min_delta", -1) < 0:
        raise ValueError(f"min_delta must be >= 0, got {training.get('min_delta')}")

    model = cfg.get("model", {})
    arch = model.get("architecture")
    if arch is not None:
        if arch not in ("mlp", "gru", "lstm"):
            raise ValueError(f"Invalid model architecture: {arch}. Allowed: 'mlp', 'gru', 'lstm'")

        if arch in ("gru", "lstm"):
            pooling = model.get("pooling")
            valid_poolings = ("last", "mean", "max", "mean_max", "best_pooling")
            if pooling not in valid_poolings:
                raise ValueError(f"Invalid pooling strategy: {pooling}. Allowed: {valid_poolings}")


def build_experiment_config(
    base_config_path: Path,
    experiment_config_path: Path | None = None,
    pooling_override: str | None = None,
) -> ExperimentConfig:
    """Build and validate a fully resolved ExperimentConfig.

    Args:
        base_config_path: Path to configs/base.yaml.
        experiment_config_path: Path to experiment-specific YAML (e.g. configs/phase_1/mlp_32.yaml).
        pooling_override: Strategy to replace 'best_pooling' if applicable ('last', 'mean', etc.).

    Returns:
        Validated ExperimentConfig object.
    """
    base_dict = load_yaml(base_config_path)

    if experiment_config_path is not None:
        exp_dict = load_yaml(experiment_config_path)
        merged = deep_merge(base_dict, exp_dict)
    else:
        merged = base_dict

    if pooling_override is not None:
        if "model" not in merged:
            merged["model"] = {}
        merged["model"]["pooling"] = pooling_override

    validate_config_dict(merged)

    # Instantiate nested dataclasses
    data_cfg = DataConfig(**merged.get("data", {}))
    training_cfg = TrainingConfig(**merged.get("training", {}))
    selection_cfg = SelectionConfig(**merged.get("selection", {}))
    system_cfg = SystemConfig(**merged.get("system", {}))
    model_cfg = ModelConfig(**merged.get("model", {}))

    return ExperimentConfig(
        project_name=merged.get("project_name", "project4_rnn"),
        version=merged.get("version", "1.0.0"),
        experiment_id=merged.get("experiment_id", "E01_MLP_32"),
        phase=merged.get("phase", 1),
        description=merged.get("description", ""),
        data=data_cfg,
        training=training_cfg,
        selection=selection_cfg,
        system=system_cfg,
        model=model_cfg,
    )
