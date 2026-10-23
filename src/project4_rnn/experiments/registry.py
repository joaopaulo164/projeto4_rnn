"""Experiment registry and phase definitions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentSpec:
    """Specification of an experimental configuration."""

    experiment_id: str
    phase: int
    config_file: str
    architecture: str
    hidden_size: int
    pooling: str | None
    description: str


EXPERIMENT_REGISTRY: dict[str, ExperimentSpec] = {
    "E01_MLP_32": ExperimentSpec(
        experiment_id="E01_MLP_32",
        phase=1,
        config_file="configs/phase_1/mlp_32.yaml",
        architecture="mlp",
        hidden_size=32,
        pooling=None,
        description="MLP baseline model with 32 units",
    ),
    "E02_GRU_32_LAST": ExperimentSpec(
        experiment_id="E02_GRU_32_LAST",
        phase=2,
        config_file="configs/phase_2/gru_32_last.yaml",
        architecture="gru",
        hidden_size=32,
        pooling="last",
        description="GRU-32 with last-state pooling",
    ),
    "E03_GRU_32_MEAN": ExperimentSpec(
        experiment_id="E03_GRU_32_MEAN",
        phase=2,
        config_file="configs/phase_2/gru_32_mean.yaml",
        architecture="gru",
        hidden_size=32,
        pooling="mean",
        description="GRU-32 with mean temporal pooling",
    ),
    "E04_GRU_32_MAX": ExperimentSpec(
        experiment_id="E04_GRU_32_MAX",
        phase=2,
        config_file="configs/phase_2/gru_32_max.yaml",
        architecture="gru",
        hidden_size=32,
        pooling="max",
        description="GRU-32 with max temporal pooling",
    ),
    "E05_GRU_32_MEAN_MAX": ExperimentSpec(
        experiment_id="E05_GRU_32_MEAN_MAX",
        phase=2,
        config_file="configs/phase_2/gru_32_mean_max.yaml",
        architecture="gru",
        hidden_size=32,
        pooling="mean_max",
        description="GRU-32 with concatenated mean+max temporal pooling",
    ),
    "E06_LSTM_32_BEST_POOLING": ExperimentSpec(
        experiment_id="E06_LSTM_32_BEST_POOLING",
        phase=3,
        config_file="configs/phase_3/lstm_32_best_pooling.yaml",
        architecture="lstm",
        hidden_size=32,
        pooling="best_pooling",
        description="LSTM-32 with winning pooling strategy from Phase 2",
    ),
    "E07_GRU_64_BEST_POOLING": ExperimentSpec(
        experiment_id="E07_GRU_64_BEST_POOLING",
        phase=4,
        config_file="configs/phase_4/gru_64_best_pooling.yaml",
        architecture="gru",
        hidden_size=64,
        pooling="best_pooling",
        description="GRU-64 (increased capacity) with winning pooling strategy from Phase 2",
    ),
    "E08_LSTM_64_BEST_POOLING": ExperimentSpec(
        experiment_id="E08_LSTM_64_BEST_POOLING",
        phase=4,
        config_file="configs/phase_4/lstm_64_best_pooling.yaml",
        architecture="lstm",
        hidden_size=64,
        pooling="best_pooling",
        description="LSTM-64 (increased capacity) with winning pooling strategy from Phase 2",
    ),
}


def get_experiments_for_phase(phase: int) -> list[ExperimentSpec]:
    """Retrieve list of experiment specifications for a given phase."""
    return [spec for spec in EXPERIMENT_REGISTRY.values() if spec.phase == phase]
