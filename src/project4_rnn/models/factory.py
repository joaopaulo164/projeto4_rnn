"""Model factory and parameter verification."""

import logging

from project4_rnn.config import ExperimentConfig, ModelConfig
from project4_rnn.constants import EXPECTED_PARAMETERS
from project4_rnn.models.base import BaseTimeSeriesModel
from project4_rnn.models.mlp import MLP32
from project4_rnn.models.recurrent import RecurrentClassifier

logger = logging.getLogger("project4_rnn.models.factory")


def build_model(
    config: ExperimentConfig | ModelConfig,
    validate_params: bool = True,
) -> BaseTimeSeriesModel:
    """Instantiate a neural network model from configuration and verify parameters.

    Args:
        config: ExperimentConfig or ModelConfig object.
        validate_params: If True, asserts exact match with theoretical parameter counts.

    Returns:
        Instantiated BaseTimeSeriesModel.

    Raises:
        ValueError: If architecture is invalid or parameter count does not match expected.
    """
    model_cfg = config.model if isinstance(config, ExperimentConfig) else config
    experiment_id = config.experiment_id if isinstance(config, ExperimentConfig) else None

    arch = model_cfg.architecture.lower()

    if arch == "mlp":
        model: BaseTimeSeriesModel = MLP32(
            input_size=model_cfg.input_size,
            hidden_size=model_cfg.hidden_size,
            num_classes=model_cfg.num_classes,
        )
    elif arch in ("gru", "lstm"):
        if model_cfg.pooling is None:
            raise ValueError(f"Pooling strategy must be specified for {arch}")
        if model_cfg.pooling == "best_pooling":
            raise ValueError(
                "Cannot build model with unresolved 'best_pooling'. "
                "Resolve pooling strategy before instantiation."
            )
        model = RecurrentClassifier(
            cell_type=arch,
            input_size=model_cfg.input_size,
            hidden_size=model_cfg.hidden_size,
            num_layers=model_cfg.num_layers,
            bidirectional=model_cfg.bidirectional,
            batch_first=model_cfg.batch_first,
            pooling=model_cfg.pooling,
            num_classes=model_cfg.num_classes,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    actual_params = model.count_parameters()
    logger.info(
        f"Built model '{arch}' with {actual_params:,} parameters "
        f"(Trainable: {model.count_trainable_parameters():,})"
    )

    if validate_params and experiment_id and experiment_id in EXPECTED_PARAMETERS:
        expected = EXPECTED_PARAMETERS[experiment_id]
        if isinstance(expected, dict):
            expected_count: int | None = expected.get(model_cfg.pooling or "")
        else:
            expected_count = expected

        if expected_count is not None and actual_params != expected_count:
            raise ValueError(
                f"Parameter count mismatch for {experiment_id}: "
                f"expected {expected_count}, got {actual_params}!"
            )

    return model
