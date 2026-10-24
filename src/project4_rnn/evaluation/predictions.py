"""Prediction output formatting, probability serialization, and tabular export for ElectricDevices."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from project4_rnn.constants import (
    INTERNAL_TO_OFFICIAL_LABEL,
    NUM_CLASSES,
    OFFICIAL_TO_INTERNAL_LABEL,
)

logger = logging.getLogger("project4_rnn.evaluation.predictions")


def save_prediction_artifacts(
    predictions_csv_path: Path,
    probabilities_csv_path: Path,
    sample_indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    experiment_id: str,
    phase: int,
    seed: int,
    checkpoint_name: str = "best_checkpoint.pt",
) -> None:
    """Save detailed per-sample multiclass predictions and class probabilities with full metadata.

    Args:
        predictions_csv_path: Path to write predictions CSV.
        probabilities_csv_path: Path to write probabilities CSV.
        sample_indices: Array of sample identifiers/indices.
        y_true: Ground truth labels (internal indices 0..6 or official labels 1..7).
        y_pred: Predicted labels (internal indices 0..6 or official labels 1..7).
        y_prob: Predicted probability matrix of shape (N, 7).
        experiment_id: Experiment identifier.
        phase: Phase number.
        seed: Execution seed.
        checkpoint_name: Name of checkpoint file used for inference.
    """
    predictions_csv_path = Path(predictions_csv_path)
    probabilities_csv_path = Path(probabilities_csv_path)

    predictions_csv_path.parent.mkdir(parents=True, exist_ok=True)
    probabilities_csv_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = len(sample_indices)
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)

    # Determine whether y_true / y_pred are 0-indexed internal or 1-indexed official
    if set(np.unique(y_true_arr)).issubset(set(range(NUM_CLASSES))):
        true_internal = y_true_arr
        true_official = np.array(
            [INTERNAL_TO_OFFICIAL_LABEL.get(int(v), int(v) + 1) for v in y_true_arr]
        )
    else:
        true_official = y_true_arr
        true_internal = np.array(
            [OFFICIAL_TO_INTERNAL_LABEL.get(int(v), int(v) - 1) for v in y_true_arr]
        )

    if set(np.unique(y_pred_arr)).issubset(set(range(NUM_CLASSES))):
        pred_internal = y_pred_arr
        pred_official = np.array(
            [INTERNAL_TO_OFFICIAL_LABEL.get(int(v), int(v) + 1) for v in y_pred_arr]
        )
    else:
        pred_official = y_pred_arr
        pred_internal = np.array(
            [OFFICIAL_TO_INTERNAL_LABEL.get(int(v), int(v) - 1) for v in y_pred_arr]
        )

    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    if y_prob_arr.ndim == 1:
        # If single column passed by mistake, convert to dummy 2D
        y_prob_arr = np.repeat(y_prob_arr[:, np.newaxis], NUM_CLASSES, axis=1) / NUM_CLASSES

    # Prediction dataframe dictionary
    pred_dict: dict[str, Any] = {
        "sample_id": sample_indices,
        "true_official_label": true_official,
        "true_internal_index": true_internal,
        "predicted_official_label": pred_official,
        "predicted_internal_index": pred_internal,
    }

    # Add per-class probability columns
    for class_idx in range(NUM_CLASSES):
        official_class_label = INTERNAL_TO_OFFICIAL_LABEL.get(class_idx, class_idx + 1)
        prob_col_name = f"prob_class_{official_class_label}"
        pred_dict[prob_col_name] = y_prob_arr[:, class_idx]

    pred_dict["experiment"] = [experiment_id] * n_samples
    pred_dict["phase"] = [phase] * n_samples
    pred_dict["seed"] = [seed] * n_samples
    pred_dict["checkpoint_used"] = [checkpoint_name] * n_samples

    pred_df = pd.DataFrame(pred_dict)
    pred_df.to_csv(predictions_csv_path, index=False)

    # Probabilities dataframe
    prob_dict: dict[str, Any] = {
        "sample_id": sample_indices,
        "true_official_label": true_official,
        "true_internal_index": true_internal,
        "predicted_official_label": pred_official,
        "predicted_internal_index": pred_internal,
    }
    for class_idx in range(NUM_CLASSES):
        official_class_label = INTERNAL_TO_OFFICIAL_LABEL.get(class_idx, class_idx + 1)
        prob_dict[f"prob_class_{official_class_label}"] = y_prob_arr[:, class_idx]

    prob_dict["experiment"] = [experiment_id] * n_samples
    prob_dict["phase"] = [phase] * n_samples
    prob_dict["seed"] = [seed] * n_samples

    prob_df = pd.DataFrame(prob_dict)
    prob_df.to_csv(probabilities_csv_path, index=False)

    logger.info(
        f"Saved predictions ({len(pred_df)} rows) to {predictions_csv_path} and "
        f"probabilities to {probabilities_csv_path}"
    )
