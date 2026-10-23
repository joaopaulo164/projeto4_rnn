"""Prediction output formatting, probability serialization, and tabular export."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("project4_rnn.evaluation.predictions")


def save_prediction_artifacts(
    predictions_csv_path: Path,
    probabilities_csv_path: Path,
    sample_indices: np.ndarray,
    y_binary: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    experiment_id: str,
    phase: int,
    seed: int,
    checkpoint_name: str = "best_checkpoint.pt",
    y_original: np.ndarray | None = None,
) -> None:
    """Save detailed per-sample predictions and class probabilities with metadata.

    Args:
        predictions_csv_path: Path to write predictions CSV.
        probabilities_csv_path: Path to write probabilities CSV.
        sample_indices: Array of sample identifiers/indices.
        y_binary: Ground truth binary labels (0/1).
        y_pred: Predicted class labels (0/1).
        y_prob: Predicted positive class probability (float).
        experiment_id: Experiment identifier.
        phase: Phase number.
        seed: Execution seed.
        checkpoint_name: Name of checkpoint file used for inference.
        y_original: Optional original multiclass labels (1-5).
    """
    predictions_csv_path = Path(predictions_csv_path)
    probabilities_csv_path = Path(probabilities_csv_path)

    predictions_csv_path.parent.mkdir(parents=True, exist_ok=True)
    probabilities_csv_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = len(sample_indices)
    orig_labels = y_original if y_original is not None else y_binary

    # DataFrame for predictions
    pred_df = pd.DataFrame(
        {
            "sample_id": sample_indices,
            "original_label": orig_labels,
            "binary_label": y_binary,
            "predicted_class": y_pred,
            "positive_class_probability": y_prob,
            "phase": [phase] * n_samples,
            "experiment": [experiment_id] * n_samples,
            "seed": [seed] * n_samples,
            "checkpoint_used": [checkpoint_name] * n_samples,
        }
    )
    pred_df.to_csv(predictions_csv_path, index=False)

    # DataFrame for probabilities (prob_class_0, prob_class_1)
    prob_class_0 = 1.0 - y_prob
    prob_df = pd.DataFrame(
        {
            "sample_id": sample_indices,
            "original_label": orig_labels,
            "binary_label": y_binary,
            "prob_class_0": prob_class_0,
            "prob_class_1": y_prob,
            "predicted_class": y_pred,
            "phase": [phase] * n_samples,
            "experiment": [experiment_id] * n_samples,
            "seed": [seed] * n_samples,
        }
    )
    prob_df.to_csv(probabilities_csv_path, index=False)

    logger.info(
        f"Saved predictions ({len(pred_df)} rows) to {predictions_csv_path} and "
        f"probabilities to {probabilities_csv_path}"
    )
