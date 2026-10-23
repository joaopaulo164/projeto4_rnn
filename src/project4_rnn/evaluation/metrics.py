"""Comprehensive classification metrics computation and evaluation."""

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("project4_rnn.evaluation.metrics")


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    loss: float | None = None,
) -> dict[str, Any]:
    """Compute all scientific classification metrics for binary classification.

    Args:
        y_true: True binary labels (0 and 1).
        y_pred: Predicted binary labels (0 and 1).
        y_prob: Predicted probabilities for positive class (class 1), shape (N,).
        loss: Optional scalar cross-entropy loss value.

    Returns:
        Dictionary containing all evaluation metrics.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Confusion matrix and specificity
    # For binary 0 and 1: tn, fp, fn, tp
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        specificity = 0.0

    # ROC-AUC and PR-AUC (require probability of positive class)
    roc_auc: float | None = None
    pr_auc: float | None = None

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=np.float64)
        # Check if both classes are present in y_true
        if len(np.unique(y_true)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_true, y_prob))
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")
                roc_auc = None

            try:
                pr_auc = float(average_precision_score(y_true, y_prob))
            except Exception as e:
                logger.warning(f"Could not compute PR-AUC: {e}")
                pr_auc = None

    metrics: dict[str, Any] = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "total_samples": int(len(y_true)),
    }

    if loss is not None:
        metrics["loss"] = float(loss)

    return metrics
