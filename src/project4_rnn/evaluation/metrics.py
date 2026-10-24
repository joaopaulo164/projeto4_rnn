"""Comprehensive multiclass classification metrics computation and evaluation for ElectricDevices."""

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from project4_rnn.constants import INTERNAL_TO_OFFICIAL_LABEL, NUM_CLASSES

logger = logging.getLogger("project4_rnn.evaluation.metrics")


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    loss: float | None = None,
    num_classes: int = NUM_CLASSES,
) -> dict[str, Any]:
    """Compute all multiclass scientific classification metrics.

    Args:
        y_true: True internal index labels (0 to 6) or official labels.
        y_pred: Predicted internal index labels (0 to 6) or official labels.
        y_prob: Predicted class probabilities matrix of shape (N, num_classes).
        loss: Optional scalar cross-entropy loss value.
        num_classes: Total number of classes (default 7).

    Returns:
        Dictionary containing all multiclass evaluation metrics, per-class metrics, and confusion matrices.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    labels_list = list(range(num_classes))
    # If inputs are already 1-indexed official labels, adjust labels_list
    if set(np.unique(y_true)).issubset(set(range(1, num_classes + 1))):
        labels_list = list(range(1, num_classes + 1))

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    macro_f1 = float(f1_score(y_true, y_pred, labels=labels_list, average="macro", zero_division=0))
    weighted_f1 = float(
        f1_score(y_true, y_pred, labels=labels_list, average="weighted", zero_division=0)
    )
    macro_prec = float(
        precision_score(y_true, y_pred, labels=labels_list, average="macro", zero_division=0)
    )
    macro_rec = float(
        recall_score(y_true, y_pred, labels=labels_list, average="macro", zero_division=0)
    )

    # Per-class metrics
    class_precisions = precision_score(
        y_true, y_pred, labels=labels_list, average=None, zero_division=0
    )
    class_recalls = recall_score(y_true, y_pred, labels=labels_list, average=None, zero_division=0)
    class_f1s = f1_score(y_true, y_pred, labels=labels_list, average=None, zero_division=0)

    # Confusion matrix (raw and normalized)
    cm_raw = confusion_matrix(y_true, y_pred, labels=labels_list)
    cm_sum = cm_raw.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(cm_sum > 0, cm_raw / cm_sum, 0.0)

    per_class_metrics: dict[str, dict[str, Any]] = {}
    for idx, class_label in enumerate(labels_list):
        official_name = (
            f"Class_{INTERNAL_TO_OFFICIAL_LABEL.get(class_label, class_label)}"
            if class_label in INTERNAL_TO_OFFICIAL_LABEL
            else f"Class_{class_label}"
        )
        support = int(np.sum(y_true == class_label))
        per_class_metrics[official_name] = {
            "internal_index": int(class_label if class_label < num_classes else class_label - 1),
            "official_label": int(INTERNAL_TO_OFFICIAL_LABEL.get(class_label, class_label)),
            "precision": float(class_precisions[idx]),
            "recall": float(class_recalls[idx]),
            "f1": float(class_f1s[idx]),
            "support": support,
        }

    # Optional multiclass ROC-AUC (One-vs-Rest)
    roc_auc_ovr: float | None = None
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=np.float64)
        if y_prob.ndim == 2 and y_prob.shape[1] == num_classes and len(np.unique(y_true)) > 1:
            try:
                roc_auc_ovr = float(
                    roc_auc_score(
                        y_true,
                        y_prob,
                        labels=labels_list,
                        multi_class="ovr",
                        average="macro",
                    )
                )
            except Exception as e:
                logger.debug(f"Multiclass ROC-AUC computation skipped: {e}")
                roc_auc_ovr = None

    metrics: dict[str, Any] = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "f1": macro_f1,  # Primary F1 alias
        "macro_precision": macro_prec,
        "precision": macro_prec,
        "macro_recall": macro_rec,
        "recall": macro_rec,
        "roc_auc": roc_auc_ovr,
        "confusion_matrix_raw": cm_raw.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "per_class_metrics": per_class_metrics,
        "total_samples": int(len(y_true)),
    }

    if loss is not None:
        metrics["loss"] = float(loss)

    return metrics
