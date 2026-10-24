"""Evaluation metrics, prediction artifacts, statistical consolidation, and plots for ElectricDevices."""

from project4_rnn.evaluation.metrics import compute_classification_metrics
from project4_rnn.evaluation.plots import (
    plot_cell_and_capacity_comparisons,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_pooling_ablation,
    plot_seed_stability,
    plot_time_series_samples,
    plot_training_curves,
)
from project4_rnn.evaluation.predictions import save_prediction_artifacts
from project4_rnn.evaluation.statistics import (
    create_summary_dataframe,
    summarize_seed_runs,
)

__all__ = [
    "compute_classification_metrics",
    "save_prediction_artifacts",
    "summarize_seed_runs",
    "create_summary_dataframe",
    "plot_training_curves",
    "plot_confusion_matrix",
    "plot_class_distribution",
    "plot_time_series_samples",
    "plot_pooling_ablation",
    "plot_seed_stability",
    "plot_cell_and_capacity_comparisons",
]
