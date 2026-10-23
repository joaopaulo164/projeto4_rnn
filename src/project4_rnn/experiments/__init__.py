"""Experiment runner, registry, selection criteria, and result consolidation."""

from project4_rnn.experiments.consolidation import consolidate_experiments
from project4_rnn.experiments.model_selection import select_final_model
from project4_rnn.experiments.pooling_selection import (
    load_selected_pooling,
    select_best_pooling,
)
from project4_rnn.experiments.registry import (
    EXPERIMENT_REGISTRY,
    ExperimentSpec,
    get_experiments_for_phase,
)
from project4_rnn.experiments.runner import (
    check_run_status,
    run_experiment_all_seeds,
    run_single_seed,
)

__all__ = [
    "EXPERIMENT_REGISTRY",
    "ExperimentSpec",
    "get_experiments_for_phase",
    "run_single_seed",
    "run_experiment_all_seeds",
    "check_run_status",
    "select_best_pooling",
    "load_selected_pooling",
    "select_final_model",
    "consolidate_experiments",
]
