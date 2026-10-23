"""Unit tests for metrics calculation, early stopping, checkpointing, and training history."""

from pathlib import Path

import numpy as np
import torch

from project4_rnn.evaluation.metrics import compute_classification_metrics
from project4_rnn.models.mlp import MLP32
from project4_rnn.training.checkpointing import load_checkpoint, save_checkpoint
from project4_rnn.training.early_stopping import EarlyStopping
from project4_rnn.training.history import TrainingHistory


def test_classification_metrics_computation() -> None:
    """Verify standard binary classification metrics including balanced accuracy and specificity."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.int64)
    y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.7, 0.85, 0.4], dtype=np.float64)

    metrics = compute_classification_metrics(y_true, y_pred, y_prob, loss=0.35)

    assert metrics["accuracy"] == 0.75
    # Class 0 recall: 3/4 = 0.75, Class 1 recall: 3/4 = 0.75 -> bal acc = 0.75
    assert metrics["balanced_accuracy"] == 0.75
    assert metrics["specificity"] == 0.75
    assert metrics["true_positives"] == 3
    assert metrics["true_negatives"] == 3
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["roc_auc"] is not None
    assert metrics["loss"] == 0.35


def test_early_stopping_behavior() -> None:
    """Verify EarlyStopping patience triggers and records best epoch."""
    es = EarlyStopping(patience=3, min_delta=0.01, mode="min")

    # Epoch 1: score 1.0 (best)
    is_best_1 = es.step(1.0, epoch=1)
    assert is_best_1 is True
    assert es.best_epoch == 1
    assert es.early_stop is False

    # Epoch 2: score 0.90 (improvement > 0.01)
    is_best_2 = es.step(0.90, epoch=2)
    assert is_best_2 is True
    assert es.best_epoch == 2
    assert es.early_stop is False

    # Epoch 3: score 0.90 (no improvement, counter=1)
    assert es.step(0.90, epoch=3) is False
    assert es.counter == 1
    assert es.early_stop is False

    # Epoch 4: score 0.91 (counter=2)
    assert es.step(0.91, epoch=4) is False
    assert es.counter == 2
    assert es.early_stop is False

    # Epoch 5: score 0.92 (counter=3 >= patience -> early_stop=True)
    assert es.step(0.92, epoch=5) is False
    assert es.early_stop is True
    assert es.best_epoch == 2


def test_checkpoint_save_and_load(tmp_path: Path) -> None:
    """Verify saving and restoring model and optimizer states."""
    model = MLP32()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    ckpt_path = tmp_path / "test_ckpt.pt"

    # Modify a weight slightly
    with torch.no_grad():
        model.fc1.weight.fill_(0.5)

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=5,
        val_loss=0.25,
        metrics={"balanced_accuracy": 0.92},
        checkpoint_path=ckpt_path,
        seed=42,
    )

    assert ckpt_path.is_file()

    # Create fresh model and load
    new_model = MLP32()
    new_opt = torch.optim.Adam(new_model.parameters(), lr=0.01)
    meta = load_checkpoint(ckpt_path, new_model, new_opt)

    assert meta["epoch"] == 5
    assert meta["val_loss"] == 0.25
    assert torch.allclose(new_model.fc1.weight, torch.tensor(0.5))


def test_training_history_tracking(tmp_path: Path) -> None:
    """Verify history accumulation and CSV export."""
    history = TrainingHistory()
    history.add_epoch(1, 0.5, 0.8, 0.4, 0.85, 0.84, 0.83, 1.2)
    history.add_epoch(2, 0.3, 0.9, 0.25, 0.92, 0.91, 0.90, 1.1)

    df = history.to_dataframe()
    assert len(df) == 2
    assert list(df["epoch"]) == [1, 2]

    csv_path = tmp_path / "history.csv"
    history.save_csv(csv_path)
    assert csv_path.is_file()
