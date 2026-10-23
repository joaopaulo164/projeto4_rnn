"""Model trainer, optimization loop, and validation workflow."""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from project4_rnn.config import ExperimentConfig
from project4_rnn.evaluation.metrics import compute_classification_metrics
from project4_rnn.models.base import BaseTimeSeriesModel
from project4_rnn.training.checkpointing import load_checkpoint, save_checkpoint
from project4_rnn.training.early_stopping import EarlyStopping
from project4_rnn.training.history import TrainingHistory

logger = logging.getLogger("project4_rnn.training.trainer")


class Trainer:
    """Trainer executing optimization, early stopping, and checkpoint management."""

    def __init__(
        self,
        model: BaseTimeSeriesModel,
        config: ExperimentConfig,
        seed: int,
        checkpoint_dir: Path | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: PyTorch model instance.
            config: Resolved ExperimentConfig.
            seed: Training execution seed.
            checkpoint_dir: Directory where checkpoints and history are saved.
            device: Target torch device (CPU or CUDA).
        """
        self.model = model
        self.config = config
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir

        if device is not None:
            self.device = device
        else:
            if config.system.device == "cuda" and torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif config.system.device == "auto" and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

        self.model.to(self.device)

        # Loss function (CrossEntropyLoss with optional balanced class weighting)
        weight_tensor: torch.Tensor | None = None
        if config.training.use_class_weights:
            # Inverse frequency balanced weights based on official split: ~291 class 0 and ~109 class 1
            weight_tensor = torch.tensor(
                [500.0 / (2.0 * 291.0), 500.0 / (2.0 * 109.0)], device=self.device
            )
            logger.info(
                f"Initialized weighted CrossEntropyLoss with weights: {weight_tensor.tolist()}"
            )
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        # Optimizer
        opt_name = config.training.optimizer.lower()
        if opt_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config.training.optimizer}")

        # Early stopping callback
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            min_delta=config.training.min_delta,
            mode=config.training.checkpoint_mode,
        )

        self.history = TrainingHistory()
        self.best_checkpoint_path: Path | None = None

    def train_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        """Execute one training epoch.

        Args:
            train_loader: Training DataLoader.

        Returns:
            Tuple of (epoch_loss, epoch_accuracy).
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for x_batch, y_batch, _ in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x_batch)
            loss = self.criterion(logits, y_batch)

            loss.backward()

            if self.config.training.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.training.max_grad_norm,
                )

            self.optimizer.step()

            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total_samples += batch_size

        epoch_loss = total_loss / total_samples if total_samples > 0 else 0.0
        epoch_acc = correct / total_samples if total_samples > 0 else 0.0
        return epoch_loss, epoch_acc

    def evaluate(
        self,
        loader: DataLoader,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, float]:
        """Evaluate model performance on a DataLoader without gradient updates.

        Args:
            loader: Validation or Test DataLoader.

        Returns:
            Tuple of (metrics_dict, predictions_array, probabilities_array, inference_duration_sec).
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        all_preds: list[int] = []
        all_targets: list[int] = []
        all_probs: list[float] = []

        start_time = time.perf_counter()

        with torch.no_grad():
            for x_batch, y_batch, _ in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(x_batch)
                loss = self.criterion(logits, y_batch)

                batch_size = x_batch.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                all_preds.extend(preds.tolist())
                all_targets.extend(y_batch.cpu().numpy().tolist())
                all_probs.extend(probs.tolist())

        inference_time = time.perf_counter() - start_time
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        y_true = np.array(all_targets, dtype=np.int64)
        y_pred = np.array(all_preds, dtype=np.int64)
        y_prob = np.array(all_probs, dtype=np.float64)

        metrics = compute_classification_metrics(y_true, y_pred, y_prob, loss=avg_loss)
        return metrics, y_pred, y_prob, inference_time

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, Any]:
        """Run full training loop with validation, early stopping, and checkpoint saving.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.

        Returns:
            Dictionary summarizing training run metrics, parameters, and timings.
        """
        logger.info(
            f"Starting training: Experiment={self.config.experiment_id}, "
            f"Seed={self.seed}, Device={self.device}, MaxEpochs={self.config.training.max_epochs}"
        )

        total_start_time = time.perf_counter()

        if self.checkpoint_dir is not None:
            self.best_checkpoint_path = self.checkpoint_dir / "best_checkpoint.pt"

        for epoch in range(1, self.config.training.max_epochs + 1):
            epoch_start = time.perf_counter()

            train_loss, train_acc = self.train_epoch(train_loader)
            val_metrics, _, _, _ = self.evaluate(val_loader)
            epoch_duration = time.perf_counter() - epoch_start

            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]
            val_bal_acc = val_metrics["balanced_accuracy"]
            val_f1 = val_metrics["f1"]

            self.history.add_epoch(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                val_balanced_accuracy=val_bal_acc,
                val_f1=val_f1,
                epoch_time_sec=epoch_duration,
            )

            # Check early stopping / best score
            is_best = self.early_stopping.step(val_loss, epoch)

            if is_best:
                if self.best_checkpoint_path is not None:
                    save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        val_loss=val_loss,
                        metrics=val_metrics,
                        checkpoint_path=self.best_checkpoint_path,
                        seed=self.seed,
                        config_dict=self.config.to_dict(),
                    )

            if epoch % 10 == 0 or is_best or self.early_stopping.early_stop:
                logger.info(
                    f"Epoch {epoch:03d}/{self.config.training.max_epochs} | "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Val BalAcc: {val_bal_acc:.4f}, Val F1: {val_f1:.4f} "
                    f"({'BEST' if is_best else ''})"
                )

            if self.early_stopping.early_stop:
                logger.info(f"Stopping early at epoch {epoch}")
                break

        total_duration = time.perf_counter() - total_start_time

        # Save training history CSV
        if self.checkpoint_dir is not None:
            self.history.save_csv(self.checkpoint_dir / "training_history.csv")

        # Reload best checkpoint weights for subsequent evaluation
        if self.best_checkpoint_path is not None and self.best_checkpoint_path.is_file():
            load_checkpoint(self.best_checkpoint_path, self.model, device=self.device)

        # Final evaluation of the restored best model on validation set
        val_eval_metrics, val_preds, val_probs, inf_time = self.evaluate(val_loader)

        total_epochs = len(self.history.records)
        mean_epoch_time = total_duration / total_epochs if total_epochs > 0 else 0.0

        run_summary = {
            "experiment_id": self.config.experiment_id,
            "phase": self.config.phase,
            "seed": self.seed,
            "best_epoch": self.early_stopping.best_epoch,
            "total_epochs": total_epochs,
            "total_training_time_sec": total_duration,
            "mean_epoch_time_sec": mean_epoch_time,
            "inference_time_sec": inf_time,
            "total_parameters": self.model.count_parameters(),
            "trainable_parameters": self.model.count_trainable_parameters(),
            "best_val_loss": self.early_stopping.best_score,
            "validation_metrics": val_eval_metrics,
        }

        return run_summary
