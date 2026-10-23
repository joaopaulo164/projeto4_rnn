"""Early stopping callback for training regularization."""

import logging

logger = logging.getLogger("project4_rnn.training.early_stopping")


class EarlyStopping:
    """Monitor a validation metric and signal when training should stop."""

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.0001,
        mode: str = "min",
    ) -> None:
        """Initialize EarlyStopping.

        Args:
            patience: Number of epochs to wait without improvement before stopping.
            min_delta: Minimum change in monitored metric to qualify as an improvement.
            mode: 'min' (e.g. for loss) or 'max' (e.g. for accuracy/f1).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()

        if self.mode not in ("min", "max"):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'min' or 'max'.")

        self.counter: int = 0
        self.best_score: float | None = None
        self.best_epoch: int = 0
        self.early_stop: bool = False
        self.is_best: bool = False

    def step(self, current_metric: float, epoch: int) -> bool:
        """Update tracker with the current metric value.

        Args:
            current_metric: Value of the monitored metric in current epoch.
            epoch: Current epoch number (1-based).

        Returns:
            True if current epoch achieved the best score so far, False otherwise.
        """
        if self.best_score is None:
            self.best_score = current_metric
            self.best_epoch = epoch
            self.counter = 0
            self.is_best = True
            return True

        if self.mode == "min":
            improvement = (self.best_score - current_metric) >= self.min_delta
        else:
            improvement = (current_metric - self.best_score) >= self.min_delta

        if improvement:
            self.best_score = current_metric
            self.best_epoch = epoch
            self.counter = 0
            self.is_best = True
        else:
            self.counter += 1
            self.is_best = False
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best score {self.best_score:.6f} at epoch {self.best_epoch}."
                )

        return self.is_best
