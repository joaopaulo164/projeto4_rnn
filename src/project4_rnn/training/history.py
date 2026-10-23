"""Training history tracking and tabular serialization."""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger("project4_rnn.training.history")


@dataclass
class EpochRecord:
    """Metrics and duration recorded for a single training epoch."""

    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_balanced_accuracy: float
    val_f1: float
    epoch_time_sec: float


class TrainingHistory:
    """Collect and persist epoch-by-epoch training and validation progress."""

    def __init__(self) -> None:
        self.records: list[EpochRecord] = []

    def add_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        val_balanced_accuracy: float,
        val_f1: float,
        epoch_time_sec: float,
    ) -> None:
        """Add record for an epoch."""
        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            val_balanced_accuracy=val_balanced_accuracy,
            val_f1=val_f1,
            epoch_time_sec=epoch_time_sec,
        )
        self.records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert recorded history to pandas DataFrame."""
        if not self.records:
            return pd.DataFrame(
                columns=[
                    "epoch",
                    "train_loss",
                    "train_acc",
                    "val_loss",
                    "val_acc",
                    "val_balanced_accuracy",
                    "val_f1",
                    "epoch_time_sec",
                ]
            )
        return pd.DataFrame([asdict(r) for r in self.records])

    def save_csv(self, path: Path) -> None:
        """Save history records to CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        logger.info(f"Saved training history ({len(df)} epochs) to {path}")
