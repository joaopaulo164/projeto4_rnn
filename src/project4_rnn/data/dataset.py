"""PyTorch Dataset implementation and deterministic DataLoader creation."""

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from project4_rnn.reproducibility import seed_worker

logger = logging.getLogger("project4_rnn.data.dataset")


class ECG5000Dataset(Dataset):
    """PyTorch Dataset for ECG5000 time series."""

    def __init__(
        self,
        features: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        reshape_for_rnn: bool = False,
    ) -> None:
        """Initialize ECG5000 dataset.

        Args:
            features: 2D array of shape (N, 140).
            labels: 1D array of shape (N,).
            reshape_for_rnn: If True, reshapes features to (N, 140, 1) for RNNs.
        """
        if isinstance(features, np.ndarray):
            self.features = torch.from_numpy(features).float()
        else:
            self.features = features.float()

        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = labels.long()

        if reshape_for_rnn and self.features.ndim == 2:
            self.features = self.features.unsqueeze(-1)  # (N, 140, 1)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return (features, label, sample_index)."""
        return self.features[idx], self.labels[idx], idx


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    batch_size: int = 32,
    seed: int = 42,
    reshape_for_rnn: bool = False,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Create deterministic PyTorch DataLoaders for train, validation, and optional test set.

    Args:
        X_train: Training features (400, 140).
        y_train: Training labels (400,).
        X_val: Validation features (100, 140).
        y_val: Validation labels (100,).
        X_test: Optional test features (4500, 140).
        y_test: Optional test labels (4500,).
        batch_size: Batch size for training and evaluation.
        seed: Seed for deterministic shuffle.
        reshape_for_rnn: Whether to shape input as (N, 140, 1) or (N, 140).
        num_workers: PyTorch DataLoader num_workers.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_dataset = ECG5000Dataset(X_train, y_train, reshape_for_rnn=reshape_for_rnn)
    val_dataset = ECG5000Dataset(X_val, y_val, reshape_for_rnn=reshape_for_rnn)

    # Deterministic generator for training shuffle
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        worker_init_fn=seed_worker,
        num_workers=num_workers,
        drop_last=False,
    )

    # Validation and test must NEVER shuffle
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = None
    if X_test is not None and y_test is not None:
        test_dataset = ECG5000Dataset(X_test, y_test, reshape_for_rnn=reshape_for_rnn)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

    return train_loader, val_loader, test_loader
