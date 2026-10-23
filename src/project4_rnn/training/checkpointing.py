"""Model checkpoint persistence, loading, and state management."""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger("project4_rnn.training.checkpointing")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    metrics: dict[str, Any],
    checkpoint_path: Path,
    seed: int,
    config_dict: dict[str, Any] | None = None,
) -> None:
    """Save training state checkpoint to disk.

    Args:
        model: PyTorch model.
        optimizer: PyTorch optimizer.
        epoch: Best epoch index.
        val_loss: Validation loss at checkpoint.
        metrics: Dictionary of validation metrics.
        checkpoint_path: Target path (.pt).
        seed: Execution random seed.
        config_dict: Optional configuration dictionary.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "val_loss": val_loss,
        "metrics": metrics,
        "seed": seed,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config_dict or {},
    }

    torch.save(state, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path} (epoch={epoch}, val_loss={val_loss:.6f})")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load model and optimizer states from a checkpoint file.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        model: Model instance to load weights into.
        optimizer: Optional optimizer instance to load state into.
        device: Target torch device.

    Returns:
        Checkpoint dictionary metadata.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    map_location = device if device is not None else torch.device("cpu")
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path, map_location=map_location, weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info(
        f"Loaded checkpoint from {checkpoint_path} "
        f"(saved at epoch={checkpoint.get('epoch')}, val_loss={checkpoint.get('val_loss')})"
    )
    return checkpoint
