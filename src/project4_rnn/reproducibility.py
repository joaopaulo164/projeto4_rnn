"""Scientific reproducibility utilities, seed management, hash computation, and environment metadata."""

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for reproducible execution.

    Args:
        seed: Integer seed value.
        deterministic: If True, configures cuDNN and PyTorch deterministic algorithms.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Enable deterministic algorithms where supported
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as det_err:
            logging.getLogger("project4_rnn.reproducibility").debug(
                f"Deterministic algorithm setting not supported: {det_err}"
            )


def seed_worker(worker_id: int) -> None:
    """Worker initialization function for PyTorch DataLoader reproducibility.

    Args:
        worker_id: DataLoader worker index.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Compute the cryptographic hash of a file.

    Args:
        file_path: Path to the target file.
        algorithm: Hash algorithm name ('sha256', 'md5', etc.).

    Returns:
        Hexadecimal hash string.
    """
    hasher = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_object_hash(data: Any, algorithm: str = "sha256") -> str:
    """Compute a deterministic hash for a Python dictionary, list, or primitive.

    Args:
        data: Data structure to serialize and hash.
        algorithm: Hash algorithm name.

    Returns:
        Hexadecimal hash string.
    """
    serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.new(algorithm, serialized).hexdigest()


def get_git_info() -> dict[str, str | bool | None]:
    """Retrieve Git commit hash and repository dirty status if in a Git repository.

    Returns:
        Dictionary with 'commit_hash' and 'is_dirty'.
    """
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
        return {
            "commit_hash": commit,
            "is_dirty": len(status) > 0,
        }
    except Exception:
        return {
            "commit_hash": None,
            "is_dirty": None,
        }


def get_environment_info(device: torch.device | None = None) -> dict[str, Any]:
    """Gather complete environment, platform, and package version metadata.

    Args:
        device: PyTorch device being used.

    Returns:
        Dictionary containing hardware, OS, Python, and PyTorch metadata.
    """
    git_info = get_git_info()

    cuda_available = torch.cuda.is_available()
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else None
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "node": platform.node(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_name": cuda_device_name,
        "selected_device": str(device) if device is not None else "auto",
        "git_commit": git_info["commit_hash"],
        "git_dirty": git_info["is_dirty"],
    }
