"""Deterministic fingerprints of raw time series, independent of label and sample index."""

import hashlib

import numpy as np

from project4_rnn.constants import SEQUENCE_LENGTH


def hash_series(values: np.ndarray) -> str:
    """Compute the SHA-256 fingerprint of a single raw series.

    The fingerprint is computed over exactly 96 values converted to contiguous little-endian
    float64, without rounding. Neither the label nor the sample index take part in the hash.

    Args:
        values: One-dimensional array with the raw (pre-normalization) series values.

    Returns:
        Hexadecimal SHA-256 digest.

    Raises:
        ValueError: If the series does not have exactly 96 values or contains non-finite values.
    """
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.shape[0] != SEQUENCE_LENGTH:
        raise ValueError(
            f"Series must be one-dimensional with exactly {SEQUENCE_LENGTH} values, "
            f"got shape {arr.shape}"
        )
    canonical = np.ascontiguousarray(arr, dtype="<f8")
    if not np.all(np.isfinite(canonical)):
        raise ValueError("Series contains NaN or infinite values and cannot be fingerprinted")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def hash_series_matrix(X: np.ndarray) -> list[str]:
    """Compute the fingerprint of every row of a (N, 96) matrix."""
    matrix = np.asarray(X)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix of series, got ndim={matrix.ndim}")
    return [hash_series(row) for row in matrix]
