"""Unit tests for the deterministic series fingerprint."""

import hashlib

import numpy as np
import pytest

from project4_rnn.data.hashing import hash_series, hash_series_matrix


def _series(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(96)


def test_hash_is_deterministic_and_sha256_hex() -> None:
    """The same series always yields the same 64-character SHA-256 digest."""
    series = _series()
    assert hash_series(series) == hash_series(series.copy())
    assert len(hash_series(series)) == 64
    int(hash_series(series), 16)


def test_hash_changes_when_the_series_changes() -> None:
    """A single changed value changes the fingerprint, even by one ulp (no rounding)."""
    series = _series()
    changed = series.copy()
    changed[10] = np.nextafter(changed[10], np.inf)
    assert hash_series(series) != hash_series(changed)


def test_hash_matches_little_endian_float64_bytes() -> None:
    """The digest is the SHA-256 of the contiguous little-endian float64 bytes."""
    series = _series(3)
    expected = hashlib.sha256(series.astype("<f8").tobytes()).hexdigest()
    assert hash_series(series) == expected


def test_hash_is_independent_of_memory_layout_and_byte_order() -> None:
    """Non-contiguous views and big-endian arrays hash like their canonical form."""
    series = _series(4)
    strided = np.stack([series, series], axis=1)[:, 0]
    assert not strided.flags["C_CONTIGUOUS"]
    assert hash_series(strided) == hash_series(series)
    assert hash_series(series.astype(">f8")) == hash_series(series)


def test_hash_converts_float32_input_to_float64_first() -> None:
    """float32 input is widened to float64, so it differs from float64 data it was cast from."""
    series32 = _series(5).astype(np.float32)
    assert hash_series(series32) == hash_series(series32.astype(np.float64))


def test_hash_does_not_depend_on_label_or_index() -> None:
    """The matrix hash of a row is the same wherever the row sits in the matrix."""
    a, b = _series(1), _series(2)
    first = hash_series_matrix(np.vstack([a, b]))
    second = hash_series_matrix(np.vstack([b, a]))
    assert first == second[::-1]


@pytest.mark.parametrize("length", [0, 95, 97])
def test_hash_requires_exactly_96_values(length: int) -> None:
    """Series of the wrong length are rejected."""
    with pytest.raises(ValueError, match="exactly 96"):
        hash_series(np.zeros(length))


def test_hash_rejects_non_finite_and_wrong_dimension() -> None:
    """NaN/inf values and non-1D input are rejected."""
    bad = _series()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        hash_series(bad)
    with pytest.raises(ValueError, match="one-dimensional"):
        hash_series(np.zeros((2, 96)))
    with pytest.raises(ValueError, match="2D matrix"):
        hash_series_matrix(np.zeros(96))
