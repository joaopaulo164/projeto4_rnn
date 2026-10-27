"""Deterministic synthetic ElectricDevices-like series used only by tests and the technical smoke test.

The generated data is NOT scientific data. It reproduces the structural properties that matter for
the split and audit logic: seven classes, length-96 series, identical series sharing a label,
identical series carrying conflicting labels, and test rows that coincide with development rows.
"""

import numpy as np

from project4_rnn.constants import NUM_CLASSES, SEQUENCE_LENGTH


def make_synthetic_series(
    n_per_class: int = 30,
    seed: int = 7,
    n_same_label_duplicate_groups: int = 4,
    n_conflicting_groups: int = 2,
    group_size: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic development set with controlled duplicate groups.

    Args:
        n_per_class: Unique base series generated per class.
        seed: Seed of the local random generator (no global state is touched).
        n_same_label_duplicate_groups: Groups of identical series that share one label.
        n_conflicting_groups: Groups of identical series whose members carry different labels.
        group_size: Number of members of each controlled duplicate group.

    Returns:
        Tuple (X, y): float64 array (N, 96) and official labels (1..7), shuffled deterministically.
    """
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[int] = []

    for cls in range(1, NUM_CLASSES + 1):
        centre = rng.normal(0.0, 1.0, size=SEQUENCE_LENGTH) + cls
        for _ in range(n_per_class):
            rows.append(centre + rng.normal(0.0, 0.5, size=SEQUENCE_LENGTH))
            labels.append(cls)

    for g in range(n_same_label_duplicate_groups):
        cls = (g % NUM_CLASSES) + 1
        base = rng.normal(0.0, 1.0, size=SEQUENCE_LENGTH) + cls
        for _ in range(group_size):
            rows.append(base.copy())
            labels.append(cls)

    for g in range(n_conflicting_groups):
        base = rng.normal(0.0, 1.0, size=SEQUENCE_LENGTH) + 10.0 + g
        for member in range(group_size):
            rows.append(base.copy())
            labels.append(((g + member) % NUM_CLASSES) + 1)

    X = np.vstack(rows).astype(np.float64)
    y = np.asarray(labels, dtype=np.int64)
    order = rng.permutation(len(y))
    return X[order], y[order]


def make_synthetic_test_series(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    n_per_class: int = 12,
    seed: int = 11,
    n_overlap: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic test set that partially coincides with the development set.

    Args:
        X_dev: Development series (N, 96).
        y_dev: Development official labels (1..7).
        n_per_class: Fresh series generated per class.
        seed: Seed of the local random generator.
        n_overlap: Development rows copied verbatim into the test set.

    Returns:
        Tuple (X_test, y_test) with official labels 1..7.
    """
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[int] = []

    for cls in range(1, NUM_CLASSES + 1):
        centre = rng.normal(0.0, 1.0, size=SEQUENCE_LENGTH) + cls
        for _ in range(n_per_class):
            rows.append(centre + rng.normal(0.0, 0.5, size=SEQUENCE_LENGTH))
            labels.append(cls)

    chosen = rng.choice(len(X_dev), size=min(n_overlap, len(X_dev)), replace=False)
    for idx in chosen:
        rows.append(np.asarray(X_dev[int(idx)], dtype=np.float64).copy())
        labels.append(int(y_dev[int(idx)]))

    X = np.vstack(rows).astype(np.float64)
    y = np.asarray(labels, dtype=np.int64)
    order = rng.permutation(len(y))
    return X[order], y[order]
