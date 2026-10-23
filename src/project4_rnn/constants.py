"""Global scientific constants, dataset specifications, and experiment definitions."""

from typing import Final

# Dataset specifications
DATASET_NAME: Final[str] = "ECG5000"
SEQUENCE_LENGTH: Final[int] = 140
NUM_FEATURES: Final[int] = 1
NUM_CLASSES: Final[int] = 2

# Official split sizes
OFFICIAL_DEV_SAMPLES: Final[int] = 500
OFFICIAL_TEST_SAMPLES: Final[int] = 4500
TOTAL_SAMPLES: Final[int] = 5000

# Development split specification
SPLIT_SEED: Final[int] = 42
TRAIN_SAMPLES: Final[int] = 400
VAL_SAMPLES: Final[int] = 100

# Training seeds
TRAIN_SEEDS: Final[list[int]] = [7, 21, 42, 84, 168]

# Binary label adaptation
# Original 1 -> Binary 0 (Negative class)
# Original 2, 3, 4, 5 -> Binary 1 (Positive class)
ORIGINAL_NEGATIVE_CLASSES: Final[tuple[int, ...]] = (1,)
ORIGINAL_POSITIVE_CLASSES: Final[tuple[int, ...]] = (2, 3, 4, 5)
LABEL_NEGATIVE: Final[int] = 0
LABEL_POSITIVE: Final[int] = 1

# Parsimony threshold for pooling selection (0.5 percentage points = 0.005)
PARSIMONY_THRESHOLD: Final[float] = 0.005

# Expected parameter counts for architectures
EXPECTED_PARAMETERS: Final[dict[str, int | dict[str, int]]] = {
    "E01_MLP_32": 4578,
    "E02_GRU_32_LAST": 3426,
    "E03_GRU_32_MEAN": 3426,
    "E04_GRU_32_MAX": 3426,
    "E05_GRU_32_MEAN_MAX": 3490,
    "E06_LSTM_32_BEST_POOLING": {
        "last": 4546,
        "mean": 4546,
        "max": 4546,
        "mean_max": 4610,
    },
    "E07_GRU_64_BEST_POOLING": {
        "last": 12994,
        "mean": 12994,
        "max": 12994,
        "mean_max": 13122,
    },
    "E08_LSTM_64_BEST_POOLING": {
        "last": 17282,
        "mean": 17282,
        "max": 17282,
        "mean_max": 17410,
    },
}

# Experiment phases mapping
EXPERIMENT_PHASES: Final[dict[str, int]] = {
    "E01_MLP_32": 1,
    "E02_GRU_32_LAST": 2,
    "E03_GRU_32_MEAN": 2,
    "E04_GRU_32_MAX": 2,
    "E05_GRU_32_MEAN_MAX": 2,
    "E06_LSTM_32_BEST_POOLING": 3,
    "E07_GRU_64_BEST_POOLING": 4,
    "E08_LSTM_64_BEST_POOLING": 4,
}

ALL_EXPERIMENT_IDS: Final[list[str]] = list(EXPERIMENT_PHASES.keys())
PHASE_2_CANDIDATES: Final[list[str]] = [
    "E02_GRU_32_LAST",
    "E03_GRU_32_MEAN",
    "E04_GRU_32_MAX",
    "E05_GRU_32_MEAN_MAX",
]
