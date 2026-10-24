"""Global scientific constants, dataset specifications, and experiment definitions."""

from typing import Final

# Dataset specifications for ElectricDevices benchmark
DATASET_NAME: Final[str] = "ElectricDevices"
DATASET_VARIANT: Final[str] = "UCR_UEA_benchmark"
DATASET_SOURCE: Final[str] = "UCR_UEA_Time_Series_Classification_Archive"
SEQUENCE_LENGTH: Final[int] = 96
NUM_FEATURES: Final[int] = 1
NUM_CLASSES: Final[int] = 7

# Official split sizes
OFFICIAL_TRAIN_SAMPLES: Final[int] = 8926
OFFICIAL_TEST_SAMPLES: Final[int] = 7711
TOTAL_SAMPLES: Final[int] = 16637

# Official file names and MD5 checksums
TRAIN_FILE_NAME: Final[str] = "ElectricDevices_TRAIN.ts"
TEST_FILE_NAME: Final[str] = "ElectricDevices_TEST.ts"
TRAIN_MD5: Final[str] = "8d3995f89026af26a56b3841fdb10f12"
TEST_MD5: Final[str] = "7fc54f2fb8bedc88ee4ebbadd0e47c40"

# Official class labels (1 to 7) mapped to contiguous 0-indexed internal labels (0 to 6)
OFFICIAL_CLASSES: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6, 7)
OFFICIAL_TO_INTERNAL_LABEL: Final[dict[int, int]] = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
}
INTERNAL_TO_OFFICIAL_LABEL: Final[dict[int, int]] = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
}

# Training/Validation split specification on official train partition (approx 80% / 20%)
SPLIT_SEED: Final[int] = 42
VAL_SPLIT_RATIO: Final[float] = 0.20

# Training seeds (5 seeds x 8 configs = 40 training runs)
TRAIN_SEEDS: Final[list[int]] = [7, 21, 42, 84, 168]

# Parsimony threshold for pooling and model selection (0.5 percentage points = 0.005 Macro F1)
PARSIMONY_THRESHOLD: Final[float] = 0.005

# Expected parameter counts for architectures on ElectricDevices (input 96, 7 classes)
EXPECTED_PARAMETERS: Final[dict[str, int | dict[str, int]]] = {
    "E01_MLP_32": 3335,
    "E02_GRU_32_LAST": 3591,
    "E03_GRU_32_MEAN": 3591,
    "E04_GRU_32_MAX": 3591,
    "E05_GRU_32_MEAN_MAX": 3815,
    "E06_LSTM_32_BEST_POOLING": {
        "last": 4711,
        "mean": 4711,
        "max": 4711,
        "mean_max": 4935,
    },
    "E07_GRU_64_BEST_POOLING": {
        "last": 13319,
        "mean": 13319,
        "max": 13319,
        "mean_max": 13767,
    },
    "E08_LSTM_64_BEST_POOLING": {
        "last": 17607,
        "mean": 17607,
        "max": 17607,
        "mean_max": 18055,
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
