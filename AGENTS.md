# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Commands

```bash
# Setup editable package with dev tools
python -m pip install -e ".[dev]"

# Testing: All tests, single file, single test by name expression, or by node ID
pytest
pytest tests/unit/test_models_and_params.py
pytest tests/unit/test_models_and_params.py -k test_mlp32_architecture_and_parameter_count
pytest tests/unit/test_models_and_params.py::test_temporal_pooling_strategies
pytest -m unit          # Markers: unit, integration, smoke
pytest -m smoke

# Code quality & type checks
ruff check src tests configs
black --check src tests
mypy src

# CLI execution
python -m project4_rnn.cli smoke-test
python -m project4_rnn.cli prepare-data --config configs/base.yaml
python -m project4_rnn.cli run-phase --phase 1 --config configs/base.yaml
python -m project4_rnn.cli run-full-experiment --config configs/base.yaml
python -m project4_rnn.cli run-final-evaluation --config configs/base.yaml
python -m project4_rnn.cli audit --artifacts-dir artifacts
```

## Code Style Guidelines

- **Imports**: Sorted via Ruff/isort (`I`). Standard library first, third-party second, internal package (`project4_rnn.*`) last.
- **Formatting**: Line length 100 (`black` and `ruff`), target Python 3.11+.
- **Typing**: Strict `mypy` (`disallow_untyped_defs = true`). Use Python 3.11+ built-in generics (`list[T]`, `dict[K, V]`, `tuple[T, ...]`, `T | None`).
- **Naming**: `snake_case` for functions/variables/modules, `PascalCase` for classes, `UPPER_SNAKE_CASE` for `Final` constants. Matrix/tensor uppercase variable names (`X_train`, `X_val`, `X_test`, `y_train`) are explicitly allowed via Ruff ignores `N803`, `N806`.
- **Error Handling**: Raise explicit domain exceptions (`ValueError`, `RuntimeError`, `DatasetContractError`) with descriptive context. Log unhandled exceptions in CLI entry points with `logger.error(..., exc_info=True)`.

## Critical Project Conventions & Gotchas

- **Parameter Count Verification**: `build_model(config)` in [`models/factory.py`](src/project4_rnn/models/factory.py:14) strictly enforces exact parameter counts against [`EXPECTED_PARAMETERS`](src/project4_rnn/constants.py:56) for ElectricDevices (E01 MLP-32 = 3,335; E02-E04 GRU-32 = 3,591; E05 GRU-32 mean_max = 3,815; E06 LSTM-32 = 4,711/4,935; E07 GRU-64 = 13,319/13,767; E08 LSTM-64 = 17,607/18,055).
- **Dynamic Pooling Resolution**: Phase 3/4 configs specify `pooling: "best_pooling"`. `build_model()` raises `ValueError` if instantiated before Phase 2 generates `artifacts/summary/pooling_selection.json` or before `pooling_override` is passed to `build_experiment_config()`.
- **Data Contracts & Label Mapping**: Official ElectricDevices files are validated via MD5 (`8d39...` train, `7fc5...` test) and shape (8,926 / 7,711 sequences of length 96, 7 classes). Official 1-indexed labels (1..7) must be mapped to 0-indexed internal labels (0..6) via `OFFICIAL_TO_INTERNAL_LABEL`.
- **Partition Isolation**: Official TRAIN (8,926 samples) is split into ~7,141 train / ~1,785 val (seed 42 in `data/processed/split_indices.json`). The 7,711 official TEST samples are strictly reserved for final evaluation of the winning model.
- **Parsimony Rule**: Model and pooling selection uses `PARSIMONY_THRESHOLD = 0.005` (0.5 pp Macro F1) in [`constants.py`](src/project4_rnn/constants.py:53) to favor simpler architectures (fewer parameters) when Macro F1 differences are within threshold.
