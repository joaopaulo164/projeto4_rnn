.PHONY: setup quality test smoke data phase-1 phase-2 phase-3 phase-4 experiment evidence audit clean

PYTHON ?= python

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

quality:
	$(PYTHON) -m ruff check src tests configs
	$(PYTHON) -m black --check src tests
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest tests/unit tests/integration -v --cov=project4_rnn --cov-report=term-missing

smoke:
	$(PYTHON) -m project4_rnn.cli smoke-test

data:
	$(PYTHON) -m project4_rnn.cli prepare-data --config configs/base.yaml
	$(PYTHON) -m project4_rnn.cli validate-data --config configs/base.yaml

phase-1:
	$(PYTHON) -m project4_rnn.cli run-phase --phase 1 --config configs/base.yaml

phase-2:
	$(PYTHON) -m project4_rnn.cli run-phase --phase 2 --config configs/base.yaml

phase-3:
	$(PYTHON) -m project4_rnn.cli run-phase --phase 3 --config configs/base.yaml

phase-4:
	$(PYTHON) -m project4_rnn.cli run-phase --phase 4 --config configs/base.yaml

experiment:
	$(PYTHON) -m project4_rnn.cli run-full-experiment --config configs/base.yaml

evidence:
	$(PYTHON) -m project4_rnn.cli generate-evidence --artifacts-dir artifacts

audit:
	$(PYTHON) -m project4_rnn.cli audit --artifacts-dir artifacts

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').glob('**/__pycache__')]; [p.unlink() for p in pathlib.Path('.').glob('**/*.pyc')]"
