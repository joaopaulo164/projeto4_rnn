.PHONY: setup quality test coverage smoke data audit-duplicates audit-readiness dry-run phase-1 phase-2 phase-3 phase-4 experiment evidence audit clean-generated clean

PYTHON ?= python

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

quality:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest

coverage:
	$(PYTHON) -m pytest --cov=src/project4_rnn --cov-report=term-missing --cov-report=xml

smoke:
	$(PYTHON) -m project4_rnn.cli smoke-test

data:
	$(PYTHON) -m project4_rnn.cli prepare-data --config configs/base.yaml
	$(PYTHON) -m project4_rnn.cli validate-data --config configs/base.yaml

audit-duplicates:
	$(PYTHON) -m project4_rnn.cli audit-duplicates --config configs/base.yaml

audit-readiness:
	$(PYTHON) -m project4_rnn.cli audit-readiness --config configs/base.yaml

dry-run:
	$(PYTHON) -m project4_rnn.cli run-full-experiment --config configs/base.yaml --dry-run

phase-1:
	$(PYTHON) -m project4_rnn.cli run-phase --phase 1 --config configs/base.yaml

phase-2:
	$(PYTHON) -m project4_rnn.cli run-phase --phase 2 --config configs/base.yaml

phase-3:
	$(PYTHON) -m project4_rnn.cli run-phase --phase 3 --config configs/base.yaml

phase-4:
	$(PYTHON) -m project4_rnn.cli run-phase --phase 4 --config configs/base.yaml

# Real, long-running experiment: run it deliberately, never as part of development checks.
experiment:
	$(PYTHON) -m project4_rnn.cli run-full-experiment --config configs/base.yaml --fail-fast

evidence:
	$(PYTHON) -m project4_rnn.cli generate-evidence --artifacts-dir artifacts

audit:
	$(PYTHON) -m project4_rnn.cli audit --artifacts-dir artifacts

clean-generated:
	$(PYTHON) -m project4_rnn.cli clean-artifacts --generated-only

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').glob('**/__pycache__')]; [p.unlink() for p in pathlib.Path('.').glob('**/*.pyc')]"
