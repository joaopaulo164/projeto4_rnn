"""Reduced technical smoke test on deterministic synthetic fixtures.

TESTE TÉCNICO — NÃO É RESULTADO CIENTÍFICO. The smoke test exercises the data contract, the
series hashing, the hash-grouped split, the duplicate audit, a two-epoch training run, checkpoint
restoration, timing consolidation, provenance and the sensitivity evaluation on tiny synthetic
data inside a temporary directory. It never reads the official dataset, never touches the real
``artifacts/`` tree, and its numbers are never reported as results.
"""

import json
import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from project4_rnn.audit.duplicates import build_duplicate_audit, write_audit_artifacts
from project4_rnn.config import ExperimentConfig, build_experiment_config
from project4_rnn.constants import NUM_CLASSES, SENSITIVITY_ARTIFACT_NAMES, SEQUENCE_LENGTH
from project4_rnn.data.hashing import hash_series_matrix
from project4_rnn.data.preprocessing import map_official_to_internal_labels
from project4_rnn.data.splitting import SplitResult, create_grouped_split
from project4_rnn.data.synthetic import make_synthetic_series, make_synthetic_test_series
from project4_rnn.evaluation.final_evaluation import (
    make_checkpoint_evaluator,
    run_sensitivity_evaluations,
    write_primary_artifacts,
)
from project4_rnn.experiments.consolidation import consolidate_experiments
from project4_rnn.experiments.provenance import (
    build_final_selection_provenance,
    build_final_test_provenance,
    select_best_seed,
    validate_final_selection,
    validate_final_test_metrics,
    verify_checkpoint_integrity,
)
from project4_rnn.experiments.runner import run_single_seed
from project4_rnn.experiments.timing import TIMING_FIELDS
from project4_rnn.models.factory import build_model
from project4_rnn.reproducibility import compute_object_hash, set_seed

logger = logging.getLogger("project4_rnn.audit.smoke")

SMOKE_BANNER = "TESTE TÉCNICO — NÃO É RESULTADO CIENTÍFICO."
SMOKE_PASS_LINE = "SMOKE TEST PASSED"
SMOKE_FAIL_LINE = "SMOKE TEST FAILED"
SMOKE_SEED = 42
SMOKE_MAX_EPOCHS = 2
SMOKE_EXPERIMENT_ID = "E01_MLP_32"


@dataclass
class SmokeCheck:
    """Outcome of one technical smoke check."""

    name: str
    passed: bool
    detail: str


class _SmokeRun:
    """Holds the typed fixture state shared by the ordered smoke checks."""

    def __init__(
        self,
        artifacts: Path,
        base_config_path: Path,
        mlp_config_path: Path,
        gru_config_path: Path,
    ) -> None:
        self.artifacts = artifacts
        self.base_config_path = base_config_path
        self.mlp_config_path = mlp_config_path
        self.gru_config_path = gru_config_path
        self.X_dev = np.empty((0, SEQUENCE_LENGTH))
        self.y_dev = np.empty(0, dtype=np.int64)
        self.X_test = np.empty((0, SEQUENCE_LENGTH))
        self.y_test = np.empty(0, dtype=np.int64)
        self.dev_hashes: list[str] = []
        self.test_hashes: list[str] = []
        self.dataset_hash = ""
        self.split: SplitResult | None = None
        self.cfg: ExperimentConfig | None = None

    def _split(self) -> SplitResult:
        if self.split is None:
            raise RuntimeError("split not created yet")
        return self.split

    def _cfg(self) -> ExperimentConfig:
        if self.cfg is None:
            raise RuntimeError("training configuration not built yet")
        return self.cfg

    def fixture_contract(self) -> str:
        set_seed(SMOKE_SEED, deterministic=True)
        self.X_dev, self.y_dev = make_synthetic_series()
        self.X_test, self.y_test = make_synthetic_test_series(self.X_dev, self.y_dev)
        if self.X_dev.shape[1] != SEQUENCE_LENGTH or self.X_test.shape[1] != SEQUENCE_LENGTH:
            raise ValueError("fixture series length is not 96")
        if not (np.isfinite(self.X_dev).all() and np.isfinite(self.X_test).all()):
            raise ValueError("fixture contains NaN/Inf")
        if sorted(set(self.y_dev.tolist())) != list(range(1, NUM_CLASSES + 1)):
            raise ValueError("fixture development labels do not cover the 7 classes")
        return f"{len(self.y_dev)} dev / {len(self.y_test)} test series, length 96, 7 classes"

    def hashing_and_split(self) -> str:
        self.dev_hashes = hash_series_matrix(self.X_dev)
        self.test_hashes = hash_series_matrix(self.X_test)
        if self.dev_hashes != hash_series_matrix(self.X_dev.copy()):
            raise ValueError("series hashing is not deterministic")
        self.dataset_hash = compute_object_hash(
            {"dev": self.dev_hashes, "test": self.test_hashes, "labels": self.y_dev.tolist()}
        )
        split = create_grouped_split(self.y_dev, self.dev_hashes, self.dataset_hash)
        train_h = {self.dev_hashes[int(i)] for i in split.train_idx}
        val_h = {self.dev_hashes[int(i)] for i in split.val_idx}
        if train_h & val_h:
            raise ValueError(f"{len(train_h & val_h)} hashes shared by train and validation")
        if len(split.train_idx) + len(split.val_idx) != len(self.y_dev):
            raise ValueError("split does not cover all samples")
        self.split = split
        return (
            f"train={len(split.train_idx)} val={len(split.val_idx)} "
            f"shared_hashes=0 split_hash={split.split_hash[:12]}"
        )

    def duplicate_audit(self) -> str:
        split = self._split()
        audit, tables = build_duplicate_audit(
            dev_hashes=self.dev_hashes,
            dev_labels=self.y_dev,
            test_hashes=self.test_hashes,
            test_labels=self.y_test,
            train_idx=split.train_idx,
            val_idx=split.val_idx,
            split_hash=split.split_hash,
        )
        written = write_audit_artifacts(audit, tables, self.artifacts / "audits")
        if audit["pairwise_overlaps"]["train_vs_validation"]["shared_hashes"] != 0:
            raise ValueError("duplicate audit reports train/validation overlap")
        return f"{len(written)} audit artifacts written to a temporary directory"

    def training_run(self) -> str:
        split = self._split()
        cfg = build_experiment_config(self.base_config_path, self.mlp_config_path)
        cfg.training.max_epochs = SMOKE_MAX_EPOCHS
        cfg.training.batch_size = 32
        tr, va = split.train_idx, split.val_idx
        summary = run_single_seed(
            config=cfg,
            seed=SMOKE_SEED,
            X_train=self.X_dev[tr].astype(np.float32),
            y_train=map_official_to_internal_labels(self.y_dev[tr]),
            X_val=self.X_dev[va].astype(np.float32),
            y_val=map_official_to_internal_labels(self.y_dev[va]),
            val_indices=va,
            dataset_hash=self.dataset_hash,
            split_hash=split.split_hash,
            artifacts_dir=self.artifacts,
            force=True,
        )
        if summary["total_epochs"] > SMOKE_MAX_EPOCHS:
            raise ValueError("smoke run exceeded the epoch limit")
        self.cfg = cfg
        return f"{cfg.experiment_id} seed {SMOKE_SEED}: {summary['total_epochs']} epoch(s)"

    def recurrent_forward_backward(self) -> str:
        cfg = build_experiment_config(self.base_config_path, self.gru_config_path)
        model = build_model(cfg, validate_params=True)
        logits = model(torch.randn(4, SEQUENCE_LENGTH, 1))
        if tuple(logits.shape) != (4, NUM_CLASSES):
            raise ValueError(f"expected 7 logits per sample, got shape {tuple(logits.shape)}")
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3]))
        loss.backward()
        if not any(p.grad is not None for p in model.parameters()):
            raise ValueError("backward produced no gradients")
        return "GRU-32 forward/backward with 7 logits"

    def timing_and_consolidation(self) -> str:
        run_dir = self.artifacts / "runs" / SMOKE_EXPERIMENT_ID / str(SMOKE_SEED)
        with open(run_dir / "execution_metadata.json", encoding="utf-8") as f:
            meta = json.load(f)
        missing = [k for k in TIMING_FIELDS if k not in meta]
        if missing:
            raise ValueError(f"timing fields missing: {missing}")
        result = consolidate_experiments(self.artifacts)
        if not result.get("exp_summaries"):
            raise ValueError("consolidation produced no summary")
        return "timing fields persisted and consolidated without zero defaults"

    def provenance_restore_sensitivity(self) -> str:
        split = self._split()
        cfg = self._cfg()
        seed_info = select_best_seed(
            self.artifacts / "runs" / SMOKE_EXPERIMENT_ID,
            expected_split_hash=split.split_hash,
            expected_dataset_hash=self.dataset_hash,
        )
        selection = build_final_selection_provenance(
            cfg, seed_info, self.dataset_hash, split.split_hash
        )
        validate_final_selection(selection)
        verify_checkpoint_integrity(selection)
        provenance = build_final_test_provenance(selection)
        evaluate = make_checkpoint_evaluator(
            cfg, Path(selection["checkpoint"]), int(selection["selected_seed"])
        )
        y_internal = map_official_to_internal_labels(self.y_test)
        X32 = self.X_test.astype(np.float32)
        metrics, preds, probs, _ = evaluate(X32, y_internal)
        if probs.shape != (len(self.y_test), NUM_CLASSES):
            raise ValueError(f"unexpected probability shape {probs.shape}")
        summary_dir = self.artifacts / "summary_fixture"
        document = write_primary_artifacts(
            summary_dir, metrics, preds, probs, y_internal, provenance, 1
        )
        validate_final_test_metrics(document)
        documents = run_sensitivity_evaluations(
            evaluate=evaluate,
            X_test=X32,
            y_test_internal=y_internal,
            y_test_official=self.y_test,
            dev_hashes=self.dev_hashes,
            dev_labels=self.y_dev.tolist(),
            test_hashes=self.test_hashes,
            full_document=document,
            provenance=provenance,
            summary_dir=summary_dir,
            phase=1,
        )
        absent = [n for n in SENSITIVITY_ARTIFACT_NAMES if not (summary_dir / n).is_file()]
        if absent:
            raise ValueError(f"sensitivity artifacts missing: {absent}")
        return (
            "checkpoint restored (SHA-256 verified), provenance complete, "
            f"{len(documents)} sensitivity variants written to a temporary directory"
        )


def _run_check(name: str, fn: Callable[[], str], checks: list[SmokeCheck]) -> bool:
    try:
        detail = fn()
    except Exception as exc:  # every failing check is reported in the smoke report
        logger.error(f"Smoke check '{name}' failed: {exc}", exc_info=True)
        checks.append(SmokeCheck(name, False, f"{type(exc).__name__}: {exc}"))
        return False
    checks.append(SmokeCheck(name, True, detail))
    return True


def render_smoke_report(checks: list[SmokeCheck]) -> str:
    """Text report; it never contains model metrics."""
    passed = all(c.passed for c in checks)
    lines = [
        SMOKE_BANNER,
        "",
        "Smoke test reduzido sobre fixture sintética (1 semente, no máximo 2 épocas).",
        "Nenhum dado oficial, checkpoint científico ou teste oficial foi utilizado.",
        f"Data: {datetime.now(UTC).isoformat()}",
        "",
    ]
    for c in checks:
        lines.append(f"[{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
    lines.extend(["", SMOKE_PASS_LINE if passed else SMOKE_FAIL_LINE, SMOKE_BANNER, ""])
    return "\n".join(lines)


def run_smoke_test(
    report_path: Path = Path("artifacts/quality/smoke_test_report.txt"),
    base_config_path: Path = Path("configs/base.yaml"),
    mlp_config_path: Path = Path("configs/phase_1/mlp_32.yaml"),
    gru_config_path: Path = Path("configs/phase_2/gru_32_mean.yaml"),
) -> bool:
    """Run the reduced technical smoke test and write its report.

    Returns:
        True when every check passed.
    """
    checks: list[SmokeCheck] = []
    with tempfile.TemporaryDirectory(prefix="p4_smoke_") as tmp:
        run = _SmokeRun(Path(tmp) / "artifacts", base_config_path, mlp_config_path, gru_config_path)
        ordered: tuple[tuple[str, Callable[[], str]], ...] = (
            ("fixture contract", run.fixture_contract),
            ("hashing and grouped split", run.hashing_and_split),
            ("duplicate audit", run.duplicate_audit),
            ("training (<=2 epochs)", run.training_run),
            ("recurrent forward/backward", run.recurrent_forward_backward),
            ("timing and consolidation", run.timing_and_consolidation),
            ("provenance, restore and sensitivity", run.provenance_restore_sensitivity),
        )
        for name, fn in ordered:
            if not _run_check(name, fn, checks):
                break

    report = render_smoke_report(checks)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return all(c.passed for c in checks)
