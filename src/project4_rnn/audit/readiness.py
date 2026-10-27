"""Technical readiness audit: is the project ready for the future full experiment?

This is deliberately different from the final scientific compliance audit. It never requires the
40 runs, a scientific pooling or model selection, or a test evaluation. It verifies the dataset
contract, the hash-grouped split, the duplicate audit artifacts, the sensitivity/timing/provenance
machinery (on in-memory fixtures), the recorded quality results, the smoke test report, the dry-run
and the documentation.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from project4_rnn.audit.duplicates import AUDIT_ARTIFACT_NAMES, sensitivity_masks
from project4_rnn.audit.quality import COVERAGE_MINIMUM_PERCENT, QUALITY_RESULTS_FILE
from project4_rnn.audit.smoke import SMOKE_BANNER, SMOKE_PASS_LINE
from project4_rnn.config import build_experiment_config
from project4_rnn.constants import (
    NUM_CLASSES,
    OFFICIAL_TEST_SAMPLES,
    OFFICIAL_TRAIN_SAMPLES,
    SEQUENCE_LENGTH,
)
from project4_rnn.data.hashing import hash_series_matrix
from project4_rnn.data.loader import load_raw_electricdevices
from project4_rnn.data.splitting import verify_split_payload
from project4_rnn.experiments.provenance import (
    FINAL_SELECTION_REQUIRED_FIELDS,
    FINAL_TEST_REQUIRED_FIELDS,
    ProvenanceError,
    validate_final_selection,
    validate_final_test_metrics,
)
from project4_rnn.experiments.timing import IncompleteRunError, validate_timing_fields

logger = logging.getLogger("project4_rnn.audit.readiness")

READY_LINE = "READY FOR FULL EXPERIMENT"
NOT_READY_LINE = "NOT READY FOR FULL EXPERIMENT"

REQUIRED_DOCS: tuple[str, ...] = (
    "README.md",
    "docs/PROJECT_SCOPE.md",
    "docs/REQUIREMENTS.md",
    "docs/ARCHITECTURE.md",
    "docs/DATASET.md",
    "docs/DATA_GOVERNANCE.md",
    "docs/METHODOLOGY.md",
    "docs/EXPERIMENTAL_PROTOCOL.md",
    "docs/MODEL_ARCHITECTURES.md",
    "docs/POOLING_ABLATION.md",
    "docs/PIPELINE_ORCHESTRATION.md",
    "docs/COMMAND_REFERENCE.md",
    "docs/CONFIGURATION.md",
    "docs/EXECUTION_GUIDE.md",
    "docs/TESTING_STRATEGY.md",
    "docs/REPRODUCIBILITY.md",
    "docs/ARTIFACTS_AND_EVIDENCE.md",
    "docs/RESULTS_GUIDE.md",
    "docs/LIMITATIONS.md",
    "docs/TROUBLESHOOTING.md",
    "docs/SCIENTIFIC_ARTICLE_GUIDE.md",
)
NEXT_RUN_SECTION = "PRÓXIMA EXECUÇÃO EXPERIMENTAL"
NOT_EXECUTED_STATEMENT = "não foi executado"


@dataclass
class ReadinessCheck:
    """One readiness check."""

    check_id: str
    name: str
    passed: bool
    details: str


def _check(check_id: str, name: str, fn: Callable[[], str]) -> ReadinessCheck:
    try:
        return ReadinessCheck(check_id, name, True, fn())
    except Exception as exc:  # the audit reports every failure instead of stopping at the first
        logger.error(f"Readiness check {check_id} failed: {exc}")
        return ReadinessCheck(check_id, name, False, f"{type(exc).__name__}: {exc}")


def check_dataset_and_split(base_config_path: Path) -> str:
    """Verify the official data contract and the persisted hash-grouped split (read-only)."""
    cfg = build_experiment_config(base_config_path)
    X_dev, y_dev, X_test, y_test, manifest = load_raw_electricdevices(
        raw_dir=Path(cfg.data.raw_dir), verify_checksums=True, processed_dir=None
    )
    if X_dev.shape != (OFFICIAL_TRAIN_SAMPLES, SEQUENCE_LENGTH):
        raise ValueError(f"unexpected train shape {X_dev.shape}")
    if X_test.shape != (OFFICIAL_TEST_SAMPLES, SEQUENCE_LENGTH):
        raise ValueError(f"unexpected test shape {X_test.shape}")
    if len(set(y_dev.tolist())) != NUM_CLASSES or len(set(y_test.tolist())) != NUM_CLASSES:
        raise ValueError("official partitions do not contain exactly 7 classes")
    if not (np.isfinite(X_dev).all() and np.isfinite(X_test).all()):
        raise ValueError("official data contains NaN/Inf")

    split_path = Path(cfg.data.processed_dir) / "split_indices.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"{split_path} not found; run 'prepare-data'")
    with open(split_path, encoding="utf-8") as f:
        payload = json.load(f)
    dev_hashes = hash_series_matrix(X_dev)
    train_idx, val_idx = verify_split_payload(
        payload,
        y_dev,
        dev_hashes,
        str(manifest["dataset_hash"]),
        cfg.data.split_seed,
        cfg.data.split_n_splits,
    )
    shared = {dev_hashes[int(i)] for i in train_idx} & {dev_hashes[int(i)] for i in val_idx}
    if shared:
        raise ValueError(f"{len(shared)} hashes shared by train and validation")
    conflicts = payload["conflicting_groups"]["count"]
    return (
        f"contract OK (8,926/7,711, length 96, 7 classes); split {payload['split_hash'][:12]}: "
        f"train={len(train_idx)} val={len(val_idx)} shared_hashes=0; "
        f"{conflicts} conflicting groups kept whole"
    )


def check_duplicate_audit_artifacts(audits_dir: Path) -> str:
    """The duplicate audit must have been generated and report zero train/validation overlap."""
    audits_dir = Path(audits_dir)
    missing = [n for n in AUDIT_ARTIFACT_NAMES if not (audits_dir / n).is_file()]
    if missing:
        raise FileNotFoundError(f"duplicate audit artifacts missing: {missing}")
    with open(audits_dir / "duplicate_overlap_audit.json", encoding="utf-8") as f:
        audit = json.load(f)
    shared = audit["pairwise_overlaps"]["train_vs_validation"]["shared_hashes"]
    if shared != 0:
        raise ValueError(f"duplicate audit reports {shared} shared train/validation hashes")
    return f"{len(AUDIT_ARTIFACT_NAMES)} audit artifacts present; train/validation shared = 0"


def check_sensitivity_functions() -> str:
    """Sensitivity filters behave correctly on a controlled in-memory fixture."""
    dev_hashes = ["a", "b", "c", "d"]
    dev_labels = [1, 2, 3, 3]
    test_hashes = ["a", "x", "c", "y", "d"]
    test_labels = [1, 4, 5, 6, 3]
    masks = sensitivity_masks(dev_hashes, dev_labels, test_hashes, test_labels)
    no_overlap = masks["no_exact_overlap"].tolist()
    no_conflict = masks["without_label_conflicts"].tolist()
    if no_overlap != [False, True, False, True, False]:
        raise ValueError(f"no_exact_overlap mask wrong: {no_overlap}")
    if no_conflict != [True, True, False, True, True]:
        raise ValueError(f"without_label_conflicts mask wrong: {no_conflict}")
    return "no-exact-overlap and without-label-conflicts filters verified on fixture"


def check_timing_consolidation() -> str:
    """Timings: zero is legitimate, absence and wrong types are rejected (never defaulted)."""
    ok = {
        "total_training_time_sec": 0.0,
        "mean_epoch_time_sec": 0.0,
        "inference_time_sec": 0.0,
        "total_epochs": 3,
    }
    validate_timing_fields(ok, "fixture/seed_0")
    for broken in (
        {k: v for k, v in ok.items() if k != "inference_time_sec"},
        {**ok, "mean_epoch_time_sec": "fast"},
    ):
        try:
            validate_timing_fields(broken, "fixture/seed_1")
        except IncompleteRunError:
            continue
        raise ValueError("an invalid timing record was accepted")
    return "legitimate 0.0 accepted; missing and invalid timing fields halt consolidation"


def check_provenance_schemas() -> str:
    """The provenance validators require every mandatory field."""
    selection: dict[str, Any] = dict.fromkeys(FINAL_SELECTION_REQUIRED_FIELDS, "x")
    selection.update(
        checkpoint_sha256="0" * 64,
        best_val_loss=0.1,
        val_macro_f1=0.1,
        val_balanced_accuracy=0.1,
    )
    validate_final_selection(selection)
    test_doc: dict[str, Any] = dict.fromkeys(FINAL_TEST_REQUIRED_FIELDS, "x")
    validate_final_test_metrics(test_doc)
    for validator, doc, field_name in (
        (validate_final_selection, selection, "split_hash"),
        (validate_final_test_metrics, test_doc, "checkpoint_sha256"),
    ):
        broken = {k: v for k, v in doc.items() if k != field_name}
        try:
            validator(broken)
        except ProvenanceError:
            continue
        raise ValueError(f"missing '{field_name}' was not detected")
    return (
        f"final_model_selection ({len(FINAL_SELECTION_REQUIRED_FIELDS)} fields) and "
        f"final_test_metrics ({len(FINAL_TEST_REQUIRED_FIELDS)} fields) schemas enforced"
    )


def check_quality_results(quality_dir: Path) -> list[ReadinessCheck]:
    """Ruff, Black, mypy, pytest and coverage, from the persisted quality-gate results."""
    path = Path(quality_dir) / QUALITY_RESULTS_FILE
    labels = {
        "ruff": ("QUAL_RUFF", "Ruff"),
        "black": ("QUAL_BLACK", "Black"),
        "mypy": ("QUAL_MYPY", "mypy"),
        "pytest": ("QUAL_PYTEST", "pytest"),
    }
    if not path.is_file():
        hint = f"{path} missing; run 'audit-readiness --run-quality'"
        missing = [ReadinessCheck(cid, name, False, hint) for cid, name in labels.values()]
        missing.append(ReadinessCheck("QUAL_COVERAGE", "Coverage", False, hint))
        return missing
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    checks: list[ReadinessCheck] = []
    for key, (cid, name) in labels.items():
        entry = results.get(key)
        passed = bool(entry and entry.get("passed"))
        detail = f"returncode {entry.get('returncode')}" if entry else "no result recorded"
        checks.append(ReadinessCheck(cid, name, passed, detail))
    coverage = results.get("coverage_percent")
    checks.append(
        ReadinessCheck(
            "QUAL_COVERAGE",
            "Coverage",
            coverage is not None and coverage >= COVERAGE_MINIMUM_PERCENT,
            f"{coverage}% (minimum {COVERAGE_MINIMUM_PERCENT}%)",
        )
    )
    return checks


def check_smoke_report(report_path: Path) -> str:
    """The reduced smoke test report must exist, pass and be labelled as non-scientific."""
    report_path = Path(report_path)
    if not report_path.is_file():
        raise FileNotFoundError(f"{report_path} missing; run 'smoke-test'")
    text = report_path.read_text(encoding="utf-8")
    if SMOKE_BANNER not in text:
        raise ValueError("smoke report lacks the technical-test banner")
    if SMOKE_PASS_LINE not in text:
        raise ValueError("smoke report does not record a pass")
    return "smoke test report present, passed and labelled as technical-only"


def check_dry_run(base_config_path: Path) -> str:
    """The dry-run plans 8 configurations x 5 seeds and has no data_validation node."""
    from project4_rnn.orchestration.pipeline import run_full_pipeline

    result = run_full_pipeline(base_config_path=base_config_path, dry_run=True)
    if result["status"] != "DRY_RUN_COMPLETED":
        raise ValueError(f"unexpected dry-run status {result['status']}")
    plan = result["plan"]
    if len(plan["experiments"]) != 8 or len(plan["seeds"]) != 5 or plan["planned_runs"] != 40:
        raise ValueError(
            f"expected 8 configurations x 5 seeds = 40 runs, got {len(plan['experiments'])} x "
            f"{len(plan['seeds'])} = {plan['planned_runs']}"
        )
    step_ids = {s["step_id"] for s in plan["steps"]}
    if "data_validation" in step_ids:
        raise ValueError("redundant data_validation node is still in the DAG")
    return "dry-run plans 8 configurations x 5 seeds = 40 future runs; nothing executed"


def check_documentation(root: Path) -> str:
    """All required documents exist; the README has the next-run section and the scope notice."""
    root = Path(root)
    problems = [f"{rel}: missing" for rel in REQUIRED_DOCS if not (root / rel).is_file()]
    readme = root / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8")
        if NEXT_RUN_SECTION not in text:
            problems.append(f"README.md: lacks section '{NEXT_RUN_SECTION}'")
        if NOT_EXECUTED_STATEMENT not in text:
            problems.append("README.md: lacks the statement that the experiment was not executed")
    if problems:
        raise ValueError("; ".join(problems))
    return f"{len(REQUIRED_DOCS)} documents present; next-run section and scope notice found"


def render_readiness_markdown(result: dict[str, Any]) -> str:
    """Markdown rendering of the readiness audit."""
    lines = [
        "# Auditoria de prontidão técnica",
        "",
        f"**{result['status']}**",
        "",
        f"- Data: `{result['timestamp']}`",
        f"- {result['scope_note']}",
        "",
        "| Verificação | Descrição | Status | Detalhes |",
        "|---|---|---|---|",
    ]
    for c in result["checks"]:
        status = "PASS" if c["passed"] else "FAIL"
        details = str(c["details"]).replace("|", "/").replace("\n", " ")
        lines.append(f"| `{c['check_id']}` | {c['name']} | **{status}** | {details} |")
    lines.append("")
    return "\n".join(lines)


def run_readiness_audit(
    base_config_path: Path = Path("configs/base.yaml"),
    output_dir: Path = Path("artifacts/audits"),
    quality_dir: Path = Path("artifacts/quality"),
    root: Path = Path("."),
) -> dict[str, Any]:
    """Run all readiness checks and write ``readiness_audit.md`` / ``readiness_audit.json``."""
    base_config_path = Path(base_config_path)
    output_dir = Path(output_dir)
    quality_dir = Path(quality_dir)

    checks: list[ReadinessCheck] = [
        _check(
            "DATA_CONTRACT_SPLIT",
            "Dataset contract and hash-grouped split",
            lambda: check_dataset_and_split(base_config_path),
        ),
        _check(
            "DUPLICATE_AUDIT",
            "Duplicate/overlap audit artifacts",
            lambda: check_duplicate_audit_artifacts(output_dir),
        ),
        _check("SENSITIVITY", "Sensitivity evaluation filters", check_sensitivity_functions),
        _check("TIMING", "Timing consolidation rules", check_timing_consolidation),
        _check("PROVENANCE", "Provenance schemas", check_provenance_schemas),
    ]
    checks.extend(check_quality_results(quality_dir))
    checks.extend(
        [
            _check(
                "SMOKE",
                "Reduced smoke test",
                lambda: check_smoke_report(quality_dir / "smoke_test_report.txt"),
            ),
            _check("DRY_RUN", "Pipeline dry-run", lambda: check_dry_run(base_config_path)),
            _check("DOCUMENTATION", "Documentation", lambda: check_documentation(root)),
        ]
    )

    ready = all(c.passed for c in checks)
    result: dict[str, Any] = {
        "status": READY_LINE if ready else NOT_READY_LINE,
        "ready": ready,
        "timestamp": datetime.now(UTC).isoformat(),
        "scope_note": (
            "Technical readiness only. The full experiment was not executed; no scientific "
            "pooling/model selection or test evaluation is required or implied."
        ),
        "checks": [asdict(c) for c in checks],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "readiness_audit.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    (output_dir / "readiness_audit.md").write_text(render_readiness_markdown(result), "utf-8")
    return result
