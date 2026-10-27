"""Tests for the scientific compliance auditor on empty, partial and corrupt artifact trees.

The auditor is a final-evidence audit; here it must fail loudly on incomplete trees and must never
crash, and it must write its report.
"""

from pathlib import Path

import pytest

from project4_rnn.audit.compliance import ComplianceAuditor
from project4_rnn.audit.report import generate_audit_report

pytestmark = pytest.mark.unit

EXPECTED_IDS = {
    "AUDIT_01_RUN_COMPLETENESS",
    "AUDIT_02_RUN_ARTIFACTS",
    "AUDIT_09_GROUPED_SPLIT",
    "AUDIT_10_TIMING_CONSOLIDATION",
    "AUDIT_11_PROVENANCE",
    "AUDIT_12_DUPLICATE_AUDIT",
    "AUDIT_13_SENSITIVITY_EVALUATION",
}


def test_empty_tree_fails_critical_checks_without_crashing(tmp_path: Path) -> None:
    auditor = ComplianceAuditor(tmp_path / "artifacts", tmp_path / "processed")
    results = auditor.run_full_audit()
    ids = {r.check_id for r in results}
    assert EXPECTED_IDS <= ids
    by_id = {r.check_id: r for r in results}
    assert by_id["AUDIT_01_RUN_COMPLETENESS"].passed is False
    assert "0/40" in by_id["AUDIT_01_RUN_COMPLETENESS"].details
    assert by_id["AUDIT_09_GROUPED_SPLIT"].passed is False
    assert by_id["AUDIT_12_DUPLICATE_AUDIT"].passed is False
    assert by_id["AUDIT_13_SENSITIVITY_EVALUATION"].severity == "WARNING"


def test_rerunning_audit_does_not_accumulate_results(tmp_path: Path) -> None:
    auditor = ComplianceAuditor(tmp_path / "artifacts", tmp_path / "processed")
    first = len(auditor.run_full_audit())
    assert len(auditor.run_full_audit()) == first


def test_corrupt_and_incomplete_status_files_are_reported(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    corrupt = artifacts / "runs" / "E01_MLP_32" / "42"
    corrupt.mkdir(parents=True)
    (corrupt / "status.json").write_text("{not json", encoding="utf-8")
    running = artifacts / "runs" / "E01_MLP_32" / "7"
    running.mkdir(parents=True)
    (running / "status.json").write_text('{"status": "RUNNING"}', encoding="utf-8")
    results = ComplianceAuditor(artifacts, tmp_path / "processed").run_full_audit()
    first = next(r for r in results if r.check_id == "AUDIT_01_RUN_COMPLETENESS")
    assert "corrupt status.json" in first.details
    assert "status=RUNNING" in first.details
    second = next(r for r in results if r.check_id == "AUDIT_02_RUN_ARTIFACTS")
    assert second.passed is False


def test_report_is_written_for_failing_audit(tmp_path: Path) -> None:
    auditor = ComplianceAuditor(tmp_path / "a", tmp_path / "p")
    target = tmp_path / "out" / "final_compliance_audit.md"
    assert generate_audit_report(auditor, target) is False
    text = target.read_text(encoding="utf-8")
    assert "REPROVADO" in text and "AUDIT_01_RUN_COMPLETENESS" in text
