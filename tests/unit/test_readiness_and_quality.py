"""Tests for the technical readiness audit helpers and the quality-gate result handling."""

import json
from pathlib import Path

import pytest

from project4_rnn.audit.duplicates import AUDIT_ARTIFACT_NAMES
from project4_rnn.audit.quality import (
    QualityCommand,
    parse_coverage_percent,
    run_quality_checks,
)
from project4_rnn.audit.readiness import (
    NEXT_RUN_SECTION,
    NOT_EXECUTED_STATEMENT,
    NOT_READY_LINE,
    READY_LINE,
    REQUIRED_DOCS,
    check_documentation,
    check_duplicate_audit_artifacts,
    check_provenance_schemas,
    check_quality_results,
    check_sensitivity_functions,
    check_smoke_report,
    check_timing_consolidation,
    render_readiness_markdown,
    run_readiness_audit,
)
from project4_rnn.audit.smoke import SMOKE_BANNER, SMOKE_PASS_LINE

pytestmark = pytest.mark.unit


def test_machinery_checks_pass() -> None:
    assert "verified" in check_sensitivity_functions()
    assert "legitimate 0.0" in check_timing_consolidation()
    assert "schemas enforced" in check_provenance_schemas()


def _write_docs(root: Path, readme_extra: str = "") -> None:
    for rel in REQUIRED_DOCS:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("doc", encoding="utf-8")
    (root / "README.md").write_text(readme_extra, encoding="utf-8")


def test_documentation_check_passes_with_required_sections(tmp_path: Path) -> None:
    _write_docs(tmp_path, f"## {NEXT_RUN_SECTION}\nO experimento {NOT_EXECUTED_STATEMENT}.")
    assert "documents present" in check_documentation(tmp_path)


def test_documentation_check_reports_every_problem(tmp_path: Path) -> None:
    _write_docs(tmp_path, "nothing")
    (tmp_path / "docs" / "DATASET.md").unlink()
    with pytest.raises(ValueError) as excinfo:
        check_documentation(tmp_path)
    message = str(excinfo.value)
    assert "docs/DATASET.md: missing" in message
    assert NEXT_RUN_SECTION in message
    assert "not executed" in message


def test_smoke_report_check(tmp_path: Path) -> None:
    report = tmp_path / "smoke.txt"
    with pytest.raises(FileNotFoundError):
        check_smoke_report(report)
    report.write_text(SMOKE_PASS_LINE, encoding="utf-8")
    with pytest.raises(ValueError, match="banner"):
        check_smoke_report(report)
    report.write_text(SMOKE_BANNER, encoding="utf-8")
    with pytest.raises(ValueError, match="pass"):
        check_smoke_report(report)
    report.write_text(f"{SMOKE_BANNER}\n{SMOKE_PASS_LINE}", encoding="utf-8")
    assert "passed" in check_smoke_report(report)


def test_duplicate_audit_artifact_check(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        check_duplicate_audit_artifacts(tmp_path)
    for name in AUDIT_ARTIFACT_NAMES:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    audit = {"pairwise_overlaps": {"train_vs_validation": {"shared_hashes": 2}}}
    (tmp_path / "duplicate_overlap_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="2 shared"):
        check_duplicate_audit_artifacts(tmp_path)
    audit["pairwise_overlaps"]["train_vs_validation"]["shared_hashes"] = 0
    (tmp_path / "duplicate_overlap_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    assert "shared = 0" in check_duplicate_audit_artifacts(tmp_path)


def test_quality_results_missing_file_fails_every_check(tmp_path: Path) -> None:
    checks = check_quality_results(tmp_path)
    assert len(checks) == 5
    assert not any(c.passed for c in checks)


def test_quality_results_reflect_recorded_outcomes(tmp_path: Path) -> None:
    results = {
        "ruff": {"passed": True, "returncode": 0},
        "black": {"passed": True, "returncode": 0},
        "mypy": {"passed": False, "returncode": 1},
        "coverage_percent": 71.5,
    }
    (tmp_path / "quality_results.json").write_text(json.dumps(results), encoding="utf-8")
    outcome = {c.check_id: c.passed for c in check_quality_results(tmp_path)}
    assert outcome == {
        "QUAL_RUFF": True,
        "QUAL_BLACK": True,
        "QUAL_MYPY": False,
        "QUAL_PYTEST": False,
        "QUAL_COVERAGE": True,
    }


def test_coverage_below_minimum_fails(tmp_path: Path) -> None:
    (tmp_path / "quality_results.json").write_text('{"coverage_percent": 12.0}', encoding="utf-8")
    coverage = [c for c in check_quality_results(tmp_path) if c.check_id == "QUAL_COVERAGE"]
    assert coverage[0].passed is False


def test_parse_coverage_percent(tmp_path: Path) -> None:
    xml = tmp_path / "coverage.xml"
    assert parse_coverage_percent(xml) is None
    xml.write_text("<coverage", encoding="utf-8")
    assert parse_coverage_percent(xml) is None
    xml.write_text('<coverage line-rate="0.7234"></coverage>', encoding="utf-8")
    assert parse_coverage_percent(xml) == 72.34
    xml.write_text("<coverage></coverage>", encoding="utf-8")
    assert parse_coverage_percent(xml) is None


def test_run_quality_checks_records_outcomes(tmp_path: Path) -> None:
    commands = (
        QualityCommand("ok", ("-c", "print('fine')")),
        QualityCommand("bad", ("-c", "import sys; sys.exit(3)")),
    )
    out = tmp_path / "quality" / "quality_results.json"
    results = run_quality_checks(root=tmp_path, output_path=out, commands=commands)
    assert results["ok"]["passed"] is True
    assert results["bad"]["returncode"] == 3
    assert results["passed"] is False
    assert results["coverage_percent"] is None
    assert json.loads(out.read_text(encoding="utf-8"))["bad"]["passed"] is False


def test_render_readiness_markdown_lists_checks() -> None:
    result = {
        "status": NOT_READY_LINE,
        "timestamp": "t",
        "scope_note": "scope",
        "checks": [{"check_id": "X", "name": "n", "passed": False, "details": "a|b"}],
    }
    text = render_readiness_markdown(result)
    assert NOT_READY_LINE in text and "FAIL" in text and "a/b" in text


def test_readiness_audit_is_not_ready_on_an_empty_project(tmp_path: Path) -> None:
    config = Path("configs/base.yaml").resolve()
    result = run_readiness_audit(
        base_config_path=config,
        output_dir=tmp_path / "audits",
        quality_dir=tmp_path / "quality",
        root=tmp_path,
    )
    assert result["ready"] is False
    assert result["status"] == NOT_READY_LINE != READY_LINE
    passed = {c["check_id"]: c["passed"] for c in result["checks"]}
    assert passed["SENSITIVITY"] and passed["TIMING"] and passed["PROVENANCE"] and passed["DRY_RUN"]
    assert not passed["DOCUMENTATION"] and not passed["SMOKE"]
    saved = json.loads((tmp_path / "audits" / "readiness_audit.json").read_text("utf-8"))
    assert saved["status"] == NOT_READY_LINE
    assert (tmp_path / "audits" / "readiness_audit.md").is_file()
