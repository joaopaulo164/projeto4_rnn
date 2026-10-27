"""Software-quality gate: Ruff, Black, mypy, pytest and coverage, with a persisted result file.

The results are written to ``artifacts/quality/quality_results.json`` so that the readiness audit
can verify them without re-running the whole test suite.
"""

import json
import logging
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("project4_rnn.audit.quality")

QUALITY_RESULTS_FILE = "quality_results.json"
COVERAGE_MINIMUM_PERCENT = 70.0
OUTPUT_TAIL_LINES = 25


@dataclass(frozen=True)
class QualityCommand:
    """A quality-gate command and the key under which its result is stored."""

    key: str
    args: tuple[str, ...]


QUALITY_COMMANDS: tuple[QualityCommand, ...] = (
    QualityCommand("ruff", ("-m", "ruff", "check", ".")),
    QualityCommand("black", ("-m", "black", "--check", ".")),
    QualityCommand("mypy", ("-m", "mypy", "src")),
    QualityCommand(
        "pytest",
        (
            "-m",
            "pytest",
            "--cov=src/project4_rnn",
            "--cov-report=term-missing",
            "--cov-report=xml",
        ),
    ),
)


def parse_coverage_percent(coverage_xml: Path) -> float | None:
    """Total line coverage (percent) from a Cobertura ``coverage.xml`` or None when unavailable."""
    coverage_xml = Path(coverage_xml)
    if not coverage_xml.is_file():
        return None
    try:
        root = ET.parse(coverage_xml).getroot()
    except ET.ParseError:
        return None
    rate = root.get("line-rate")
    return round(float(rate) * 100.0, 2) if rate is not None else None


def _tail(text: str) -> str:
    return "\n".join(text.strip().splitlines()[-OUTPUT_TAIL_LINES:])


def run_quality_checks(
    root: Path = Path("."),
    output_path: Path = Path("artifacts/quality") / QUALITY_RESULTS_FILE,
    commands: tuple[QualityCommand, ...] = QUALITY_COMMANDS,
) -> dict[str, Any]:
    """Run the quality commands and persist their outcome.

    Returns:
        Mapping with one entry per command (returncode, passed, output tail), the measured
        coverage percentage, and an overall ``passed`` flag.
    """
    root = Path(root)
    results: dict[str, Any] = {}
    for command in commands:
        logger.info(f"Running quality check '{command.key}'")
        proc = subprocess.run(
            [sys.executable, *command.args],
            cwd=root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        results[command.key] = {
            "command": "python " + " ".join(command.args),
            "returncode": proc.returncode,
            "passed": proc.returncode == 0,
            "output_tail": _tail(proc.stdout + "\n" + proc.stderr),
        }
    coverage = parse_coverage_percent(root / "coverage.xml")
    results["coverage_percent"] = coverage
    results["coverage_minimum_percent"] = COVERAGE_MINIMUM_PERCENT
    results["coverage_ok"] = coverage is not None and coverage >= COVERAGE_MINIMUM_PERCENT
    results["timestamp"] = datetime.now(UTC).isoformat()
    results["passed"] = all(results[c.key]["passed"] for c in commands) and results["coverage_ok"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results
