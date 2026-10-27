"""Smoke test verifying the reduced technical smoke test through its CLI entry point."""

import argparse
from pathlib import Path

import pytest

from project4_rnn.audit.smoke import SMOKE_BANNER, SMOKE_PASS_LINE
from project4_rnn.cli import cmd_smoke_test


@pytest.mark.smoke
def test_smoke_cli_command(tmp_path: Path) -> None:
    """Run the reduced technical smoke test and check its report."""
    report = tmp_path / "smoke_test_report.txt"
    exit_code = cmd_smoke_test(argparse.Namespace(report=str(report)))
    assert exit_code == 0
    text = report.read_text(encoding="utf-8")
    assert SMOKE_BANNER in text
    assert SMOKE_PASS_LINE in text
