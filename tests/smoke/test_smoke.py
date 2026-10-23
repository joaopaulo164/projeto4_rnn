"""Smoke test verifying end-to-end functionality via CLI function."""

import argparse

import pytest

from project4_rnn.cli import cmd_smoke_test


@pytest.mark.smoke
def test_smoke_cli_command() -> None:
    """Run CLI smoke test directly."""
    args = argparse.Namespace()
    exit_code = cmd_smoke_test(args)
    assert exit_code == 0
