"""Artifact verification and file existence audit routines."""

import json
from pathlib import Path
from typing import Any


def verify_file_exists_and_non_empty(file_path: Path) -> tuple[bool, str]:
    """Check if file exists and has size > 0."""
    p = Path(file_path)
    if not p.exists():
        return False, f"File missing: {p}"
    if p.stat().st_size == 0:
        return False, f"File is empty (0 bytes): {p}"
    return True, "OK"


def verify_json_validity(file_path: Path) -> tuple[bool, str, dict[str, Any] | None]:
    """Check if file exists and contains valid JSON."""
    ok, msg = verify_file_exists_and_non_empty(file_path)
    if not ok:
        return False, msg, None

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        return True, "OK", data
    except Exception as e:
        return False, f"Invalid JSON syntax in {file_path}: {e}", None
