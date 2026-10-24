"""Artifact verification, cleanup, and file existence audit routines."""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger("project4_rnn.audit.artifacts")


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


def clean_generated_artifacts(
    artifacts_dir: Path = Path("artifacts"),
    processed_data_dir: Path = Path("data/processed"),
    smoke_artifacts_dir: Path = Path("artifacts_smoke"),
    generated_only: bool = True,
) -> list[str]:
    """Safely remove generated experimental artifacts and processed cache.

    Preserves source code, configs, tests, docs, raw data, and migration reports.

    Args:
        artifacts_dir: Base directory for experimental artifacts.
        processed_data_dir: Directory containing processed data and split indices.
        smoke_artifacts_dir: Directory containing smoke test artifacts.
        generated_only: If True, only targets generated folders.

    Returns:
        List of paths successfully cleaned.
    """
    cleaned_paths: list[str] = []
    artifacts_dir = Path(artifacts_dir)
    processed_data_dir = Path(processed_data_dir)
    smoke_artifacts_dir = Path(smoke_artifacts_dir)

    # Subdirectories inside artifacts/ that contain generated outputs
    target_subdirs = [
        "runs",
        "summary",
        "audits",
        "pipeline_runs",
        "quality",
        "smoke",
        "logs",
    ]

    if artifacts_dir.exists():
        for subdir_name in target_subdirs:
            target_path = artifacts_dir / subdir_name
            if target_path.exists():
                shutil.rmtree(target_path, ignore_errors=True)
                cleaned_paths.append(str(target_path))
                logger.info(f"Cleaned generated artifact directory: {target_path}")

    if processed_data_dir.exists():
        for item in processed_data_dir.iterdir():
            if item.name not in (".gitkeep", "README.md"):
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
                cleaned_paths.append(str(item))
                logger.info(f"Cleaned processed data item: {item}")

    if smoke_artifacts_dir.exists():
        shutil.rmtree(smoke_artifacts_dir, ignore_errors=True)
        cleaned_paths.append(str(smoke_artifacts_dir))
        logger.info(f"Cleaned smoke artifacts directory: {smoke_artifacts_dir}")

    return cleaned_paths
