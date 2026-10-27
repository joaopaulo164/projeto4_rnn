"""Artifact verification, cleanup, and file existence audit routines."""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger("project4_rnn.audit.artifacts")

# Directories inside artifacts/ that hold generated experimental outputs and are removed by
# clean-artifacts. Everything else (notably artifacts/corrections and artifacts/quality) is kept.
GENERATED_ARTIFACT_SUBDIRS: tuple[str, ...] = (
    "runs",
    "summary",
    "audits",
    "pipeline_runs",
    "smoke",
    "logs",
)
GENERATED_ARTIFACT_FILES: tuple[str, ...] = ("environment.json", "OBSOLETE_ARTIFACTS.json")
PROTECTED_ARTIFACT_SUBDIRS: tuple[str, ...] = ("corrections", "quality")
PRESERVED_PROCESSED_NAMES: tuple[str, ...] = (".gitkeep", "README.md")
PROJECT_ROOT_MARKERS: tuple[str, ...] = ("pyproject.toml", "src", ".git")


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
    except (OSError, json.JSONDecodeError) as e:
        return False, f"Invalid JSON syntax in {file_path}: {e}", None


def _assert_safe_target(path: Path, label: str) -> None:
    """Refuse to clean a directory that looks like a project root."""
    for marker in PROJECT_ROOT_MARKERS:
        if (path / marker).exists():
            raise ValueError(
                f"Refusing to clean {label} '{path}': it contains '{marker}' and looks like a "
                "project root, not an artifacts directory."
            )


def plan_clean_generated_artifacts(
    artifacts_dir: Path = Path("artifacts"),
    processed_data_dir: Path = Path("data/processed"),
    smoke_artifacts_dir: Path = Path("artifacts_smoke"),
) -> list[Path]:
    """List the paths clean_generated_artifacts would remove (nothing is deleted)."""
    artifacts_dir = Path(artifacts_dir)
    processed_data_dir = Path(processed_data_dir)
    smoke_artifacts_dir = Path(smoke_artifacts_dir)
    _assert_safe_target(artifacts_dir, "artifacts directory")
    _assert_safe_target(processed_data_dir, "processed data directory")

    targets: list[Path] = []
    if artifacts_dir.is_dir():
        for name in GENERATED_ARTIFACT_SUBDIRS:
            candidate = artifacts_dir / name
            if candidate.exists():
                targets.append(candidate)
        for name in GENERATED_ARTIFACT_FILES:
            candidate = artifacts_dir / name
            if candidate.exists():
                targets.append(candidate)
    if processed_data_dir.is_dir():
        targets.extend(
            sorted(
                item
                for item in processed_data_dir.iterdir()
                if item.name not in PRESERVED_PROCESSED_NAMES
            )
        )
    if smoke_artifacts_dir.exists() and smoke_artifacts_dir.resolve() != artifacts_dir.resolve():
        _assert_safe_target(smoke_artifacts_dir, "smoke artifacts directory")
        targets.append(smoke_artifacts_dir)
    return targets


def clean_generated_artifacts(
    artifacts_dir: Path = Path("artifacts"),
    processed_data_dir: Path = Path("data/processed"),
    smoke_artifacts_dir: Path = Path("artifacts_smoke"),
    dry_run: bool = False,
) -> list[str]:
    """Remove generated experimental artifacts and the processed-data cache.

    Only the directories listed in ``GENERATED_ARTIFACT_SUBDIRS``, the generated files in
    ``GENERATED_ARTIFACT_FILES``, the contents of ``processed_data_dir`` (except ``.gitkeep`` and
    ``README.md``) and the smoke directory are removed. Source code, configs, tests, docs, raw data,
    ``artifacts/corrections`` and ``artifacts/quality`` are never touched.

    Args:
        artifacts_dir: Base directory for experimental artifacts.
        processed_data_dir: Directory containing processed data and split indices.
        smoke_artifacts_dir: Directory containing smoke test artifacts.
        dry_run: If True, only report what would be removed.

    Returns:
        Paths removed (or that would be removed when ``dry_run``).

    Raises:
        ValueError: If a target directory looks like a project root.
    """
    targets = plan_clean_generated_artifacts(artifacts_dir, processed_data_dir, smoke_artifacts_dir)
    if dry_run:
        return [str(t) for t in targets]

    cleaned: list[str] = []
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)
        cleaned.append(str(target))
        logger.info(f"Cleaned generated item: {target}")
    return cleaned
