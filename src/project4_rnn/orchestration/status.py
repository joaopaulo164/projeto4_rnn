"""Execution status enumerations and step outcome tracking."""

from enum import StrEnum
from typing import Final


class ExecutionStatus(StrEnum):
    """Lifecycle states for pipeline steps and experiment runs."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIP = "SKIP"
    RERUN = "RERUN"
    INVALIDATED = "INVALIDATED"


STATUS_PENDING: Final[str] = ExecutionStatus.PENDING.value
STATUS_RUNNING: Final[str] = ExecutionStatus.RUNNING.value
STATUS_COMPLETED: Final[str] = ExecutionStatus.COMPLETED.value
STATUS_FAILED: Final[str] = ExecutionStatus.FAILED.value
STATUS_SKIP: Final[str] = ExecutionStatus.SKIP.value
STATUS_RERUN: Final[str] = ExecutionStatus.RERUN.value
STATUS_INVALIDATED: Final[str] = ExecutionStatus.INVALIDATED.value
