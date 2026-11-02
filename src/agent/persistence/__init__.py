"""Dot-style Task Persistence module for autonomous browser agents."""

from .schema import (
    TaskState,
    TaskStatus,
    ActionPhase,
    HumanApprovalStatus,
    CheckpointEventType,
    CheckpointRecord,
)
from .store import (
    TaskPersistenceStore,
    TaskExistsError,
    TaskNotFoundError,
    IncompatibleSchemaError,
    CorruptedCheckpointError,
    TaskLockedError,
)
from .manager import TaskPersistenceManager, TaskReconciliationError

__all__ = [
    "TaskState",
    "TaskStatus",
    "ActionPhase",
    "HumanApprovalStatus",
    "CheckpointEventType",
    "CheckpointRecord",
    "TaskPersistenceStore",
    "TaskExistsError",
    "TaskNotFoundError",
    "IncompatibleSchemaError",
    "CorruptedCheckpointError",
    "TaskLockedError",
    "TaskPersistenceManager",
    "TaskReconciliationError",
]

