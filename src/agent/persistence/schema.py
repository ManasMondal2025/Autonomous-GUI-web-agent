"""Persistent task-state schema and checkpoint data models."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

SCHEMA_VERSION = 1

SECRET_KEY_PATTERNS = [
    r"pass(word)?",
    r"secret",
    r"token",
    r"cookie",
    r"auth",
    r"api[_-]?key",
    r"credential",
    r"session[_-]?id",
    r"private[_-]?key",
    r"pin",
    r"otp",
]
_SECRET_RE = re.compile("|".join(f"(?:{p})" for p in SECRET_KEY_PATTERNS), re.IGNORECASE)

_URL_SECRET_RE = re.compile(
    r"([?&])(pass(?:word)?|token|key|api[_-]?key|secret|auth|session[_-]?id)=([^&]+)",
    re.IGNORECASE
)
_HEADER_AUTH_RE = re.compile(
    r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/-]+=*",
    re.IGNORECASE
)
_EXCEPTION_SECRET_RE = re.compile(
    r"(?i)\b(?:auth[_-]?)?(password|secret|token|api[_-]?key)\s*[:=]\s*(?:'[^']*'|\"[^\"]*\"|[^\s'\",;&]+)",
    re.IGNORECASE
)


def scrub_string(s: str) -> str:
    """Scrub sensitive query parameters, authorization headers, and credentials from a string."""
    if not isinstance(s, str):
        return s
    res = _URL_SECRET_RE.sub(r"\1\2=[REDACTED]", s)
    res = _HEADER_AUTH_RE.sub(r"\1 [REDACTED]", res)
    res = _EXCEPTION_SECRET_RE.sub(r"\1=[REDACTED]", res)
    return res


def scrub_secrets(data: Any) -> Any:
    """Recursively scrub sensitive keys, URLs, headers, and token patterns from persisted data."""
    if isinstance(data, dict):
        scrubbed = {}
        for k, v in data.items():
            if _SECRET_RE.search(str(k)):
                scrubbed[k] = "[REDACTED]"
            else:
                scrubbed[k] = scrub_secrets(v)
        return scrubbed
    elif isinstance(data, list):
        return [scrub_secrets(item) for item in data]
    elif isinstance(data, str):
        return scrub_string(data)
    return data


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    PAUSED_HITL = "paused_hitl"


class ActionPhase(str, Enum):
    PROPOSED = "proposed"
    EXECUTED = "executed"
    VERIFIED = "verified"
    UNKNOWN = "unknown"


class HumanApprovalStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class CheckpointEventType(str, Enum):
    TASK_INIT = "task_init"
    SUBGOAL_INIT = "subgoal_init"
    ACTION_PROPOSED = "action_proposed"
    ACTION_EXECUTED = "action_executed"
    ACTION_VERIFIED = "action_verified"
    SUBGOAL_COMPLETE = "subgoal_complete"
    RECOVERY_ATTEMPT = "recovery_attempt"
    HITL_PAUSE = "hitl_pause"
    TASK_COMPLETE = "task_complete"
    TASK_FAILED = "task_failed"
    PROCESS_INTERRUPTED = "process_interrupted"


class CheckpointRecord(BaseModel):
    checkpoint_id: str
    task_id: str
    checkpoint_version: int = SCHEMA_VERSION
    event_type: CheckpointEventType
    action_phase: ActionPhase = ActionPhase.UNKNOWN
    step_number: int = 0
    subgoal_id: Optional[str] = None
    subgoal_intent: Optional[str] = None
    action: Optional[Dict[str, Any]] = None
    browser_url: Optional[str] = None
    state_summary: Dict[str, Any] = Field(default_factory=dict)
    verified_facts: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @model_validator(mode="before")
    @classmethod
    def sanitize(cls, values: Any) -> Any:
        if isinstance(values, dict):
            return scrub_secrets(values)
        return values


class TaskState(BaseModel):
    task_id: str
    task_description: str
    status: TaskStatus = TaskStatus.IN_PROGRESS
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    current_subgoal: Optional[Dict[str, Any]] = None
    completed_subgoals: List[Dict[str, Any]] = Field(default_factory=list)
    pending_subgoals: List[Dict[str, Any]] = Field(default_factory=list)
    verified_results: List[Dict[str, Any]] = Field(default_factory=list)
    failed_attempts: List[Dict[str, Any]] = Field(default_factory=list)
    current_url: Optional[str] = None
    browser_session_reference: Optional[str] = None
    checkpoint_version: int = SCHEMA_VERSION
    last_verified_checkpoint: Optional[str] = None
    pending_action: Optional[Dict[str, Any]] = None
    human_approval_status: HumanApprovalStatus = HumanApprovalStatus.NOT_REQUIRED
    final_result: Optional[Dict[str, Any]] = None
    is_locked: bool = False
    locked_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("checkpoint_version")
    @classmethod
    def validate_version(cls, v: int) -> int:
        if v > SCHEMA_VERSION:
            raise ValueError(f"Incompatible checkpoint version: {v} > {SCHEMA_VERSION}")
        return v

    @model_validator(mode="before")
    @classmethod
    def sanitize(cls, values: Any) -> Any:
        if isinstance(values, dict):
            return scrub_secrets(values)
        return values

