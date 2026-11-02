"""Data models for OpenClaw-style Reusable Agent Skills."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SkillMode(str, Enum):
    AUTO = "auto"
    REUSE_ONLY = "reuse-only"
    DISCOVER = "discover"


class ProcedureStep(BaseModel):
    step_id: int
    intent: str
    action_type: str
    target: str
    value: Optional[str] = None
    expected_transition: Dict[str, Any] = Field(default_factory=dict)
    optional: bool = False


class SkillDefinition(BaseModel):
    name: str
    version: int = 1
    purpose: str
    applicable_task_types: List[str] = Field(default_factory=list)
    inputs: List[str] = Field(default_factory=list)
    preconditions: List[str] = Field(default_factory=list)
    procedure: List[ProcedureStep] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    expected_state_transitions: Dict[str, Any] = Field(default_factory=dict)
    failure_conditions: List[str] = Field(default_factory=list)
    recovery: Dict[str, Any] = Field(default_factory=dict)
    verification_requirements: List[str] = Field(default_factory=list)
    known_limitations: List[str] = Field(default_factory=list)
    evidence: str = ""


class SkillExecutionResult(BaseModel):
    skill_name: str
    skill_version: int
    executed: bool
    success: bool
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list)
    verification_result: Dict[str, Any] = Field(default_factory=dict)
    stv_passed: bool = False
    recovery_attempts: int = 0
    fallback_used: bool = False
    failure_reason: Optional[str] = None
    diagnostics: Dict[str, Any] = Field(default_factory=dict)

