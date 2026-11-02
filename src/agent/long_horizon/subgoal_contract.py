"""SubgoalContract schema + pre-execution validation gate.

Phase 2+3 of the upgrade. A SubgoalContract is the typed output of the
State-Aware Single-Subgoal Planner. It is validated deterministically before
any execution happens — the LLM cannot bypass this gate by returning
well-formed JSON that violates safety invariants.

Design rules:
  - One subgoal at a time, generated from the latest verified state.
  - No fabricated coordinates — GUI-Actor remains the visual grounder.
  - Postconditions must be testable by the existing STV.
  - Finite retry/replan budgets are declared in the contract.
  - Validator rejects or repairs contracts (bounded retries) before execution.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

SUPPORTED_ACTION_INTENTS = frozenset({
    "click", "type", "scroll", "wait", "press",
    "navigate", "select", "check", "uncheck",
})

MAX_RETRIES_DEFAULT    = 2
MAX_REPLANS_DEFAULT    = 2
MAX_ACTIONS_DEFAULT    = 10


# ──────────────────────────────────────────────────────────────────────────────
# Sub-models
# ──────────────────────────────────────────────────────────────────────────────

class Postcondition(BaseModel):
    """An observable predicate that must hold after the subgoal succeeds."""
    predicate: str   # vocabulary: EQUALS, CONTAINS, VISIBLE, PAGE_IS, …
    target: str = ""
    value: str = ""
    required: bool = True

    @field_validator("predicate")
    @classmethod
    def predicate_known(cls, v: str) -> str:
        allowed = {"EQUALS", "CONTAINS", "NOT_CONTAINS", "VISIBLE", "EXISTS",
                   "PAGE_IS", "SELECTED", "APPLIED", "CHANGED",
                   "GREATER_THAN", "LESS_THAN"}
        upper = v.strip().upper()
        if upper not in allowed:
            raise ValueError(f"Unknown predicate {v!r}")
        return upper


class RecoveryHint(BaseModel):
    failure_class: str = ""    # e.g. WRONG_PAGE, ACTION_FAILURE, PLANNING_FAILURE
    suggested_action: str = "" # e.g. "navigate back", "replan", "reground"
    can_retry: bool = True


class RetryPolicy(BaseModel):
    max_action_retries: int = Field(default=MAX_RETRIES_DEFAULT, ge=0, le=10)
    max_replans: int        = Field(default=MAX_REPLANS_DEFAULT,  ge=0, le=5)
    max_actions_budget: int = Field(default=MAX_ACTIONS_DEFAULT,  ge=1, le=60)


# ──────────────────────────────────────────────────────────────────────────────
# Primary contract
# ──────────────────────────────────────────────────────────────────────────────

class SubgoalContract(BaseModel):
    """Typed output of one planner call.

    The planner emits semantic intent; GUI-Actor resolves coordinates;
    Playwright executes; STV verifies. The contract links all three.
    """
    subgoal_id: str
    objective: str = Field(description="One-sentence goal for this subgoal")

    # Relevant TaskSpec fields (carried for traceability)
    task_constraint_keys: List[str] = Field(
        default_factory=list,
        description="Keys from TaskSpec.constraints that apply here")

    # Preconditions — verified or explicitly unresolved
    preconditions: List[str] = Field(default_factory=list)
    preconditions_verified: bool = Field(
        default=False,
        description="True only when the controller has verified all preconditions")

    # Action
    action_intent: str = Field(
        description="Semantic action (click, type, scroll …) — NO coordinates")
    action_target_description: str = Field(
        default="",
        description="Natural-language description of the target element for GUI-Actor")
    action_value: Optional[str] = Field(
        default=None,
        description="Value for type/select actions")

    # Expected postconditions
    postconditions: List[Postcondition] = Field(default_factory=list)

    # Evidence
    completion_evidence: List[str] = Field(
        default_factory=list,
        description="What STV must observe to commit this subgoal as complete")

    # Safety
    allowed_side_effects: List[str] = Field(default_factory=list)
    requires_confirmation: bool = False

    # Recovery
    recovery_hint: Optional[RecoveryHint] = None
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)

    model_config = {"use_enum_values": True}

    @field_validator("action_intent")
    @classmethod
    def intent_supported(cls, v: str) -> str:
        lower = v.strip().lower()
        # Allow compound descriptions like "click the submit button"
        first_word = lower.split()[0] if lower.split() else lower
        if first_word not in SUPPORTED_ACTION_INTENTS:
            # Soft: warn but don't reject — GUI-Actor may handle it
            logger.warning(f"SubgoalContract: action_intent first word "
                           f"{first_word!r} not in known set "
                           f"{sorted(SUPPORTED_ACTION_INTENTS)}")
        return lower

    @field_validator("objective")
    @classmethod
    def objective_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("objective must not be empty")
        return v.strip()

    @model_validator(mode="after")
    def postconditions_must_be_testable(self) -> "SubgoalContract":
        required = [p for p in self.postconditions if p.required]
        for p in required:
            if not p.target and not p.value:
                raise ValueError(
                    f"Required postcondition {p.predicate!r} has neither "
                    f"target nor value — STV cannot verify it")
        return self

    @model_validator(mode="after")
    def no_coordinates_in_contract(self) -> "SubgoalContract":
        """GUI-Actor, not the planner, is responsible for coordinates."""
        coord_indicators = ("x=", "y=", "coordinate", "pixel", "px")
        combined = f"{self.action_intent} {self.action_target_description}".lower()
        if any(ind in combined for ind in coord_indicators):
            raise ValueError(
                "SubgoalContract must not contain pixel coordinates — "
                "GUI-Actor handles grounding")
        return self


# ──────────────────────────────────────────────────────────────────────────────
# Contract validator (deterministic pre-execution gate)
# ──────────────────────────────────────────────────────────────────────────────

class ContractValidationError(Exception):
    """Raised when a contract fails the pre-execution gate."""

    def __init__(self, violations: List[str]) -> None:
        self.violations = violations
        super().__init__("; ".join(violations))


def validate_contract_pre_execution(
        contract: SubgoalContract,
        task_spec=None,
        verified_facts: Optional[Dict[str, Any]] = None,
        current_url: str = "",
) -> List[str]:
    """Deterministic pre-execution validation.

    Returns a list of violation strings. Empty list = contract is safe to
    execute. The LLM cannot cause an empty list by returning crafted JSON;
    every check here is deterministic Python logic.
    """
    violations: List[str] = []
    verified_facts = verified_facts or {}

    # 1. Postconditions must exist for required subgoals
    required_posts = [p for p in contract.postconditions if p.required]
    if not required_posts:
        violations.append(
            f"subgoal {contract.subgoal_id!r}: no required postconditions — "
            "STV cannot verify completion")

    # 2. Action intent must be in the supported set (first word check)
    first = contract.action_intent.split()[0] if contract.action_intent else ""
    if first not in SUPPORTED_ACTION_INTENTS:
        violations.append(
            f"action_intent first word {first!r} is not supported by the "
            f"existing executor; supported: {sorted(SUPPORTED_ACTION_INTENTS)}")

    # 3. Type/check actions must have a value or description
    if first in ("type", "select") and not contract.action_value:
        violations.append(
            f"action {first!r} has no action_value — executor cannot proceed")

    # 4. TaskSpec constraint traceability
    if task_spec is not None:
        spec_keys = {c.key for c in getattr(task_spec, "constraints", [])}
        for key in contract.task_constraint_keys:
            if spec_keys and key not in spec_keys:
                violations.append(
                    f"contract references constraint {key!r} not in TaskSpec")

    # 5. Consequential actions must have confirmation flag
    consequential = {"checkout", "pay", "purchase", "submit", "confirm",
                     "delete", "cancel", "reset"}
    if any(c in contract.action_intent.lower() for c in consequential):
        if not contract.requires_confirmation:
            violations.append(
                f"consequential action in {contract.action_intent!r} must "
                "set requires_confirmation=True")

    # 6. Completion evidence must be non-empty
    if not contract.completion_evidence:
        violations.append(
            f"subgoal {contract.subgoal_id!r}: completion_evidence is empty — "
            "cannot commit subgoal without evidence specification")

    return violations


def repair_contract(
        contract: SubgoalContract,
        violations: List[str],
) -> SubgoalContract:
    """Best-effort deterministic repair for known violation patterns.

    Never invents postconditions or evidence — only fills safe defaults.
    Returns the original contract when repair is not possible (caller
    rejects or escalates).
    """
    import copy as _copy
    data = _copy.deepcopy(contract.model_dump())
    changed = False

    # Repair: missing completion_evidence → derive from postconditions
    if not data.get("completion_evidence"):
        evidence = []
        for p in data.get("postconditions", []):
            if isinstance(p, dict):
                evidence.append(
                    f"{p.get('predicate', 'CHECK')} "
                    f"{p.get('target', '')} {p.get('value', '')}".strip())
        if evidence:
            data["completion_evidence"] = evidence
            changed = True

    # Repair: requires_confirmation missing for consequential action
    consequential = {"checkout", "pay", "purchase", "submit", "confirm",
                     "delete", "cancel", "reset"}
    intent = data.get("action_intent", "").lower()
    if any(c in intent for c in consequential) and not data.get("requires_confirmation"):
        data["requires_confirmation"] = True
        changed = True

    if not changed:
        return contract

    try:
        return SubgoalContract(**data)
    except Exception as e:
        logger.warning(f"Contract repair failed: {e}")
        return contract
