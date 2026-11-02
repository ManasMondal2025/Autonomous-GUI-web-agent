"""Failure taxonomy: classify STV/guardrail failures, map to replan scope.

Pure deterministic functions -- no model, no state. The recovery ladder
(RecoveryPolicy budgets) stays authoritative; scope only selects the
SMALLEST sufficient rung, never a larger one than the policy allows:

  GROUNDING_FAILURE -> ACTION_REPAIR (reground)
  VALUE_FAILURE     -> MICRO_REPLAN (substitute/refuse already ran)
  ACTION_FAILURE    -> ACTION_REPAIR (retry)
  STATE_FAILURE     -> ACTION_REPAIR (wait, then retry)
  WRONG_PAGE        -> MICRO_REPLAN (local replan / navigation recovery)
  CONTRACT_FAILURE  -> MICRO_REPLAN
  MILESTONE_FAILURE -> MILESTONE_REPLAN (else GLOBAL_REPLAN on repeat)
  PLANNING_FAILURE  -> MICRO_REPLAN
  ENVIRONMENT_FAILURE -> ACTION_REPAIR (wait)
  NO_PROGRESS       -> MICRO_REPLAN (escalating)
"""

import logging

logger = logging.getLogger(__name__)

FAILURE_CLASSES = (
    "GROUNDING_FAILURE",
    "VALUE_FAILURE",
    "ACTION_FAILURE",
    "STATE_FAILURE",
    "CONTRACT_FAILURE",
    "MILESTONE_FAILURE",
    "PLANNING_FAILURE",
    "ENVIRONMENT_FAILURE",
    "NO_PROGRESS",
    "WRONG_PAGE",
)

REPLAN_SCOPES = (
    "ACTION_REPAIR",
    "MICRO_REPLAN",
    "MILESTONE_REPLAN",
    "GLOBAL_REPLAN",
)

_SCOPE_FOR_CLASS = {
    "GROUNDING_FAILURE": "ACTION_REPAIR",
    "VALUE_FAILURE": "MICRO_REPLAN",
    "ACTION_FAILURE": "ACTION_REPAIR",
    "STATE_FAILURE": "ACTION_REPAIR",
    "WRONG_PAGE": "MICRO_REPLAN",
    "CONTRACT_FAILURE": "MICRO_REPLAN",
    "MILESTONE_FAILURE": "MILESTONE_REPLAN",
    "PLANNING_FAILURE": "MICRO_REPLAN",
    "ENVIRONMENT_FAILURE": "ACTION_REPAIR",
    "NO_PROGRESS": "MICRO_REPLAN",
}

_DECISION_FALLBACK = {
    "RETRY": "ACTION_FAILURE",
    "REPAIR": "GROUNDING_FAILURE",
    "REPLAN": "CONTRACT_FAILURE",
    "ROLLBACK": "STATE_FAILURE",
    "WAIT": "STATE_FAILURE",
    "PASS": "ACTION_FAILURE",
}


def classify(decision: str = "", reason: str = "",
             value_issue: bool = False,
             grounding_miss: bool = False,
             exec_error: bool = False,
             loading: bool = False,
             repeated: bool = False,
             precondition: bool = False,
             milestone: bool = False,
             planner: bool = False,
             environment: bool = False,
             no_progress: bool = False,
             wrong_page: bool = False) -> str:
    """Deterministic failure class. Explicit signals win, in fixed
    priority; otherwise fall back to the STV decision mapping. Never
    raises; unknown input maps to the decision fallback."""
    _ = reason  # reserved for future reason-text rules; signals decide
    if environment or exec_error and "observe" in str(reason).lower():
        return "ENVIRONMENT_FAILURE"
    if wrong_page:
        return "WRONG_PAGE"
    if value_issue:
        return "VALUE_FAILURE"
    if grounding_miss:
        return "GROUNDING_FAILURE"
    if milestone:
        return "MILESTONE_FAILURE"
    if precondition:
        return "CONTRACT_FAILURE"
    if no_progress or repeated:
        return "NO_PROGRESS"
    if planner:
        return "PLANNING_FAILURE"
    if loading:
        return "STATE_FAILURE"
    if exec_error:
        return "ACTION_FAILURE"
    return _DECISION_FALLBACK.get(str(decision or "").upper(),
                                  "ACTION_FAILURE")


def scope_for(failure_class: str) -> str:
    """Smallest sufficient replan scope for a failure class."""
    return _SCOPE_FOR_CLASS.get(str(failure_class or ""),
                                "MICRO_REPLAN")
