"""State compatibility gate: block wrong-page actions before execution.

Compares the EXPECTED page/stage (from the active milestone contract)
against the OBSERVED page/stage (from verified state) before an action is
grounded or executed. Deterministic and conservative: unresolvable sides
are permissive (allow + warn), never blocking, because a wrong block is
as harmful as a wrong action. The value gate remains responsible for
values; this gate owns page/stage compatibility only.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def _norm_stage(value: Any) -> str:
    return str(value or "").strip().lower()


def expected_stage(milestone: Dict[str, Any]) -> str:
    """Where the milestone contract says its work happens."""
    milestone = milestone or {}
    for key in ("stage",):
        if _norm_stage(milestone.get(key)):
            return _norm_stage(milestone.get(key))
    pre = milestone.get("pre_state", {}) or {}
    if isinstance(pre, dict):
        for key in ("stage", "page"):
            if _norm_stage(pre.get(key)):
                return _norm_stage(pre.get(key))
    return ""


def observed_stage(state: Dict[str, Any],
                   observation: Optional[Dict[str, Any]] = None) -> str:
    """Where verified evidence says the browser is."""
    state = state or {}
    if _norm_stage(state.get("verified_stage")) not in ("", "unknown"):
        return _norm_stage(state.get("verified_stage"))
    if isinstance(observation, dict):
        for key in ("stage", "page"):
            if _norm_stage(observation.get(key)):
                return _norm_stage(observation.get(key))
    return ""


def check(state: Dict[str, Any], milestone: Dict[str, Any],
          action: Dict[str, Any],
          observation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return {allowed, reason, expected_state, observed_state,
    recommended_recovery}."""
    exp = expected_stage(milestone or {})
    obs = observed_stage(state or {}, observation)
    base = {"expected_state": exp or "(unspecified)",
            "observed_state": obs or "(unknown)"}
    if not exp or not obs:
        return {**base, "allowed": True,
                "reason": "state unresolvable: permissive (no block on "
                          "missing stage evidence)",
                "recommended_recovery": None}
    if exp == obs:
        return {**base, "allowed": True,
                "reason": f"stage compatible ({obs})",
                "recommended_recovery": None}
    action_desc = ""
    if isinstance(action, dict):
        action_desc = (f"{action.get('action', '')} "
                       f"{action.get('target', '')}".strip())
    return {**base, "allowed": False,
            "reason": "STATE_INCOMPATIBLE: milestone "
                      f"{(milestone or {}).get('id', '?')} expects stage "
                      f"{exp!r} but verified state is stage {obs!r}"
                      + (f" for {action_desc!r}" if action_desc else ""),
            "recommended_recovery": "LOCAL_REPLAN"}
