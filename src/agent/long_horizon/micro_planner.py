"""Micro-planner: rolling-horizon local transition planner.

Plans at most MAX_ACTIONS (3, prefer 1) from CURRENT VERIFIED STATE toward
the ACTIVE MILESTONE -- never the whole trajectory. Reuses the shared
planner infrastructure (prompt loading, JSON extraction + repair, action
validation, transition normalization) from subgoal_planner; only the
prompt and input assembly are micro-specific.
"""

import logging
from typing import Dict, Any, List, Optional

from .subgoal_planner import (
    PlannerError,
    VALID_ACTIONS,
    _generate_json,
    _load_prompt,
    _normalize_transition,
)

logger = logging.getLogger(__name__)

MAX_ACTIONS = 3
MAX_CHUNK_SIZE = 6


class MicroPlanner:
    """One rolling step: verified state + active milestone -> <=3 actions (or up to 6 in experimental action-chunking mode)."""

    def __init__(self, vlm, horizon: int = 3, action_chunking: bool = False, chunk_size: int = 3):
        self.vlm = vlm
        self.action_chunking = action_chunking
        self.chunk_size = max(2, min(int(chunk_size or 3), MAX_CHUNK_SIZE)) if action_chunking else MAX_ACTIONS
        max_cap = self.chunk_size if action_chunking else MAX_ACTIONS
        self.horizon = max(1, min(horizon, max_cap))

    def plan(self, state: Dict[str, Any],
             milestone: Dict[str, Any],
             ui: Optional[Dict[str, Any]] = None,
             failures: Optional[List[Any]] = None,
             banned_action: Optional[Dict[str, Any]] = None,
             image=None, horizon: Optional[int] = None,
             goal: str = "",
             ) -> List[Dict[str, Any]]:
        """Generate the next micro-plan. Raises PlannerError when unusable."""
        import json as _json
        max_cap = self.chunk_size if self.action_chunking else MAX_ACTIONS
        h = max(1, min(horizon or self.horizon, max_cap))
        state = state or {}
        milestone = milestone or {}
        prompt = _load_prompt("micro_planner.txt").format(
            horizon=h,
            goal=str(goal or "")[:800],
            verified_state=_json.dumps(state, indent=1, default=str)[:1500],
            milestone_block=_render_milestone(milestone),
            ui_block=_render_ui(ui),
            failure_block=_render_failures(failures),
            banned_block=_render_banned(banned_action),
        )
        if self.action_chunking:
            prompt += (
                f"\n\nEXPERIMENTAL ACTION CHUNKING (target chunk size: {h}):\n"
                f"- Produce 2 to {h} predictable consecutive actions that can safely execute together.\n"
                "- Prefer short, predictable chunks over long, speculative action sequences.\n"
                "- Stop the chunk before navigation, modal popups, submit buttons, or uncertain transitions.\n"
                "- Prefer 1 action when the next step depends on observing the result of this action.\n"
                "- Use the existing action schema."
            )
        data, _ = _generate_json(self.vlm, image, prompt, ["actions"])
        raw = data.get("actions", [])
        if not isinstance(raw, list) or not raw:
            raise PlannerError("micro planner returned no actions")
        out = []
        for a in raw[:h]:
            if not isinstance(a, dict):
                continue
            atype = str(a.get("action", "")).lower()
            if atype not in VALID_ACTIONS:
                logger.warning(f"Micro planner emitted invalid action "
                               f"{atype!r}; skipping")
                continue
            out.append({
                "action": atype,
                "target": str(a.get("target", ""))[:200],
                "value": a.get("value"),
                "expected_transition": _normalize_transition(
                    a.get("expected_transition"), atype),
            })
        if not out:
            raise PlannerError("micro planner returned no valid actions")
        return out


def _render_milestone(milestone: Dict[str, Any]) -> str:
    lines = [f"ACTIVE MILESTONE {milestone.get('id', '?')}: "
             f"{milestone.get('intent', '')}"[:200]]
    if milestone.get("target"):
        lines.append(f"Target: {milestone['target']}"[:200])
    params = milestone.get("parameters", {}) or {}
    if params:
        lines.append("Parameters (EXACT literals, never invent):")
        for k, v in list(params.items())[:12]:
            lines.append(f"  {k} = {v}"[:160])
    for key, label in (("pre_state", "Required pre-state (must hold now):"),
                       ("post_state", "Intended post-state (verify after):")):
        val = milestone.get(key, {}) or {}
        if isinstance(val, dict) and val:
            lines.append(f"{label}")
            for k, v in list(val.items())[:8]:
                lines.append(f"  {k} = {v}"[:160])
    req = milestone.get("required_facts", []) or []
    if req:
        lines.append("Required facts:")
        lines.extend(f"  - {r}"[:160] for r in req[:12])
    lits = milestone.get("literals", []) or []
    if lits:
        lines.append("Known literals (EXACT values for this milestone):")
        lines.extend(f"  - {x}"[:160] for x in lits[:12])
    try:
        from .progress import outstanding_literals
        facts = milestone.get("verified_facts", {}) or {}
        ftexts = set()
        if isinstance(facts, dict):
            for k, v in facts.items():
                ftexts.add(str(k).lower())
                if v not in (None, True):
                    ftexts.add(str(v).lower())
        missing = outstanding_literals(lits, ftexts)
    except Exception:
        missing = []
    if missing:
        lines.append("Outstanding (required, NOT yet verified -- do these):")
        lines.extend(f"  - fill: {x}"[:160] for x in missing[:12])
    terms = milestone.get("terminal_predicates", []) or []
    if terms:
        lines.append("Terminal predicates (ALL must hold when done):")
        for p in terms[:12]:
            if isinstance(p, dict) and p.get("predicate"):
                lines.append(
                    f"  - {p.get('target', '?')} {p.get('predicate')} "
                    f"{p.get('value', '')}"[:160])
            elif p:
                lines.append(f"  - {p}"[:160])
    else:
        done = str(milestone.get("terminal_state", "") or "")
        if done:
            lines.append(f"Terminal state: {done}"[:200])
    return "\n".join(lines)[:1500]


def _render_ui(ui: Optional[Dict[str, Any]]) -> str:
    ui = ui or {}
    parts = []
    if ui.get("url"):
        parts.append(f"URL: {ui['url']}"[:200])
    if ui.get("page_text"):
        parts.append("Page text (truncated):\n"
                     + str(ui["page_text"])[:1200])
    if ui.get("element_count") is not None:
        parts.append(f"Interactable elements: {ui['element_count']}")
    return "\n".join(parts)[:1500] if parts else "(no UI snapshot)"


def _render_failures(failures: Optional[List[Any]]) -> str:
    if not failures:
        return "(none -- first attempt from this state)"
    lines = ["RECENT STV FAILURE TRACE (repair from this, do not repeat):"]
    for f in (failures or [])[-2:]:
        if isinstance(f, dict):
            lines.append(
                f"- [{f.get('failure_class', f.get('scope', '?'))}] "
                f"{f.get('reason', '')}"[:300])
            trace = f.get("trace", {}) if isinstance(
                f.get("trace"), dict) else {}
            for key in ("expected_state", "observed_state", "failed_action"):
                if trace.get(key):
                    lines.append(f"  {key}: {trace[key]}"[:200])
        else:
            lines.append(f"- {f}"[:300])
    return "\n".join(lines)[:1200]


def _render_banned(banned_action: Optional[Dict[str, Any]]) -> str:
    if not banned_action:
        return ""
    return ("BANNED ACTION (failed repeatedly -- output something "
            "meaningfully different, never this again):\n"
            f"- {banned_action.get('action', '?')} "
            f"{banned_action.get('target', '')} "
            f"{banned_action.get('value', '')}"[:300])
