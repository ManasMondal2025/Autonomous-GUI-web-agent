"""Experimental intent gate: controller-level information-sufficiency check.

EXPERIMENTAL ONLY. Disabled by default via::

    ENABLE_EXPERIMENTAL_INTENT_GATE=1

When disabled (default), the agent behaves exactly as before: the gate is
never consulted and no code path changes. When enabled, the baseline
controller routes potentially information-requiring actions (`type`, and
clicks on selection controls) through a TEXT-ONLY judge (no screenshot)
before browser execution. ASK verdicts reuse the existing HITL machinery
(`hitl.resolve_ask_user`); every failure mode falls back to ALLOW.

The gate is deliberately generic: no login-specific rules. It asks whether
THE PROPOSED ACTION needs a parameter the user never provided and that
cannot be safely inferred -- never merely because a dropdown exists.
"""

import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

FLAG_ENV = "ENABLE_EXPERIMENTAL_INTENT_GATE"

# Tags / roles that usually carry a user-chosen value (a parameter).
INFO_TAGS = {"select", "input", "textarea", "datalist"}
INFO_ROLES = {
    "combobox", "listbox", "textbox", "searchbox", "spinbutton",
    "radiogroup", "checkbox",
}
# Input types that take a user value. (Generic, not login-specific: any
# missing value blocks the action the same way.)
INFO_INPUT_TYPES = {
    "text", "search", "date", "datetime-local", "month", "week", "time",
    "email", "tel", "password", "number", "url",
}
# Click targets that are almost always pure navigation, never parameters.
NAV_TEXT_RE = re.compile(
    r"^(home|back|forward|menu|students?|dashboard|next|previous|prev|"
    r"continue|close|cancel|login page|sign in page|open .*page)$",
    re.IGNORECASE,
)

GATE_SYSTEM_PROMPT = """You are an information-sufficiency validator for a web agent.
The web agent is about to execute an action.
Your job is NOT to decide where to click.
Your job is ONLY to determine whether the information required for this specific action is sufficiently specified by:
1. the user's current task,
2. previous conversation/history,
3. information already supplied by the user,
4. unambiguous information available from the UI.
If the action requires a parameter that is missing and cannot be safely inferred, the agent must ask the user.
If the required information is available, allow the action.
The mere existence of a dropdown/input does NOT mean information is missing.
Return ONLY a single JSON object with keys "decision" ("ALLOW" or "ASK"),
"missing_field" (the missing parameter name, or empty string), and "question"
(the exact question to ask the human, or empty string)."""


def is_intent_gate_enabled() -> bool:
    """Feature flag. Default OFF: unset/anything-but-truthy -> False."""
    return os.getenv(FLAG_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def describe_element(el: Dict[str, Any]) -> str:
    """One-line semantic summary of a DOM element for the text judge."""
    if not isinstance(el, dict):
        return "?"
    label = (el.get("text") or el.get("ariaLabel")
             or el.get("placeholder") or el.get("title")
             or el.get("name") or "").strip()[:60]
    return (f"id={el.get('id')} tag={el.get('tag', '?')} "
            f"type={el.get('type', '')} role={el.get('role', '')} "
            f"label={label!r}")


def is_information_control(el: Optional[Dict[str, Any]]) -> bool:
    """True if the element semantically carries a user-chosen value."""
    if not isinstance(el, dict):
        return False
    tag = str(el.get("tag", "")).lower()
    role = str(el.get("role", "")).lower()
    if tag in INFO_TAGS and not (
            tag == "input" and str(el.get("type", "")).lower() == "hidden"):
        if tag == "input" and str(el.get("type", "")).lower() not in INFO_INPUT_TYPES \
                and str(el.get("type", "")).lower() not in ("",):
            # e.g. checkbox/radio/button/submit inputs: value selection only
            # for text-like types; checkbox/radio groups still need a choice.
            if str(el.get("type", "")).lower() not in ("checkbox", "radio"):
                return False
        return True
    return role in INFO_ROLES


def find_action_target(
    action: Any,
    action_dict: Optional[Dict[str, Any]],
    dom_elements: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Resolve the proposed action to its DOM element, if any."""
    dom_elements = dom_elements or []
    if isinstance(action, dict) and action_dict is None:
        action_dict = action
    tid = getattr(action, "target_id", None)
    if tid is None and isinstance(action_dict, dict):
        tid = action_dict.get("target_id")
    if tid is not None:
        for el in dom_elements:
            if isinstance(el, dict) and str(el.get("id")) == str(tid):
                return el
        return None
    # Coordinate-style actions: match by point-in-box (smallest wins).
    x = getattr(action, "x", None)
    y = getattr(action, "y", None)
    if x is None and isinstance(action_dict, dict):
        x, y = action_dict.get("x"), action_dict.get("y")
    if x is None or y is None:
        return None
    best, best_area = None, None
    for el in dom_elements:
        if not isinstance(el, dict):
            continue
        bb = el.get("bbox") or {}
        try:
            ex, ey, ew, eh = (float(bb.get("x", 0)), float(bb.get("y", 0)),
                              float(bb.get("width", 0)), float(bb.get("height", 0)))
        except (TypeError, ValueError):
            continue
        if ew <= 0 or eh <= 0 or not (ex <= x < ex + ew and ey <= y < ey + eh):
            continue
        area = ew * eh
        if best is None or area < best_area:
            best, best_area = el, area
    return best


def should_consult_gate(
    action: Any,
    action_dict: Optional[Dict[str, Any]],
    dom_elements: List[Dict[str, Any]],
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Decide whether the gate should judge this action.

    Returns (consult, reason, target). Only `type` and clicks on
    information-selection controls are gated; navigation clicks and
    non-state actions bypass (fail-closed toward ALLOW).
    """
    raw = getattr(action, "action_type", None)
    if raw is None and isinstance(action, dict):
        # Callers may pass the raw action dict as `action`.
        action_dict = action if action_dict is None else action_dict
        raw = action.get("action_type", action.get("action"))
    at = str(raw or "").lower()
    if isinstance(action_dict, dict):
        at = str(action_dict.get("action_type",
                                 action_dict.get("action", at))).lower()
    if at in ("", "done", "fail", "ask_user", "wait", "scroll"):
        return False, f"non_information_action:{at or '?'}", None
    target = find_action_target(action, action_dict, dom_elements)
    if at == "type":
        return True, "type_action_needs_value", target
    if at == "click":
        if target is not None and is_information_control(target):
            return False if _looks_like_navigation(target) else True, \
                ("navigation_control" if _looks_like_navigation(target)
                 else "information_selection_control"), target
        return False, "non_information_control", target
    return False, f"unknown_action:{at}", target


def _looks_like_navigation(el: Dict[str, Any]) -> bool:
    """Heuristic: link-ish controls with nav-like labels are navigation."""
    tag = str(el.get("tag", "")).lower()
    role = str(el.get("role", "")).lower()
    if tag == "a" or role in ("link", "menuitem", "tab", "menuitemradio"):
        return True
    label = (el.get("text") or el.get("ariaLabel") or "").strip()
    return bool(label) and bool(NAV_TEXT_RE.match(label))


def build_dom_context(
    target: Optional[Dict[str, Any]],
    dom_elements: List[Dict[str, Any]],
    max_controls: int = 12,
) -> str:
    """Semantic UI summary: target first, then sibling value controls."""
    lines = []
    if target is not None:
        lines.append(f"Proposed target: {describe_element(target)}")
        cur = str(target.get("value", "") or "").strip()[:60]
        if cur:
            lines.append(f"Target current value: {cur!r}")
    others = [e for e in (dom_elements or [])
              if isinstance(e, dict) and e is not target
              and is_information_control(e)][:max_controls]
    if others:
        lines.append("Other value controls on page:")
        lines.extend(f"  - {describe_element(e)}" for e in others)
    if not lines:
        lines.append("(no value controls identified in DOM)")
    return "\n".join(lines)


def format_history(history: Any, limit: int = 6) -> str:
    """Compact history text incl. prior human answers (stored in reasons)."""
    if not history:
        return "(no prior steps)"
    lines = []
    for h in list(history)[-limit:]:
        if not isinstance(h, dict):
            continue
        lines.append(
            f"Step {h.get('step', '?')}: {h.get('action', '?')} "
            f"target={h.get('target_id', '?')} -> {h.get('result', '?')} "
            f"[{str(h.get('reason', ''))[:160]}]")
    return "\n".join(lines) if lines else "(no prior steps)"


def fallback_question(task_goal: str, missing_field: str) -> str:
    field = (missing_field or "the required information").strip() or \
        "the required information"
    goal = (task_goal or "the task").strip()[:120]
    return (f"To proceed with '{goal}', I need '{field}'. "
            f"Please provide it.")


class IntentGate:
    """Text-only information-sufficiency judge (experimental).

    `text_llm` is any object with `generate(image, prompt) -> str` (the
    existing VLM interface); text queries use a small blank image because
    the repo has no text-only model abstraction. Pass None to run in
    fail-safe mode (every verdict ALLOW) -- useful for flag-on plumbing
    tests without loading weights.
    """

    def __init__(
        self,
        text_llm: Any = None,
        blank_image_size: tuple = (32, 32),
        max_question_chars: int = 300,
    ):
        self.text_llm = text_llm
        self.blank_image_size = blank_image_size
        self.max_question_chars = max_question_chars

    @property
    def has_judge(self) -> bool:
        return self.text_llm is not None

    def _query(self, prompt: str) -> str:
        from PIL import Image as _Image
        blank = _Image.new("RGB", self.blank_image_size, color="white")
        return self.text_llm.generate(blank, prompt)

    def validate_action_intent(
        self,
        task_goal: str,
        history: Any,
        action: Any,
        dom_context: str,
    ) -> Dict[str, Any]:
        """Return {decision, missing_field, question, reason, latency_s}."""
        t0 = time.time()
        if self.text_llm is None:
            logger.info("[INTENT-GATE] decision=ALLOW reason=no_text_judge")
            return {"decision": "ALLOW", "missing_field": "",
                    "question": "",
                    "reason": "no text judge configured (fail-safe)",
                    "latency_s": round(time.time() - t0, 3)}
        action_desc = self._describe_action(action)
        prompt = (f"{GATE_SYSTEM_PROMPT}\n\nUSER TASK:\n{task_goal}\n\n"
                  f"HISTORY:\n{format_history(history)}\n\n"
                  f"PROPOSED ACTION:\n{action_desc}\n\n"
                  f"UI/DOM CONTEXT:\n{dom_context}\n\nQuestion:\nDoes this "
                  f"specific action require information that the user has not "
                  f"provided and that cannot be safely inferred? Return ONLY "
                  f"the JSON object.")
        try:
            raw = self._query(prompt)
        except Exception as e:
            logger.warning("[INTENT-GATE] ERROR judge crashed: %s", e)
            logger.info("[INTENT-GATE] fallback=ALLOW")
            return {"decision": "ALLOW", "missing_field": "",
                    "question": "", "reason": f"judge error: {e}",
                    "latency_s": round(time.time() - t0, 3)}
        verdict = self._parse_verdict(raw, task_goal)
        verdict["latency_s"] = round(time.time() - t0, 3)
        logger.info("[INTENT-GATE] decision=%s missing_field=%s",
                    verdict["decision"], verdict.get("missing_field", ""))
        return verdict

    @staticmethod
    def _describe_action(action: Any) -> str:
        at = getattr(action, "action_type", "?")
        parts = [f"action={at}"]
        for k in ("target_id", "x", "y", "value", "question", "reason"):
            v = getattr(action, k, None)
            if v is not None:
                parts.append(f"{k}={str(v)[:120]}")
        return ", ".join(parts)

    def _parse_verdict(self, raw: str, task_goal: str) -> Dict[str, Any]:
        import re as _re
        m = _re.search(r"\{.*\}", raw or "", _re.DOTALL)
        if not m:
            logger.warning("[INTENT-GATE] ERROR unparseable judge output: %r",
                            (raw or "")[:160])
            logger.info("[INTENT-GATE] fallback=ALLOW")
            return {"decision": "ALLOW", "missing_field": "",
                    "question": "", "reason": "unparseable judge output"}
        try:
            data = json.loads(m.group(0))
        except Exception as e:
            logger.warning("[INTENT-GATE] ERROR bad judge JSON: %s", e)
            logger.info("[INTENT-GATE] fallback=ALLOW")
            return {"decision": "ALLOW", "missing_field": "",
                    "question": "", "reason": f"bad judge JSON: {e}"}
        decision = str(data.get("decision", "")).strip().upper()
        if decision not in ("ALLOW", "ASK"):
            logger.warning("[INTENT-GATE] ERROR unknown decision %r",
                            data.get("decision"))
            logger.info("[INTENT-GATE] fallback=ALLOW")
            return {"decision": "ALLOW", "missing_field": "",
                    "question": "",
                    "reason": f"unknown decision: {data.get('decision')!r}"}
        missing = str(data.get("missing_field", "") or "").strip()[:120]
        question = str(data.get("question", "") or "").strip()
        if decision == "ASK":
            if not question:
                question = fallback_question(task_goal, missing)
            question = question[:self.max_question_chars]
        else:
            question = ""
        return {"decision": decision, "missing_field": missing,
                "question": question, "reason": "judge verdict"}
