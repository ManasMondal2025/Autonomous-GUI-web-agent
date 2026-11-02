"""Observation Builder: compact, task-relevant context for the planner.

Aggregates existing DOM/accessibility data, URL, visible text, form states,
and recent action outcomes into a structured observation dict. Keeps the
context concise — no full DOM or screenshot history dumps.

Uses ONLY existing observation mechanisms; does not add a new vision model.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Maximum lengths to prevent token bloat
MAX_VISIBLE_TEXT   = 2000
MAX_DOM_ELEMENTS   = 50
MAX_FORM_VALUES    = 20
MAX_RECENT_ACTIONS = 3


def build_observation_context(
        observation: Dict[str, Any],
        task_spec=None,
        verified_state: Optional[Dict[str, Any]] = None,
        recent_actions: Optional[List[Dict[str, Any]]] = None,
        last_verdict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a compact, task-relevant context dict for the planner.

    Args:
        observation: Raw observation from Observer.observe().
        task_spec: Optional TaskSpec (used to filter relevant DOM elements).
        verified_state: TaskMemory.verified_state_summary() output.
        recent_actions: Last N action dicts from the controller history.
        last_verdict: Last STV verdict dict.

    Returns:
        A structured dict that can be JSON-serialised and passed to the
        planner prompt without bloating it.
    """
    url = observation.get("url", "")
    visible_text = _trim(observation.get("visible_text", ""), MAX_VISIBLE_TEXT)
    dom_elements = observation.get("dom_elements", []) or []
    screenshot_bytes = observation.get("screenshot_bytes")  # kept as-is

    # Form state: non-empty input values and checked/selected markers
    form_state = _extract_form_state(dom_elements)

    # Key element summary (trimmed, task-relevant)
    element_summary = _summarize_elements(dom_elements, task_spec)

    # Last action outcome (from recent_actions or last_verdict)
    last_action_summary = _last_action_summary(recent_actions, last_verdict)

    # Page identity signals
    page_identity = _page_identity(url, visible_text)

    ctx: Dict[str, Any] = {
        "url": url,
        "page_identity": page_identity,
        "visible_text_excerpt": visible_text,
        "form_state": form_state,
        "interactive_element_count": len(dom_elements),
        "element_summary": element_summary,
        "last_action": last_action_summary,
        "last_verdict": _trim_verdict(last_verdict),
        "verified_state_summary": verified_state or {},
        # Screenshot bytes are NOT included here — they are passed separately
        # to VLM calls via the existing image= parameter to avoid duplication.
    }
    return ctx


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _trim(text: Any, max_len: int) -> str:
    s = str(text or "").strip()
    if len(s) <= max_len:
        return s
    return s[:max_len - 3] + "..."


def _extract_form_state(dom_elements: List[Dict[str, Any]]) -> Dict[str, str]:
    """Extract filled form values and checked/selected states from DOM."""
    state: Dict[str, str] = {}
    count = 0
    for el in dom_elements or []:
        if count >= MAX_FORM_VALUES:
            break
        if not isinstance(el, dict):
            continue
        value = str(el.get("value", "") or "").strip()
        if value:
            name = (el.get("name") or el.get("ariaLabel") or
                    el.get("placeholder") or el.get("id") or
                    el.get("tag", "field"))
            state[str(name)[:60]] = value[:200]
            count += 1
        if el.get("checked"):
            label = str(
                el.get("text") or el.get("value") or el.get("name") or
                "checkbox").strip()
            state[f"[checked] {label[:50]}"] = "true"
            count += 1
        if el.get("selected"):
            label = str(
                el.get("text") or el.get("value") or el.get("name") or
                "option").strip()
            state[f"[selected] {label[:50]}"] = "true"
            count += 1
    return state


def _summarize_elements(
        dom_elements: List[Dict[str, Any]],
        task_spec=None,
) -> List[Dict[str, str]]:
    """Return a concise list of interactive element descriptions."""
    goal_words: set = set()
    if task_spec is not None:
        goal_words = set(
            getattr(task_spec, "normalized_goal", "").lower().split())

    summaries = []
    for el in (dom_elements or [])[:MAX_DOM_ELEMENTS]:
        if not isinstance(el, dict):
            continue
        tag = str(el.get("tag") or el.get("type") or "element")
        text = str(el.get("text") or el.get("ariaLabel") or
                   el.get("placeholder") or el.get("value") or "").strip()
        role = str(el.get("role") or "").strip()

        # Relevance filter: if we have goal words, prefer relevant elements
        # but always include buttons, inputs, selects
        is_interactive = tag.lower() in {"button", "a", "input", "textarea",
                                         "select"} or role in {
            "button", "link", "textbox", "combobox", "checkbox"}
        text_words = set(text.lower().split())
        is_relevant = bool(goal_words & text_words)

        if is_interactive or is_relevant or not goal_words:
            entry: Dict[str, str] = {"tag": tag}
            if text:
                entry["text"] = text[:100]
            if role:
                entry["role"] = role
            summaries.append(entry)

    return summaries[:MAX_DOM_ELEMENTS]


def _page_identity(url: str, visible_text: str) -> Dict[str, str]:
    """Extract page identity signals for mismatch detection."""
    import re
    lower = visible_text.lower()
    page_type = "unknown"

    # Ordered checks — first match wins
    if any(w in lower for w in ("order confirmed", "thank you for your order",
                                 "order id", "confirmation number")):
        page_type = "confirmation"
    elif any(w in lower for w in ("checkout", "payment", "billing")):
        page_type = "checkout"
    elif any(w in lower for w in ("shopping cart", "your cart", "items in cart")):
        page_type = "cart"
    elif any(w in lower for w in ("search results", "results for")):
        page_type = "search_results"
    elif any(w in lower for w in ("sign in", "log in", "login")):
        page_type = "login"
    elif any(w in lower for w in ("product", "add to cart")):
        page_type = "product"

    # URL-based refinement
    url_lower = url.lower()
    if "cart" in url_lower:
        page_type = "cart"
    elif "checkout" in url_lower:
        page_type = "checkout"
    elif "confirm" in url_lower or "order" in url_lower:
        page_type = "confirmation"
    elif "login" in url_lower or "signin" in url_lower:
        page_type = "login"

    return {"url": url, "type": page_type}


def _last_action_summary(
        recent_actions: Optional[List[Dict[str, Any]]],
        last_verdict: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compact summary of the most recent action and its outcome."""
    summary: Dict[str, Any] = {}
    if recent_actions:
        last = recent_actions[-1]
        if isinstance(last, dict):
            summary["action"] = str(last.get("action", last.get("type", "?")))
            summary["target"] = str(last.get("target", ""))[:100]
            summary["result"] = str(last.get("result", ""))[:200]
    if last_verdict:
        summary["verdict"] = str(last_verdict.get("decision", "?"))
        summary["verdict_reason"] = str(last_verdict.get("reason", ""))[:200]
    return summary


def _trim_verdict(verdict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Compact STV verdict for the planner context."""
    if not verdict:
        return {}
    return {
        "decision": str(verdict.get("decision", "?")),
        "confidence": verdict.get("confidence", 0.0),
        "reason": str(verdict.get("reason", ""))[:200],
        "evidence": (verdict.get("evidence_items", []) or [])[:3],
    }


def is_stale_observation(
        observation: Dict[str, Any],
        previous_observation: Optional[Dict[str, Any]],
) -> bool:
    """Detect when an observation is likely stale (same URL + same text).

    A stale observation must never be reused as post-action evidence.
    """
    if previous_observation is None:
        return False
    same_url = observation.get("url") == previous_observation.get("url")
    same_text = (
        (observation.get("visible_text") or "") ==
        (previous_observation.get("visible_text") or ""))
    return same_url and same_text
