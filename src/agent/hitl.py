"""Human-in-the-loop helper: ask_user handling shared by controllers.

The `ask_user` action never touches the browser. Both the baseline
AgentController and the LongHorizonController funnel through `resolve_ask_user`
so prompting, memory logging, and STV-bypass semantics cannot drift.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

AskHandler = Callable[[str], str]

MAX_ASKS_DEFAULT = 3

TAKEOVER_TIMEOUT_S_DEFAULT = 120.0
TAKEOVER_POLL_S_DEFAULT = 2.0

# Route settings for per-question HITL transport selection.
ASK_ROUTE_TERMINAL = "terminal"
ASK_ROUTE_TAKEOVER = "takeover"
ASK_ROUTE_AUTO = "auto"

# Credential-shaped asks are routed to noVNC takeover in auto mode so
# secrets never touch stdin/logs. Kept to generic secret/login concepts --
# no site-specific strings.
CREDENTIAL_RE = None  # compiled lazily (see _credential_re())


def _credential_re():
    global CREDENTIAL_RE
    if CREDENTIAL_RE is None:
        import re as _re
        CREDENTIAL_RE = _re.compile(
            r"(passw(or)?d|passwd|pwd|passcode|credential|secret|api[\s_-]?key|"
            r"auth[\s_-]?token|otp|\b2fa\b|one[\s_-]?time[\s_-]?(code|password)|"
            r"pin\s*code|login\s+(as|with|using).*(password|credential))",
            _re.IGNORECASE)
    return CREDENTIAL_RE


def is_credential_ask(question: Any, context: Any = "") -> bool:
    """True if the ask is credential-shaped (passwords, tokens, OTP...).

    Matches on the question plus optional context (e.g. the action's reason
    or missing-field note). Empty/whitespace input never matches.
    """
    text = f"{question or ''}\n{context or ''}".strip()
    if not text:
        return False
    try:
        return bool(_credential_re().search(text))
    except Exception:
        return False


def route_ask_transport(question: Any, context: Any = "",
                        route: str = ASK_ROUTE_TERMINAL) -> str:
    """Choose 'takeover' or 'terminal' for one ask_user turn.

    - 'takeover' -> always the noVNC wait (caller falls back on timeout).
    - 'terminal' -> always stdin (today's default behavior).
    - 'auto'     -> credential-shaped asks go takeover (secrets stay out of
      stdin/logs); everything else goes terminal. Unknown settings fail
      closed toward 'terminal'. Never raises.
    """
    try:
        r = str(route or ASK_ROUTE_TERMINAL).strip().lower()
        if r == ASK_ROUTE_TAKEOVER:
            return "takeover"
        if r == ASK_ROUTE_AUTO:
            return "takeover" if is_credential_ask(question, context) else \
                "terminal"
        return "terminal"
    except Exception:
        return "terminal"


def default_ask_handler(question: str) -> str:
    """Terminal prompt used for interactive runs (scripts/run_agent.py)."""
    print(f"\n[Agent Needs Help]: {question}")
    return input("Your response: ")


def is_ask_user(action_or_dict: Any) -> bool:
    """True for validated Actions and raw action dicts with ask_user type."""
    if action_or_dict is None:
        return False
    at = getattr(action_or_dict, "action_type", None)
    if at is None and isinstance(action_or_dict, dict):
        at = action_or_dict.get("action_type", action_or_dict.get("action"))
    # ActionType is str-valued with use_enum_values=True, so compare by value.
    try:
        from actions.action_schema import ActionType
        return at == ActionType.ASK_USER or at == ActionType.ASK_USER.value
    except Exception:
        return at == "ask_user"


def resolve_ask_user(
    action: Any,
    action_dict: Dict[str, Any],
    ask_handler: Optional[AskHandler] = None,
) -> Dict[str, Any]:
    """Prompt the human and build the success verification + memory payload.

    Returns {"question", "human_answer", "action_dict", "verification"}.
    Never touches the browser, never raises for empty answers (empty string
    is recorded verbatim so the VLM can re-ask with a better question).
    """
    question = getattr(action, "question", None)
    if question is None and isinstance(action_dict, dict):
        question = action_dict.get("question", "")
    question = str(question or "")
    handler = ask_handler or default_ask_handler
    print(f"\n[Agent Needs Help]: {question}")
    try:
        human_answer = handler(question)
    except (EOFError, KeyboardInterrupt):
        # Non-interactive stdin (CI) or user abort: record explicitly so the
        # turn stays auditable instead of crashing the run.
        logger.warning("HITL prompt aborted/closed; recording empty answer")
        human_answer = ""
    human_answer = "" if human_answer is None else str(human_answer)
    enriched = dict(action_dict or {})
    enriched["question"] = question
    enriched["human_answer"] = human_answer
    enriched["reason"] = (
        f"Asked human: {question[:200]} | "
        f"Answer: {human_answer[:200]}"
    )
    verification = {
        "success": True,
        "reason": "ask_user answered, no browser state change expected",
        "no_state_change_expected": True,
    }
    logger.info("HITL ask_user resolved (answer %d chars)", len(human_answer))
    return {
        "question": question,
        "human_answer": human_answer,
        "action_dict": enriched,
        "verification": verification,
    }


def _dom_value_signature(dom_elements: Any) -> Tuple[int, frozenset]:
    """DOM signature INCLUDING form values (typed text must count as change).

    Only structural identity is kept (tags/labels/counts); values contribute
    a presence hash, never their content, so summaries stay secret-free.
    """
    els = dom_elements if isinstance(dom_elements, list) else []
    sig = set()
    for e in els:
        if not isinstance(e, dict):
            continue
        sig.add((
            str(e.get("tag", "")),
            str(e.get("text", "") or "")[:50],
            str(e.get("ariaLabel", "") or "")[:50],
            str(e.get("placeholder", "") or "")[:50],
            bool(str(e.get("value", "") or "").strip()),
        ))
    return len(els), frozenset(sig)


def browser_state_changed(before: Dict[str, Any], after: Dict[str, Any],
                          visual_threshold: float = 0.995) -> Tuple[bool, str]:
    """True if a human likely acted in the browser between observations.

    Checks URL, DOM structure + form-value presence, and pixels, in that
    order. Never raises; never includes user-typed content in the reason.
    """
    try:
        if (before or {}).get("url", "") != (after or {}).get("url", ""):
            return True, "url changed"
        nb, sb = _dom_value_signature((before or {}).get("dom_elements"))
        na, sa = _dom_value_signature((after or {}).get("dom_elements"))
        if nb != na or sb != sa:
            return True, "page content changed"
        bb, ab = (before or {}).get("screenshot_bytes"), \
            (after or {}).get("screenshot_bytes")
        if bb is not None and ab is not None and bb != ab:
            try:
                from PIL import Image as _Image
                import io as _io
                from utils.image_utils import compute_image_similarity
                sim = compute_image_similarity(
                    _Image.open(_io.BytesIO(bb)).convert("RGB"),
                    _Image.open(_io.BytesIO(ab)).convert("RGB"))
                if float(sim) < visual_threshold:
                    return True, "screenshot changed"
            except Exception:
                # Pixel compare unavailable: identical bytes already ruled
                # out above, so treat differing bytes as a change.
                return True, "screenshot bytes changed"
        return False, "no change detected"
    except Exception as e:
        logger.warning("Takeover diff failed: %s", e)
        return False, f"diff error: {type(e).__name__}"


def wait_for_browser_takeover(
    question: str,
    observer: Any,
    snapshot: Dict[str, Any],
    timeout_s: float = TAKEOVER_TIMEOUT_S_DEFAULT,
    poll_interval_s: float = TAKEOVER_POLL_S_DEFAULT,
) -> Dict[str, Any]:
    """Pause for noVNC takeover WITHOUT touching stdin.

    Prints a display-only banner, then polls `observer.observe()` until
    `browser_state_changed` fires or `timeout_s` elapses. Returns
    {"resumed": bool, "observation": latest_obs, "change_summary": str,
     "waited_s": float}. Never raises for observer errors (keeps polling);
    never calls input().
    """
    try:
        timeout_s = float(timeout_s)
    except (TypeError, ValueError):
        timeout_s = TAKEOVER_TIMEOUT_S_DEFAULT
    try:
        poll_interval_s = float(poll_interval_s)
    except (TypeError, ValueError):
        poll_interval_s = TAKEOVER_POLL_S_DEFAULT
    if not timeout_s > 0:
        timeout_s = TAKEOVER_TIMEOUT_S_DEFAULT
    if not 0 < poll_interval_s <= timeout_s:
        poll_interval_s = min(max(poll_interval_s, 0.1), timeout_s)
    t0 = time.time()
    print(f"\n[Agent Needs Help - noVNC takeover]: {question}")
    print("Take over in the noVNC window now. "
          "The agent resumes automatically when the page changes. "
          f"(timeout {timeout_s:.0f}s)")
    latest = snapshot
    while True:
        elapsed = time.time() - t0
        if elapsed >= timeout_s:
            logger.info("Takeover timed out after %.1fs with no page change",
                        elapsed)
            return {"resumed": False, "observation": latest,
                    "change_summary": "takeover timeout: no page change",
                    "waited_s": round(elapsed, 1)}
        time.sleep(max(0.1, poll_interval_s))
        try:
            latest = observer.observe(save_screenshot=False)
        except Exception as e:
            logger.warning("Takeover re-observe failed (retrying): %s", e)
            continue
        changed, why = browser_state_changed(snapshot, latest)
        if changed:
            elapsed = time.time() - t0
            logger.info("Takeover change detected after %.1fs (%s)",
                        elapsed, why)
            return {"resumed": True, "observation": latest,
                    "change_summary": f"human acted in browser ({why})",
                    "waited_s": round(elapsed, 1)}
