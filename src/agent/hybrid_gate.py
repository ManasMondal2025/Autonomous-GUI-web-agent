"""Experiment 2: hybrid deterministic + selective-LLM intent gate.

EXPERIMENTAL ONLY. Disabled by default. Preserves Experiment 1 untouched.

Pipeline per gated action (deterministic FIRST, LLM only on ambiguity):

    explicit value present?  -> ALLOW  (0 LLM calls)
    clearly missing?         -> ASK    (0 LLM calls, deterministic question)
    relative/ambiguous hint? -> qwen2.5:7b text judge (no screenshot)

Mode selection (env ``INTENT_GATE_MODE``): off | llm | hybrid.
Default ``hybrid`` (production). Unset MODE + legacy
``ENABLE_EXPERIMENTAL_INTENT_GATE=1`` resolves to ``llm`` (Experiment 1
behavior preserved exactly); explicit ``off`` disables all gating.

Secret handling: credential VALUES are redacted in every log/report
(``[REDACTED]``). Field names are safe to log.
"""

import json
import logging
import os
import re
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from .intent_gate import (
    should_consult_gate,
    find_action_target,
    build_dom_context,
    format_history,
)

logger = logging.getLogger(__name__)

MODE_ENV = "INTENT_GATE_MODE"
LEGACY_FLAG_ENV = "ENABLE_EXPERIMENTAL_INTENT_GATE"
MODEL_ENV = "INTENT_GATE_7B_MODEL"
OLLAMA_URL_ENV = "INTENT_GATE_OLLAMA_URL"

DEFAULT_7B_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Words that make a reference relative (needs UI/LLM resolution).
# Bare "my" is excluded: "log in to my account" is not a relative VALUE
# reference. Only "my X" compounds count.
RELATIVE_HINTS = {
    "latest", "newest", "recent", "most recent", "current", "usual",
    "preferred", "favourite", "favorite", "correct", "right one",
    "earliest", "first", "last",
    "my usual", "my preferred", "my favourite", "my favorite",
    "my current", "my latest", "my newest", "my default",
}
SECRET_FIELDS = {"password", "passwd", "pwd", "secret", "token", "passcode"}
DEGREE_TOKENS = {"msc", "bsc", "phd", "mba", "btech", "mtech", "ms", "bs", "ba", "ma"}
STOPWORDS = {
    "the", "a", "an", "to", "for", "of", "in", "on", "my", "me", "all",
    "show", "please", "with", "and", "or", "is", "are", "it", "this",
    "that", "open", "page", "click", "type", "search", "view", "go",
}

QUESTION_TEMPLATES = {
    "batch": "Which batch would you like to view?",
    "course": "Which course would you like to view?",
    "department": "Which department would you like to view?",
    "date": "What date would you like to use?",
    "username": "What username should I use?",
    "password": "What password should I use?",
    "address": "What address should I use?",
    "search": "What would you like to search for?",
}


# --------------------------------------------------------------------------
# Mode + redaction helpers
# --------------------------------------------------------------------------

def get_intent_gate_mode() -> str:
    """off | llm (Exp1) | hybrid (Exp2, production default)."""
    raw = os.getenv(MODE_ENV, "").strip().lower()
    if raw in ("off", "llm", "hybrid"):
        return raw
    if os.getenv(LEGACY_FLAG_ENV, "").strip().lower() in ("1", "true", "yes", "on"):
        return "llm"
    return "hybrid"


def redact_secrets(text: Any) -> str:
    """Redact credential values; field names stay visible."""
    s = str(text or "")
    # password[:=]value / with password value / username value
    s = re.sub(r"(?i)(password|passwd|pwd|secret|token)\s*[:=]?\s*(\S+)",
               r"\1=[REDACTED]", s)
    s = re.sub(r"(?i)\bwith password\s+(\S+)", "with password [REDACTED]", s)
    s = re.sub(r"(?i)\busername\s+([A-Za-z0-9_.\-@]+)", "username [REDACTED]", s)
    s = re.sub(r"(?i)\blog ?in as\s+([A-Za-z0-9_.\-@]+)", "log in as [REDACTED]", s)
    return s


def normalize_value(s: Any) -> str:
    """lowercase, trim, collapse spaces, strip punctuation (keep alnum)."""
    s = str(s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------
# Field semantics
# --------------------------------------------------------------------------

def semantic_field(el: Optional[Dict[str, Any]]) -> str:
    """Human name of the parameter a control represents (e.g. 'Batch')."""
    if not isinstance(el, dict):
        return ""
    if str(el.get("type", "")).lower() == "password":
        return "Password"
    for key in ("ariaLabel", "placeholder", "name", "title"):
        v = str(el.get(key, "") or "").strip().rstrip(":*?")
        if v:
            return v[:60]
    text = str(el.get("text", "") or "").strip()
    if text and len(text) <= 40 and "\n" not in text:
        return text
    return str(el.get("tag", "") or "field")


def _is_option_like(e: Dict[str, Any]) -> Optional[str]:
    """Option text if the element looks like a selectable value, else None."""
    if not isinstance(e, dict):
        return None
    tag = str(e.get("tag", "")).lower()
    role = str(e.get("role", "")).lower()
    if tag in ("option",) or role in ("option", "menuitemradio"):
        t = (str(e.get("text", "") or "").strip()
             or str(e.get("ariaLabel", "") or "").strip())
        return t or None
    if tag == "button":
        t = (str(e.get("text", "") or "").strip()
             or str(e.get("ariaLabel", "") or "").strip())
        if t and len(t) <= 24:
            return t
    return None


def ui_options_for(field: str, target: Optional[Dict[str, Any]],
                   dom: List[Dict[str, Any]]) -> List[str]:
    """Visible candidate values: sibling option-like elements' texts."""
    opts = []
    for e in dom or []:
        if e is target:
            continue
        t = _is_option_like(e)
        if t:
            opts.append(t)
    # De-dupe, preserve order, cap.
    seen, out = set(), []
    for o in opts:
        if o not in seen:
            seen.add(o)
            out.append(o)
    return out[:12]


def ordered_options_for(target: Optional[Dict[str, Any]],
                        dom: List[Dict[str, Any]]) -> List[str]:
    """DOM-ordered option texts INCLUDING the target (for order checks).

    The chronological-order test is only meaningful on the listed order;
    prepending the target (as full_options did) fabricates disorder.
    """
    out = []
    for e in dom or []:
        if not isinstance(e, dict):
            continue
        if e is target:
            t = (str(e.get("text", "") or "").strip()
                 or str(e.get("ariaLabel", "") or "").strip())
            if t:
                out.append(t)
            continue
        t = _is_option_like(e)
        if t:
            out.append(t)
    seen, deduped = set(), []
    for o in out:
        if o not in seen:
            seen.add(o)
            deduped.append(o)
    return deduped[:13]


# --------------------------------------------------------------------------
# Deterministic user-info extraction (no LLM)
# --------------------------------------------------------------------------

LOGIN_WORDS = {"login", "log in", "sign in", "signin", "credential",
               "username", "password", "account"}


def parse_history_answers(history: Any) -> List[Tuple[str, List[str]]]:
    """Parse prior HITL Q&A pairs: [(question, [value, ...])].

    Matches the `Asked human: Q | Answer: A` reason format written by
    `hitl.resolve_ask_user`, plus bare `Answer: A` lines (question="").
    """
    pairs = []
    texts = []
    if isinstance(history, list):
        for h in history:
            if isinstance(h, dict):
                texts.append(str(h.get("reason", "")))
                texts.append(str(h.get("result", "")))
    for t in texts:
        m = re.search(r"Asked human:\s*(.*?)\s*\|\s*Answer:\s*([^\n]+)",
                      t, re.IGNORECASE)
        if m:
            q, a = m.group(1).strip(), m.group(2).strip()
        else:
            m2 = re.search(r"Answer:\s*([^\n|]+)", t, re.IGNORECASE)
            if not m2:
                continue
            q, a = "", m2.group(1).strip()
        parts = [p.strip(" ,;") for p in re.split(r"[/,;]", a) if p.strip(" ,;")]
        if parts:
            pairs.append((q, parts))
    return pairs


def history_covers(field_label: str, kind: str,
                   pairs: List[Tuple[str, List[str]]],
                   task_goal: str) -> Tuple[bool, str]:
    """True if a prior human answer supplies this field's value."""
    norm_label = normalize_value(field_label)
    blob = normalize_value(task_goal or "")
    for q, parts in pairs:
        nq = normalize_value(q)
        if norm_label and norm_label in nq and parts:
            return True, "prior answer names this field"
        # Login-shaped Q&A ("credentials?", "username and password?"):
        # first part -> username, second -> password.
        if kind in ("username", "password") and parts:
            loginish = any(w in nq or w in blob for w in LOGIN_WORDS)
            if loginish:
                if kind == "username":
                    return True, "username answered in history"
                if len(parts) >= 2 or "password" in nq or "password" in blob:
                    return True, "password answered in history"
    return False, ""


def extract_user_provided_info(task_goal: str, history: Any) -> Dict[str, Any]:
    """Pull candidate values from goal + history text (deterministic)."""
    hist_texts = []
    if isinstance(history, list):
        for h in history:
            if isinstance(h, dict):
                hist_texts.append(str(h.get("reason", "")))
                hist_texts.append(str(h.get("result", "")))
    blob = f"{task_goal or ''}\n" + "\n".join(hist_texts)
    quoted = re.findall(r'"([^"]+)"', blob) + re.findall(r"'([^']+)'", blob)
    years = re.findall(r"\b((?:19|20)\d{2})\b", blob)
    degrees = [t for t in re.findall(r"[A-Za-z.]+", blob)
               if normalize_value(t).replace(" ", "") in
               {d.replace(".", "") for d in DEGREE_TOKENS} or
               normalize_value(t) in DEGREE_TOKENS]
    dates = (re.findall(r"\b\d{4}-\d{2}-\d{2}\b", blob)
             + re.findall(r"(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b", blob))
    m_user = re.search(r"(?i)(?:as|username|user)\s+([A-Za-z0-9_.\-@]+)", blob)
    m_pass = re.search(r"(?i)password\s+(\S+)", blob)
    tokens = {normalize_value(t) for t in re.findall(r"[A-Za-z0-9@.]+", blob)}
    tokens.discard("")
    lowered = blob.lower()
    # Single words match on tokens (avoids "first" hitting "firsthand");
    # multi-word phrases match on substrings.
    hints = sorted(
        {h for h in RELATIVE_HINTS
         if (h in lowered if " " in h else h in tokens)})
    return {
        "quoted": quoted,
        "years": years,
        "degrees": degrees,
        "dates": dates,
        "username": m_user.group(1) if m_user else "",
        "password": m_pass.group(1) if m_pass else "",
        "tokens": tokens,
        "relative_hints": hints,
        "answer_pairs": [(q, list(v)) for q, v in
                         parse_history_answers(history)],
        "raw": blob,
    }


def _field_kind(label: str, el: Optional[Dict[str, Any]]) -> str:
    n = normalize_value(label)
    if isinstance(el, dict) and str(el.get("type", "")).lower() == "password":
        return "password"
    if "user" in n and "name" in n or n in ("username", "user", "login", "email"):
        return "username"
    if "password" in n or "passcode" in n:
        return "password"
    if "batch" in n or n in ("year", "session"):
        return "batch"
    if "course" in n or "degree" in n or "program" in n:
        return "course"
    if "date" in n or "depart" in n or "when" in n or "month" in n:
        return "date"
    if "search" in n or (isinstance(el, dict)
                         and str(el.get("role", "")).lower() == "searchbox"):
        return "search"
    if "address" in n or "city" in n or "street" in n:
        return "address"
    return "generic"


def deterministic_match(field: str, kind: str, provided: Dict[str, Any],
                         options: List[str],
                         el: Optional[Dict[str, Any]],
                         task_goal: str = "",
                         resolve_options: Optional[List[str]] = None,
                         ) -> Tuple[str, str]:
    """Return (ALLOW | ASK | AMBIGUOUS, reason). Never calls an LLM."""
    norm_opts = [normalize_value(o) for o in options]
    tokens = provided.get("tokens", set())
    quoted_n = [normalize_value(q) for q in provided.get("quoted", [])]
    # A prior human answer for this field satisfies it outright.
    covered, why_covered = history_covers(
        field, kind, provided.get("answer_pairs", []), task_goal)
    if covered:
        return "ALLOW", why_covered

    def _relative_or_ambiguous():
        # Recency-aware deterministic resolution; AMBIGUOUS on failure.
        # NOTE: `task_goal` is captured from the enclosing scope.
        # The resolver needs the FULL listed set (target included);
        # `options` (siblings only) stays as-is for mention matching so a
        # control's own label can never self-authorize.
        full = list(resolve_options) if resolve_options is not None \
            else list(options)
        pick, why = resolve_relative_reference(task_goal, full)
        if pick is not None:
            return "ALLOW", f"resolved relative reference: {pick} ({why})"
        return "AMBIGUOUS", f"relative hint ({why})"

    def opt_mentioned() -> Optional[str]:
        for o, n in zip(options, norm_opts):
            if n and (n in tokens or n in quoted_n):
                return o
        return None

    # Control already holds a value -> nothing to ask.
    if isinstance(el, dict) and str(el.get("value", "") or "").strip():
        return "ALLOW", "control already has a value"

    if kind == "batch":
        if provided.get("years"):
            # Recency precedence: later corrections override earlier values.
            return "ALLOW", f"explicit year {provided['years'][-1]}"
        hit = opt_mentioned()
        if hit:
            return "ALLOW", f"explicit option {hit!r}"
        if provided.get("relative_hints"):
            return _relative_or_ambiguous()
        return "ASK", "batch value absent"
    if kind == "course":
        deg = [d for d in provided.get("degrees", [])]
        if deg:
            return "ALLOW", f"explicit course {deg[-1]}"
        hit = opt_mentioned()
        if hit:
            return "ALLOW", f"explicit option {hit!r}"
        if provided.get("relative_hints"):
            return _relative_or_ambiguous()
        return "ASK", "course value absent"
    if kind == "username":
        if provided.get("username"):
            return "ALLOW", "username provided"
        if provided.get("relative_hints"):
            return "AMBIGUOUS", "relative hint"
        return "ASK", "username absent"
    if kind == "password":
        if provided.get("password"):
            return "ALLOW", "password provided"
        if provided.get("relative_hints"):
            return "AMBIGUOUS", "relative hint"
        return "ASK", "password absent"
    if kind == "date":
        if provided.get("dates") or provided.get("years"):
            return "ALLOW", "explicit date"
        hit = opt_mentioned()
        if hit:
            return "ALLOW", f"explicit option {hit!r}"
        if provided.get("relative_hints"):
            return _relative_or_ambiguous()
        return "ASK", "date absent"
    if kind == "search":
        content = {t for t in tokens
                   if t not in STOPWORDS and t != normalize_value(field)}
        content |= {q for q in quoted_n if q}
        if content:
            return "ALLOW", "search query present"
        return "ASK", "search query absent"
    # generic select-like: explicit option mention -> ALLOW.
    hit = opt_mentioned()
    if hit:
        return "ALLOW", f"explicit option {hit!r}"
    # Choice-option buttons (e.g. a "2024" year button): the user naming the
    # option itself counts as explicit. Narrow to option-like targets only:
    # control labels ("Batch") naturally appear in goals and must NOT allow.
    if isinstance(el, dict) and (
            str(el.get("tag", "")).lower() in ("button", "option")
            or str(el.get("role", "")).lower() in ("option", "menuitemradio")):
        t = normalize_value(
            str(el.get("text", "") or "") or str(el.get("ariaLabel", "") or ""))
        if t and (t in tokens or t in quoted_n):
            return "ALLOW", f"target option {t!r} explicitly named"
    if provided.get("relative_hints"):
        return _relative_or_ambiguous()
    return "ASK", "value absent"


MONTHS = {"january": 1, "february": 2, "march": 3, "april": 4,
          "may": 5, "june": 6, "july": 7, "august": 8, "september": 9,
          "october": 10, "november": 11, "december": 12,
          "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
          "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12}


def parse_temporal_option(text: str):
    """Parse year / month-year / ISO date to a comparable (y, m, d) or None.

    Generic: no hardcoded labels, only numeric/month-name shapes.
    """
    t = str(text or "").strip()
    m = re.fullmatch(r"\s*((?:19|20)\d{2})\s*", t)
    if m:
        return (int(m.group(1)), 0, 0)
    m = re.fullmatch(r"\s*([A-Za-z]+)\s+((?:19|20)\d{2})\s*", t)
    if m and m.group(1).lower() in MONTHS:
        return (int(m.group(2)), MONTHS[m.group(1).lower()], 0)
    m = re.fullmatch(r"\s*((?:19|20)\d{2})-(\d{2})-(\d{2})\s*", t)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def resolve_relative_reference(task_goal: str,
                                options: List[str]):
    """Resolve positional/temporal hints against listed UI options.

    Returns (option_text, reason) or (None, reason). Conservative:
    - first/last: positional, always safe when options are listed.
    - latest/newest/most-recent/earliest: ONLY when every option parses as
      temporal AND the listed order is sorted (either direction), proving
      the UI itself establishes chronology. Unordered or non-temporal
      options -> None (caller escalates to the LLM judge).
    """
    opts = [o for o in (options or []) if str(o or "").strip()]
    if not opts:
        return None, "no listed options"
    toks = set(normalize_value(task_goal).split())
    if "first" in toks:
        return opts[0], "first listed option"
    if "last" in toks:
        return opts[-1], "last listed option"
    wants_latest = ("latest" in toks or "newest" in toks
                    or "recent" in toks
                    or "most recent" in normalize_value(task_goal))
    wants_earliest = "earliest" in toks
    if not (wants_latest or wants_earliest):
        return None, "no resolvable relative hint"
    keys = [parse_temporal_option(o) for o in opts]
    if any(k is None for k in keys):
        return None, "options not all temporal"
    if not (keys == sorted(keys) or keys == sorted(keys, reverse=True)):
        return None, "option order does not establish chronology"
    pick = opts[keys.index(max(keys))] if wants_latest \
        else opts[keys.index(min(keys))]
    return pick, f"{'latest' if wants_latest else 'earliest'} of ordered options"


def deterministic_question(field: str) -> str:
    key = normalize_value(field).split()[0] if field else ""
    return QUESTION_TEMPLATES.get(key, f"Please provide the {field}." if field
                                  else "Please provide the missing information.")


# --------------------------------------------------------------------------
# Qwen2.5:7b text judge over Ollama HTTP (stdlib only)
# --------------------------------------------------------------------------

JUDGE_SYSTEM = """You are a strict information-sufficiency validator for a web automation agent.
Your ONLY task is to decide whether a required UI field has a sufficiently specified value.
You are NOT responsible for deciding where to click.
You are NOT responsible for visual grounding.
You are NOT allowed to invent information.
The user may provide a value explicitly or indirectly using an unambiguous phrase such as "2023", "latest batch", "the newest version".
A value is SUFFICIENT only if the user's request/history and the provided UI options allow the required field to be determined unambiguously.
If the value is clearly present, return ALLOW.
If the value is clearly missing and cannot be inferred, return ASK.
If the case is genuinely ambiguous, return ASK.
IMPORTANT: Do not ask for information that is already present in the user request.
NEVER invent a value.
Return ONLY ALLOW or ASK."""


class OllamaTextJudge:
    """Minimal Ollama client (urllib, no new deps). Counts calls."""

    def __init__(self, model: Optional[str] = None,
                 base_url: Optional[str] = None,
                 timeout: int = 120, num_predict: int = 16):
        self.model = model or os.getenv(MODEL_ENV, DEFAULT_7B_MODEL)
        base = (base_url or os.getenv(OLLAMA_URL_ENV, DEFAULT_OLLAMA_URL)
                ).rstrip("/")
        self.url = base + "/api/generate"
        self.timeout = timeout
        self.num_predict = num_predict
        self.calls = 0

    def generate(self, prompt: str) -> str:
        import urllib.error as _uerr
        self.calls += 1
        body = json.dumps({
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0, "num_predict": self.num_predict},
        }).encode()
        req = urllib.request.Request(
            self.url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            raise RuntimeError(f"ollama judge request failed: {e}") from e
        return str(data.get("response", "")).strip()


# --------------------------------------------------------------------------
# Hybrid gate
# --------------------------------------------------------------------------

HYBRID_NAV_LABELS = {
    "settings", "profile", "help", "logout", "sign out", "search",
}


def _in_choice_group(el: Dict[str, Any],
                     dom: List[Dict[str, Any]]) -> bool:
    """True if the target is one of >=2 sibling short-text buttons.

    Plain buttons are normally navigation (bypass), but a row of value-like
    buttons (years, slots, colors) is a value choice and must be gated.
    """
    if not isinstance(el, dict):
        return False
    if str(el.get("tag", "")).lower() != "button":
        return False
    label = (str(el.get("text", "") or "").strip()
             or str(el.get("ariaLabel", "") or "").strip())
    if not label or len(label) > 24:
        return False
    if label.lower() in HYBRID_NAV_LABELS:
        return False
    peers = 0
    for e in dom or []:
        if not isinstance(e, dict) or e is el:
            continue
        if str(e.get("tag", "")).lower() != "button":
            continue
        t = (str(e.get("text", "") or "").strip()
             or str(e.get("ariaLabel", "") or "").strip())
        if t and len(t) <= 24:
            peers += 1
    return peers >= 1


def hybrid_should_consult(action: Any, action_dict: Optional[Dict[str, Any]],
                           dom_elements: List[Dict[str, Any]]):
    """Experiment-2 routing: Experiment-1 rules + choice-group options.

    Returns (consult, reason, target). Never raises.
    """
    from .intent_gate import _looks_like_navigation
    try:
        consult, why, target = should_consult_gate(
            action, action_dict, dom_elements or [])
    except Exception:
        return False, "routing_error", None
    if consult or target is None:
        return consult, why, target
    if isinstance(action, dict):
        at = action.get("action_type", action.get("action", ""))
    elif isinstance(action_dict, dict):
        at = action_dict.get("action_type", action_dict.get("action", ""))
    else:
        at = getattr(action, "action_type", "")
    if str(at).lower() != "click":
        return False, why, target
    try:
        if _looks_like_navigation(target):
            return False, "navigation_control", target
    except Exception:
        pass
    if _in_choice_group(target, dom_elements or []):
        return True, "choice_group_option", target
    return False, why, target


class HybridGate:
    """Deterministic-first gate; 7B judge only for AMBIGUOUS cases."""

    def __init__(self, llm: Any = None):
        # llm: object with generate(prompt)->str (OllamaTextJudge or stub).
        # None -> AMBIGUOUS falls back to ALLOW (fail-safe).
        self.llm = llm

    @property
    def llm_calls(self) -> int:
        return getattr(self.llm, "calls", 0) if self.llm is not None else 0

    def decide(self, task_goal: str, history: Any, action: Any,
               dom_elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Full routing + verdict. Never raises (fail-safe ALLOW)."""
        t0 = time.time()
        try:
            return self._decide(task_goal, history, action, dom_elements, t0)
        except Exception as e:
            logger.warning("[HYBRID-GATE] ERROR %s", redact_secrets(e))
            logger.info("[HYBRID-GATE] fallback=ALLOW")
            return {"decision": "ALLOW", "missing_field": "", "question": "",
                    "reason": f"gate error: {type(e).__name__}",
                    "path": "error-fallback", "llm_called": False,
                    "latency_s": round(time.time() - t0, 3)}

    def _decide(self, task_goal, history, action, dom_elements, t0):
        consult, why, target = hybrid_should_consult(
            action, None, dom_elements or [])
        at = self._action_name(action)
        if not consult:
            logger.info("[HYBRID-GATE] bypass action=%s reason=%s", at, why)
            return self._out("ALLOW", "", "", f"bypass:{why}", "bypass",
                             False, t0)
        target = find_action_target(action, None, dom_elements or []) or target
        field = semantic_field(target)
        kind = _field_kind(field, target)
        options = ui_options_for(field, target, dom_elements or [])
        full_options = ordered_options_for(target, dom_elements or [])
        provided = extract_user_provided_info(task_goal, history)
        logger.info("[HYBRID-GATE] action=%s field=%s kind=%s options=%s",
                    at, field, kind, options)
        verdict, why_det = deterministic_match(
            field, kind, provided, options, target, task_goal,
            resolve_options=full_options)
        if verdict == "ALLOW":
            logger.info("[HYBRID-GATE] decision=ALLOW path=deterministic "
                        "reason=%s", why_det)
            return self._out("ALLOW", "", "", why_det, "deterministic-allow",
                             False, t0)
        if verdict == "ASK":
            q = deterministic_question(field)
            logger.info("[HYBRID-GATE] decision=ASK path=deterministic "
                        "missing_field=%s", field)
            logger.info("[HYBRID-GATE] question=%.120s", q)
            return self._out("ASK", field, q, why_det, "deterministic-ask",
                             False, t0)
        # AMBIGUOUS -> selective 7B judge.
        return self._llm_judge(task_goal, history, action, field, kind,
                               provided, options, target, t0)

    def _llm_judge(self, task_goal, history, action, field, kind,
                   provided, options, target, t0):
        if self.llm is None:
            logger.info("[HYBRID-GATE] no-llm fallback=ALLOW")
            return self._out("ALLOW", "", "", "ambiguous but no judge",
                             "no-llm-fallback", False, t0)
        safe_provided = {k: ("[REDACTED]" if k.lower() in SECRET_FIELDS
                             or "pass" in k.lower() else v)
                         for k, v in provided.items() if k != "raw"
                         and k != "tokens"}
        safe_provided["tokens"] = sorted(
            t for t in provided.get("tokens", set()))[:40]
        prompt = (
            f"{JUDGE_SYSTEM}\n\nTASK:\n{redact_secrets(task_goal)}\n\n"
            f"RELEVANT HISTORY:\n{redact_secrets(format_history(history))}\n\n"
            f"REQUIRED UI FIELD:\n{field}\n\n"
            f"USER-PROVIDED INFORMATION:\n"
            f"{redact_secrets(json.dumps(safe_provided, default=str))}\n\n"
            f"UI CONTROL:\n{field} (kind={kind})\n\n"
            f"AVAILABLE UI VALUES:\n{', '.join(options) if options else '(none listed)'}\n\n"
            f"PROPOSED ACTION:\n{self._describe(action)}\n\n"
            f"Return ONLY ALLOW or ASK.")
        try:
            raw = self.llm.generate(prompt)
        except Exception as e:
            logger.warning("[HYBRID-GATE] ERROR judge crashed: %s",
                           redact_secrets(e))
            logger.info("[HYBRID-GATE] fallback=ALLOW")
            return self._out("ALLOW", "", "", f"judge error: {type(e).__name__}",
                             "llm-error-fallback", False, t0)
        m = re.search(r"\b(ALLOW|ASK)\b", (raw or "").upper())
        if not m:
            logger.warning("[HYBRID-GATE] ERROR unparseable judge output")
            logger.info("[HYBRID-GATE] fallback=ALLOW")
            return self._out("ALLOW", "", "", "unparseable judge output",
                             "llm-error-fallback", True, t0)
        if m.group(1) == "ALLOW":
            logger.info("[HYBRID-GATE] decision=ALLOW path=llm")
            return self._out("ALLOW", "", "", "7b judge ALLOW", "llm",
                             True, t0)
        q = deterministic_question(field)
        logger.info("[HYBRID-GATE] decision=ASK path=llm missing_field=%s",
                    field)
        logger.info("[HYBRID-GATE] question=%.120s", q)
        return self._out("ASK", field, q, "7b judge ASK", "llm", True, t0)

    @staticmethod
    def _action_name(action: Any) -> str:
        if isinstance(action, dict):
            return str(action.get("action_type", action.get("action", "?")))
        return str(getattr(action, "action_type", "?"))

    @staticmethod
    def _describe(action: Any) -> str:
        if isinstance(action, dict):
            d = dict(action)
        else:
            d = {k: getattr(action, k, None) for k in
                 ("action_type", "target_id", "x", "y", "value", "reason")}
        return redact_secrets(json.dumps(
            {k: v for k, v in d.items() if v is not None}, default=str)[:300])

    @staticmethod
    def _out(decision, missing_field, question, reason, path,
             llm_called, t0) -> Dict[str, Any]:
        return {"decision": decision, "missing_field": missing_field,
                "question": question, "reason": reason, "path": path,
                "llm_called": llm_called,
                "latency_s": round(time.time() - t0, 3)}
