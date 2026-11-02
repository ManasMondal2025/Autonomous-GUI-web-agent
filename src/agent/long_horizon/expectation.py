"""Semantic expectation normalization for STV.

Planners phrase the same state predicate many ways ("The Select country
dropdown contains India", "Country = India", "India is selected for
country"). STV must verify the underlying predicate -- TARGET + PREDICATE
+ EXPECTED VALUE -- against deterministic DOM state first, page text second,
never by requiring literal lexical overlap between planner wording and
webpage text.

Public API:
  normalize_expectation(text) -> Optional[dict]
      {"target": [words], "predicate": str, "value": str, "raw": str}
      None only when nothing intelligible remains (empty/stopwords-only).
      Non-empty but unparseable text normalizes to a VISIBLE check on its
      content words (never a silent pass, never a crash).

  verify_item(norm, before_pool, after_pool, dom_elements)
      -> (ok: bool, note: str)
      DOM-first: when the target names form field(s), the predicate is
      checked against actual field values (e.g. select value == "India").
      Page-text fallback only when no field matches the target.

Predicate vocabulary (only these; each exists for a live project need):
  EQUALS, CONTAINS, NOT_CONTAINS, GREATER_THAN, LESS_THAN, SELECTED,
  VISIBLE, EXISTS, APPLIED, PAGE_IS, CHANGED.

Conservatism (§9): a predicate that names a field but fails against it
returns False (no page-global rescue -- that was the old false-PASS).
Unresolvable numbers for GT/LT return False. Only VISIBLE/EXISTS/PAGE_IS
(and field-less EQUALS/CONTAINS/SELECTED/APPLIED) consult page text.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

PREDICATES = ("EQUALS", "CONTAINS", "NOT_CONTAINS", "GREATER_THAN",
              "LESS_THAN", "SELECTED", "VISIBLE", "EXISTS", "APPLIED",
              "PAGE_IS", "CHANGED")

# Nouns that are requirement chrome, not state ("country dropdown" names
# the country field; "dropdown" itself never appears verbatim). Stripped
# from TARGETS only -- never from values, never globally.
TARGET_META_NOUNS = frozenset({
    "field", "fields", "dropdown", "selector", "selectors", "control",
    "controls", "input", "inputs", "box", "boxes", "button", "buttons",
    "option", "options", "value", "values", "entry", "entries", "form",
    "select", "selected", "menu", "detail", "details",
})

_VERBS = frozenset({
    "is", "are", "was", "were", "be", "been", "being", "has", "have",
    "had", "contains", "contain", "containing", "shows", "show", "showing",
    "shown", "displays", "display", "selected", "select", "applied",
    "apply", "visible", "remove", "removed", "gone", "deleted", "cleared",
    "empty",
})

_GENERIC_CHANGE_WORDS = frozenset({
    "state", "change", "changes", "changed", "changing", "result",
    "results", "page", "action", "update", "updated", "updates",
})

_QUOTED_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')
_NUMBER_RE = re.compile(r"[\d][\d,]*\.?\d*")
_MEANINGFUL_TYPES = frozenset({
    "email", "tel", "number", "password", "radio", "checkbox", "url",
    "date",
})


def _words(text: str) -> List[str]:
    """Content words via the shared STV pipeline (stopwords + glue out)."""
    from .stv import _content_words
    return _content_words(text or "")


def _strip_quotes(text: str) -> str:
    return _QUOTED_RE.sub(" ", text or "")


def _quoted_spans(text: str) -> List[str]:
    out = []
    for m in _QUOTED_RE.finditer(text or ""):
        span = (m.group(1) or m.group(2) or "").strip()
        if span:
            out.append(span)
    return out


def _target_words(text: str, value_words: List[str]) -> List[str]:
    vw = set(value_words)
    return [w for w in _words(text)
            if w not in TARGET_META_NOUNS
            and w not in _VERBS
            and w not in vw]


def _bare_value(text: str, target_words: List[str]) -> str:
    """Best-effort value for unstructured phrases: strip target-ish words,
    keep the rest (numbers, quoted spans, distinctive tokens)."""
    tw = set(target_words)
    spans = _quoted_spans(text)
    if spans:
        return " ".join(spans)
    rest = [w for w in _words(text) if w not in tw]
    if rest:
        return " ".join(rest)
    return " ".join(target_words)


def normalize_expectation(text: str) -> Optional[Dict[str, Any]]:
    """Parse one planner expectation into TARGET+PREDICATE+VALUE.

    Returns None ONLY for empty/stopwords-only input (no requirement
    stated). Anything else yields a predicate; unstructured phrases become
    VISIBLE checks on their content words, generic "state changes" prose
    becomes CHANGED.
    """
    raw = str(text or "").strip()
    if not raw:
        return None
    low = raw.lower()
    bare = re.fullmatch(r'\s*"([^"]+)"\s*', raw)
    if bare and bare.group(1).strip():
        # Bare quoted literal (deterministic type-action expectation):
        # verbatim visibility, never parsed as prose. Checked before the
        # stopword guard so short literals ("3", "M") stay verifiable.
        return {"target": [], "predicate": "VISIBLE", "value": raw,
                "raw": raw}
    if not _words(raw):
        return None  # stopwords-only: no requirement stated

    m = re.search(r"^(.*?)\s*!=\s*(.+)$", raw)
    if m:
        value = m.group(2).strip().strip("\"'").rstrip(".,;:!?")
        return {"target": _target_words(m.group(1), _words(value)),
                "predicate": "NOT_CONTAINS", "value": value, "raw": raw}
    m = re.search(
        r"^(.*?)\b(greater than|more than|above|over|at least)\b\s*"
        r"([$]?[\d][\d,]*\.?\d*)\s*$"
        r"|^(.*?)\b(less than|fewer than|below|under|at most)\b\s*"
        r"([$]?[\d][\d,]*\.?\d*)\s*$", raw, re.IGNORECASE)
    if m:
        if m.group(2) is not None:
            head, value = m.group(1), m.group(3).strip()
            pred = "GREATER_THAN"
        else:
            head, value = m.group(4), m.group(6).strip()
            pred = "LESS_THAN"
        return {"target": _target_words(head, _words(value)),
                "predicate": pred, "value": value, "raw": raw}
    m = re.search(r"^(.*?)\s*(>=|<=|>|<)\s*([$]?[\d][\d,]*\.?\d*)\s*$", raw)
    if m and m.group(1).strip():
        pred = ("GREATER_THAN" if m.group(2).startswith(">")
                else "LESS_THAN")
        value = m.group(3).strip()
        return {"target": _target_words(m.group(1), _words(value)),
                "predicate": pred, "value": value, "raw": raw}
    m = re.search(r"^(.*?)\s*=\s*(.+)$", raw)
    if m and m.group(1).strip() and m.group(2).strip():
        value = m.group(2).strip().strip("\"'").rstrip(".,;:!?")
        return {"target": _target_words(m.group(1), _words(value)),
                "predicate": "EQUALS", "value": value, "raw": raw}
    m = re.search(r"(.*?)\bfrom\b\s+(.+?)\s+\bto\b\s+(.+)$", raw,
                  re.IGNORECASE)
    if m and m.group(2).strip() and m.group(3).strip():
        # "Change PIN from 700016 to 700017": only the new value can hold.
        value = m.group(3).strip().strip("\"'").rstrip(".,;:!?")
        return {"target": _target_words(m.group(1), _words(value)),
                "predicate": "EQUALS", "value": value, "raw": raw}
    m = re.search(
        r"(.*?)\b(does not|doesn't|dont|do not|is not|isn't|are not|"
        r"no|without)\b(.*)$",
        raw, re.IGNORECASE)
    if m and (m.group(1).strip() or m.group(3).strip()):
        rest = (m.group(1) + " " + m.group(3)).strip()
        spans = _quoted_spans(raw)
        value = spans[0] if spans else _bare_value(rest, [])
        return {"target": _target_words(rest, _words(value)),
                "predicate": "NOT_CONTAINS", "value": value, "raw": raw}
    m = re.search(
        r"(.*?)\b(contains|contain|containing|has|have|shows|show|showing|"
        r"displays|display)\b(.*)$", raw, re.IGNORECASE)
    if m and m.group(3).strip():
        spans = _quoted_spans(raw)
        value = spans[0] if spans else m.group(3).strip().strip(
            "\"'").rstrip(".,;:!?")
        return {"target": _target_words(m.group(1), _words(value)),
                "predicate": "CONTAINS", "value": value, "raw": raw}
    m = re.search(r"^(.*?)\bis\b\s*selected\s+for\s+(.+)$", raw,
                  re.IGNORECASE)
    if m and m.group(1).strip():
        spans = _quoted_spans(m.group(1))
        value = spans[0] if spans else m.group(1).strip().strip(
            "\"'").rstrip(".,;:!?")
        return {"target": _target_words(m.group(2), _words(value)),
                "predicate": "SELECTED", "value": value, "raw": raw}
    m = re.search(r"(.*?)\b(is|are|was|now)\b(.*)$", raw, re.IGNORECASE)
    if m and m.group(3).strip():
        spans = _quoted_spans(raw)
        value = spans[0] if spans else m.group(3).strip().strip(
            "\"'").rstrip(".,;:!?")
        pred = "SELECTED" if "select" in low else "EQUALS"
        return {"target": _target_words(m.group(1), _words(value)),
                "predicate": pred, "value": value, "raw": raw}
    m = re.search(r"(.*?)\bselected\b(.*)$", raw, re.IGNORECASE)
    if m:
        spans = _quoted_spans(raw)
        value = spans[0] if spans else _bare_value(raw, [])
        return {"target": _target_words(_strip_quotes(raw), _words(value)),
                "predicate": "SELECTED", "value": value, "raw": raw}
    m = re.search(r"(.*?)\b(applied|used|active|enabled)\b(.*)$", raw,
                  re.IGNORECASE)
    if m:
        spans = _quoted_spans(raw)
        value = spans[0] if spans else _bare_value(raw, [])
        return {"target": _target_words(_strip_quotes(raw), _words(value)),
                "predicate": "APPLIED", "value": value, "raw": raw}
    m = re.search(
        r"(.*?)\b(removed|gone|deleted|cleared|empty)\b(.*)$|^no\s+(.+)$",
        raw, re.IGNORECASE)
    if m:
        spans = _quoted_spans(raw)
        value = spans[0] if spans else _bare_value(raw, [])
        return {"target": _target_words(_strip_quotes(raw), _words(value)),
                "predicate": "NOT_CONTAINS", "value": value, "raw": raw}
    content = set(_words(raw))
    if content and content <= (_GENERIC_CHANGE_WORDS | {"the", "a"}):
        return {"target": sorted(content), "predicate": "CHANGED",
                "value": "", "raw": raw}
    m = re.search(
        r"^(the|this|current)?\s*(page|stage|screen)\b\s*(is|shows?|"
        r"displays?)?\s*(.*)$"
        r"|\bon\b\s*(.+?)\s*\b(page|stage|screen)\b"
        r"|\b(page|stage|screen)\b\s*(is|shows?|visible|open|loaded|"
        r"displayed)\b",
        raw, re.IGNORECASE)
    if m:
        spans = _quoted_spans(raw)
        nums = _NUMBER_RE.findall(raw)
        caps = re.findall(r"\b([A-Z][A-Za-z]+)\b", raw)
        distinctive = spans + nums + [c for c in caps if len(c) > 2]
        value = " ".join(distinctive) if distinctive else raw.strip()
        return {"target": _target_words(_strip_quotes(raw), _words(value)),
                "predicate": "PAGE_IS", "value": value or raw, "raw": raw}
    # Unstructured phrase ("Status Started", "Rahul Sharma"): VISIBLE on
    # its content words -- the same semantics as the legacy keyword path.
    return {"target": _words(raw), "predicate": "VISIBLE", "value": raw,
            "raw": raw}


# -- DOM access ------------------------------------------------------------

def _field_index(dom_elements) -> List[Dict[str, Any]]:
    """Flatten DOM interactables to {ident: [words], value: str, el}."""
    from .stv import _content_words
    out = []
    for el in dom_elements or []:
        if not isinstance(el, dict):
            continue
        ident = " ".join(str(el.get(k, "") or "") for k in (
            "text", "ariaLabel", "placeholder", "name", "id"))
        itype = str(el.get("type", "") or "").lower()
        if itype in _MEANINGFUL_TYPES:
            ident += f" {itype}"
        out.append({"ident": _content_words(ident),
                    "value": str(el.get("value", "") or ""),
                    "tag": str(el.get("tag", "") or "").lower(),
                    "el": el})
    return out


def _parse_number(s: str) -> Optional[float]:
    try:
        cleaned = str(s).replace("$", "").replace(",", "").strip()
        m = _NUMBER_RE.search(cleaned)
        if not m:
            return None
        return float(m.group(0))
    except (TypeError, ValueError):
        return None


def _values_equal(a: str, b: str) -> bool:
    na, nb = _parse_number(a), _parse_number(b)
    if na is not None and nb is not None:
        return na == nb
    norm = lambda s: " ".join(str(s or "").lower().split())
    return bool(norm(a)) and norm(a) == norm(b)


def _pool_has(pool: str, value: str) -> bool:
    """Page-level fallback: quoted value verbatim, else all keywords."""
    from .stv import _content_words, _norm_text
    spans = _quoted_spans(value)
    if spans:
        ev = _norm_text(pool)
        return all(s.strip().lower() in ev for s in spans if s.strip())
    words = _content_words(value)
    if not words:
        return True
    ev = _norm_text(pool)
    return all(w in ev for w in words)


# -- verification -----------------------------------------------------------

def verify_item(norm: Optional[Dict[str, Any]], before_pool: str,
                after_pool: str,
                dom_elements) -> Tuple[bool, str]:
    """Verify one normalized expectation. DOM-first, page fallback.

    Returns (ok, evidence_note). ok=False covers both "predicate disproven"
    and "predicate unresolvable" (conservative: uncertainty never passes).
    """
    if norm is None:
        return False, "unparseable expectation: cannot verify"
    pred = norm.get("predicate")
    target = norm.get("target", []) or []
    value = str(norm.get("value", "") or "")
    fields = _field_index(dom_elements)
    matched = [f for f in fields
               if target and set(target) & set(f["ident"])]

    if pred == "CHANGED":
        ok = _norm_pool(before_pool) != _norm_pool(after_pool)
        return ok, ("page state changed"
                    if ok else "page state identical")

    if pred in ("VISIBLE", "EXISTS"):
        ok = _pool_has(after_pool, value)
        return ok, (f"page shows {value[:60]!r}"
                    if ok else f"page lacks {value[:60]!r}")

    if pred == "PAGE_IS":
        distinctive = _quoted_spans(value) + _NUMBER_RE.findall(value) + [
            c for c in re.findall(r"\b([A-Z][A-Za-z]+)\b", value)
            if len(c) > 2]
        probe = " ".join(distinctive) if distinctive else value
        ok = _pool_has(after_pool, probe)
        return ok, (f"page is {probe[:60]!r}"
                    if ok else f"page is not {probe[:60]!r}")

    if pred == "APPLIED":
        from .stv import _norm_text
        ev = _norm_text(after_pool)
        marked = ("appl" in ev) or ("discount" in ev)
        ok = bool(marked and _pool_has(after_pool, value))
        return ok, (f"{value[:60]!r} applied"
                    if ok else f"{value[:60]!r} not shown as applied")

    if pred == "NOT_CONTAINS":
        words = _words(value)
        if matched:
            for f in matched:
                hay = _pool_words(f["value"])
                if words and set(words) <= set(hay):
                    return False, (f"forbidden {value[:60]!r} present in "
                                   f"matched field")
            return True, f"{value[:60]!r} absent from matched field(s)"
        ev_words = _pool_words(after_pool)
        if words and set(words) <= set(ev_words):
            return False, f"forbidden {value[:60]!r} present on page"
        return True, f"{value[:60]!r} absent"

    if pred in ("GREATER_THAN", "LESS_THAN"):
        want = _parse_number(value)
        if want is None:
            return False, f"no numeric threshold in {value[:60]!r}"
        nums: List[float] = []
        for f in matched:
            n = _parse_number(f["value"])
            if n is not None:
                nums.append(n)
        if not nums:
            nums = _numbers_near(after_pool, target)
        if not nums:
            return False, "no resolvable number for comparison"
        good = (any(n > want for n in nums) if pred == "GREATER_THAN"
                else any(n < want for n in nums))
        op = ">" if pred == "GREATER_THAN" else "<"
        return good, (f"observed {nums} {op} {want}"
                      if good else f"nothing {op} {want} in {nums}")

    # EQUALS / CONTAINS / SELECTED: DOM-first on matched fields.
    if pred in ("EQUALS", "CONTAINS", "SELECTED"):
        if matched:
            for f in matched:
                if pred == "EQUALS" and _values_equal(f["value"], value):
                    return True, (f"DOM field == {value[:60]!r}")
                if pred == "CONTAINS" and _pool_has(f["value"], value):
                    return True, (f"DOM field contains {value[:60]!r}")
                if pred == "SELECTED" and _selected_holds(f, value):
                    return True, (f"DOM field has {value[:60]!r} selected")
            return False, (f"matched field(s) do not satisfy "
                           f"{pred} {value[:60]!r}")
        # No field names the target: page-level fallback (legacy parity),
        # except bare numbers, which match anywhere ("3" hits "T3:"). A
        # numeric EQUALS needs its number near a target word.
        if pred == "EQUALS":
            want = _parse_number(value)
            if want is not None and target:
                nums = _numbers_near(after_pool, target)
                good = any(n == want for n in nums)
                return good, (f"page shows {want:g} near target"
                               if good else
                               f"no {want:g} near target words")
            ok = _values_equal(after_pool.strip(), value) or _pool_has(
                after_pool, value)
            return ok, ("page satisfies EQUALS " + value[:60]
                        if ok else "page lacks " + value[:60])
        ok = _pool_has(after_pool, value)
        return ok, (f"page shows {value[:60]!r}"
                    if ok else f"page lacks {value[:60]!r}")

    return False, f"unknown predicate {pred!r}"


def _selected_holds(field: Dict[str, Any], value: str) -> bool:
    """A select/radio/checkbox field has `value` selected."""
    tag, el = field.get("tag", ""), field.get("el", {})
    val = str(field.get("value", "") or "")
    if tag == "select" or str(el.get("type", "")).lower() in (
            "select-one", "select-multiple"):
        return _values_equal(val, value)
    if tag == "input" and str(el.get("type", "")).lower() in (
            "radio", "checkbox"):
        return bool(el.get("checked")) or _values_equal(val, value)
    return _values_equal(val, value)


def _numbers_near(pool: str, target: List[str],
                  window: int = 60) -> List[float]:
    """Numbers occurring within `window` chars of a target word."""
    from .stv import _norm_text
    text = _norm_text(pool)
    out: List[float] = []
    for w in target or []:
        for m in re.finditer(re.escape(w), text):
            snippet = text[max(0, m.start() - window):m.end() + window]
            for nm in _NUMBER_RE.findall(snippet):
                try:
                    out.append(float(nm.replace(",", "")))
                except ValueError:
                    pass
    return out


def _pool_words(pool: str) -> List[str]:
    from .stv import _norm_text
    return _norm_text(pool).split()


def _norm_pool(pool: str) -> str:
    from .stv import _norm_text
    return _norm_text(pool)
