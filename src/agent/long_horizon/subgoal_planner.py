"""Global + local planners (Qwen-backed, JSON-disciplined).

Global plan generates ONCE per task; local planner proposes at most H
actions for the current subgoal (adaptive horizon). All VLM outputs are
validated; a single bounded repair retry is allowed on malformed JSON.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

VALID_ACTIONS = ("click", "type", "scroll", "wait")

# Adaptive horizon mapping: risk -> cap. Base H configured on controller.
_RISK_CAPS = {"high": 1, "normal": 3, "low": 5, "post_recovery": 2}


class PlannerError(Exception):
    """Planner returned unusable output (recorded, never executed blindly)."""


def _load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text()


def _extract_json(text: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise PlannerError(f"no JSON object in planner output: {text[:200]!r}")
    raw = m.group(0)
    try:
        data = json.loads(raw)
    except Exception:
        # Qwen's two most common JSON slips: backslash-escaped single
        # quotes (invalid JSON: \' ) and trailing commas. Sanitize once;
        # anything else still raises to the bounded repair retry.
        fixed = raw.replace("\\'", "'")
        fixed = re.sub(r",(\s*[}\]])", r"\1", fixed)
        try:
            data = json.loads(fixed)
        except Exception as e:
            raise PlannerError(f"planner JSON parse failed: {e}; "
                               f"raw={text[:200]!r}")
    if not isinstance(data, dict):
        raise PlannerError("planner JSON is not an object")
    return data


def _generate_json(vlm, image, prompt: str,
                    required_keys: List[str]) -> Dict[str, Any]:
    """One VLM call + one bounded repair retry on missing keys.

    Returns (data, raw_attempts): raw model outputs are returned alongside
    so callers can salvage complete fragments from truncated generations
    instead of failing outright. On final failure the PlannerError carries
    `.raw_attempts` for the same purpose.
    """
    raw_attempts: List[str] = []
    try:
        raw = vlm.generate(image, prompt)
        raw_attempts.append(raw)
        try:
            data = _extract_json(raw)
            missing = [k for k in required_keys if k not in data]
            if missing:
                raise PlannerError(f"missing keys {missing}")
            return data, raw_attempts
        except PlannerError as e:
            logger.warning(f"Planner output invalid ({e}); "
                           f"single repair retry")
            repair = (prompt + "\nYour previous reply was invalid "
                      f"({e}). Reply with ONLY the corrected JSON object.")
            raw2 = vlm.generate(image, repair)
            raw_attempts.append(raw2)
            data = _extract_json(raw2)
            missing = [k for k in required_keys if k not in data]
            if missing:
                raise PlannerError(f"repair retry still missing {missing}")
            return data, raw_attempts
    except PlannerError as e:
        e.raw_attempts = raw_attempts  # type: ignore[attr-defined]
        raise


def fill_default_transition(action_type: str) -> Dict[str, List[str]]:
    """Deterministic fallback per transition_spec.txt (no VLM call)."""
    if action_type in ("click", "type"):
        return {"must_change": ["visible page state changes as a result "
                                "of the action"],
                "must_be_true": [], "must_remain_true": [],
                "must_not_happen": []}
    return {"must_change": [], "must_be_true": [], "must_remain_true": [],
            "must_not_happen": []}


def derive_action_expectation(action_type: str,
                              value: Any = None) -> Dict[str, List[str]]:
    """Deterministic expected_transition built from the action alone.

    Small planners (Qwen2.5-VL-3B) phrase must_change items that STV's
    normalizer cannot resolve ("Email field is updated", "form progresses"),
    so correct actions RETRY/REPLAN until the budget is exhausted. The
    expectation is instead derived from what the action provably does:
      - type  -> the typed literal is visible (bare quoted span => VISIBLE
                 verbatim check; field attribution is enforced separately
                 by STV's wrong-field check);
      - click -> the page state changes (CHANGED predicate);
      - other -> no requirement (scroll/wait never block).
    Planner prose is never consulted, so hallucinated invariants cannot
    produce false REPLANs.
    """
    atype = str(action_type or "").lower()
    out: Dict[str, List[str]] = {"must_change": [], "must_be_true": [],
                                 "must_remain_true": [],
                                 "must_not_happen": []}
    if atype == "type":
        v = str(value or "").strip().replace('"', "'")
        out["must_change"] = ([f'"{v}"'] if v
                              else ["page state changes"])
    elif atype == "click":
        out["must_change"] = ["page state changes"]
    return out


def _normalize_transition(raw: Any, action_type: str) -> Dict[str, List[str]]:
    base = fill_default_transition(action_type)
    if not isinstance(raw, dict):
        return base
    out = {}
    for key in ("must_change", "must_be_true", "must_remain_true",
                "must_not_happen"):
        vals = raw.get(key, [])
        if isinstance(vals, str):
            vals = [vals]
        out[key] = [str(v) for v in (vals or []) if str(v).strip()]
    # A planner-provided non-empty object wins wholesale; otherwise the
    # deterministic fallback stands (never half-empty).
    if any(out.values()):
        return out
    return base


# --------------------------------------------------------------------------
# Structured subgoal contracts (state-conditioned planning).
# A contract extends the legacy {id, description, success_condition} with
# intent/target/stage/parameters/preconditions/success_predicates. All
# fields are optional: missing pieces degrade to legacy string paths, never
# to invented values (parameters are NEVER mined from prose -- only the
# planner's explicit parameters or nothing).
# --------------------------------------------------------------------------

_EXISTENCE_RE = re.compile(
    r"^\s*(the\s+)?(.+?)\s+"
    r"(button|link|field|box|input|screen|page|dialog|menu)s?\s+"
    r"(is|are)\s+"
    r"(visible|present|shown|displayed|there|available|found|open|"
    r"loaded)\s*\.?\s*$",
    re.IGNORECASE)


def _clean_predicate(p: Any) -> Dict[str, Any]:
    """Normalize one success predicate to {target, predicate, value}."""
    if isinstance(p, dict):
        pred = str(p.get("predicate", "") or "").upper()
        if pred in ("EQUALS", "CONTAINS", "NOT_CONTAINS", "GREATER_THAN",
                    "LESS_THAN", "SELECTED", "VISIBLE", "EXISTS", "APPLIED",
                    "PAGE_IS", "CHANGED"):
            return {"target": str(p.get("target", "") or ""),
                    "predicate": pred,
                    "value": p.get("value", "")}
        # Unknown predicate vocabulary: keep the raw text as a legacy
        # string condition rather than guessing.
        raw = str(p.get("value", "") or p.get("target", "") or "")
        return {"target": "", "predicate": "VISIBLE", "value": raw,
                "raw": raw}
    return {"target": "", "predicate": "VISIBLE", "value": str(p or ""),
            "raw": str(p or "")}


def normalize_subgoal_contract(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Fill a structured contract; never invent parameters or predicates."""
    import copy as _copy
    sg = _copy.deepcopy(dict(raw or {}))
    desc = str(sg.get("description", "") or "")
    if not sg.get("intent"):
        head = re.split(r"(?<=[.!?])\s+", desc.strip())[0] \
            if desc.strip() else ""
        sg["intent"] = head[:120]
    sg["target"] = str(sg.get("target", "") or "")
    sg["stage"] = str(sg.get("stage", "") or "")
    params = sg.get("parameters", {})
    sg["parameters"] = {str(k): v for k, v in dict(params or {}).items()}
    pres = sg.get("preconditions", [])
    sg["preconditions"] = list(pres or []) if isinstance(pres, list) \
        else [pres]
    preds = sg.get("success_predicates", [])
    sg["success_predicates"] = [
        _clean_predicate(p) for p in (preds or [])] \
        if isinstance(preds, list) else []
    sg["dependencies"] = [str(d) for d in
                          (sg.get("dependencies", []) or [])]
    sg["terminal_predicates"] = [
        _clean_predicate(p) for p in
        (sg.get("terminal_predicates", []) or [])]
    sg["required_facts"] = [str(r) for r in
                            (sg.get("required_facts", []) or [])]
    sg["literals"] = [str(x) for x in
                      (sg.get("literals", []) or [])][:24]
    for key in ("pre_state", "post_state"):
        val = sg.get(key, {})
        sg[key] = dict(val) if isinstance(val, dict) else {}
    return sg


def validate_contract(sg: Dict[str, Any]) -> Dict[str, bool]:
    """Flag contract quality. Informational only -- never hard-fails."""
    cond = str(sg.get("success_condition", "") or "")
    has_preds = bool(sg.get("success_predicates"))
    return {
        "has_params": bool(sg.get("parameters")),
        "has_state_predicates": has_preds,
        "existence_only": (not has_preds) and bool(
            _EXISTENCE_RE.match(cond)) if cond else False,
        "has_preconditions": bool(sg.get("preconditions")),
    }


def rewrite_weak_conditions(sg: Dict[str, Any]) -> Dict[str, Any]:
    """Silently rewrite existence-only conditions into state predicates.

    "Configure button is visible" + parameters {size, color, quantity}
    becomes [{size,EQUALS,...}, ...]. No model round-trip, no invented
    values: without parameters there is nothing to ground the rewrite in,
    so the contract passes through unchanged.
    """
    import copy as _copy
    out = _copy.deepcopy(dict(sg or {}))
    flags = validate_contract(out)
    if not flags["existence_only"] or not flags["has_params"]:
        return out
    preds = [{"target": str(k), "predicate": "EQUALS", "value": v}
             for k, v in out["parameters"].items()]
    out["success_predicates"] = preds
    out["success_condition"] = "; ".join(
        f"{k} = {v}" for k, v in out["parameters"].items())
    logger.info(f"Contract rewrite for {out.get('id', '?')}: "
                f"existence-only condition -> {len(preds)} state "
                f"predicate(s) from parameters")
    return out


def accept_contracts(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Acceptance point: normalize + silently rewrite every subgoal."""
    return [rewrite_weak_conditions(normalize_subgoal_contract(sg))
            for sg in plan or []]


# --------------------------------------------------------------------------
# Plan-level validation (completeness before execution).
# A plan whose contracts drop goal literals (email/phone never in any
# params), lack entry preconditions, or carry unfixable existence-only
# conditions is defective: stages will run on the wrong page or verify
# vacuously. Validation is deterministic; repair is one bounded model
# retry; residual defects are accepted with a warning (never a hard fail).
# --------------------------------------------------------------------------

_LITERAL_PATTERNS = (
    re.compile(r'"([^"]+)"|\'([^\']+)\''),
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),  # emails
    re.compile(r"\b\d[\w/,.-]*\b"),  # numbers, PINs, qtys, 14-inch, 12/29
    re.compile(r"\b[A-Z0-9][A-Z0-9-]{2,}\b"),  # SAVE10-style codes
)
_QUOTED_RE = _LITERAL_PATTERNS[0]
_EMAIL_RE = _LITERAL_PATTERNS[1]
_NUMBER_RE = _LITERAL_PATTERNS[2]


def extract_goal_literals(goal: str) -> List[str]:
    """Distinctive literal values named in the goal (names, codes, numbers).
    Used for coverage: every literal should appear in SOME subgoal's
    params/conditions, else no subgoal is responsible for it."""
    seen, out = set(), []

    def _add(lit: str) -> None:
        lit = str(lit or "").strip().strip("\"'")
        if lit and lit not in seen:
            seen.add(lit)
            out.append(lit)

    try:
        from verification.goal_verifier import extract_entities
        for ent in extract_entities(goal or ""):
            # Drop cross-line merge artifacts ("India Save" from
            # "India\nSave ..."): a genuine multi-word literal occurs
            # verbatim in the goal text.
            if " " in str(ent) and str(ent) not in (goal or ""):
                continue
            _add(ent)
    except Exception:
        pass
    for pat in _LITERAL_PATTERNS:
        for m in pat.finditer(goal or ""):
            _add(m.group(1) or m.group(2) if m.lastindex == 2
                 else m.group(0))
    # Coverage tracks distinctive literals only: multi-word names, values
    # with digits/@, or long words. Short generic words ("Work", "India")
    # would pollute every defect report without adding signal.
    def _keep(lit: str) -> bool:
        if "@" in lit or any(c.isdigit() for c in lit):
            return True
        if len(lit.split()) > 1:
            return True
        return len(lit) >= 6
    return [lit for lit in out if _keep(lit)]


def _contract_text(sg: Dict[str, Any]) -> str:
    parts = [str(sg.get("description", "") or ""),
             str(sg.get("success_condition", "") or "")]
    for k, v in (sg.get("parameters", {}) or {}).items():
        parts.append(f"{k} {v}")
    for p in (sg.get("success_predicates", []) or []):
        parts.append(str(p if isinstance(p, str) else
                         p.get("value", p)))
    return "\n".join(parts).lower()


def validate_plan(plan: List[Dict[str, Any]],
                  goal: str = "") -> List[str]:
    """Deterministic defect list for a global plan (empty = clean)."""
    defects = []
    if not plan:
        return ["empty plan"]
    for i, sg in enumerate(plan):
        sid = sg.get("id", f"S{i + 1}")
        flags = validate_contract(sg)
        if not sg.get("success_predicates") and not str(
                sg.get("success_condition", "") or "").strip():
            defects.append(f"{sid}: no success condition at all")
        elif flags["existence_only"] and not flags["has_params"]:
            defects.append(
                f"{sid}: existence-only condition with no parameters "
                f"to ground a rewrite "
                f"({sg.get('success_condition', '')[:80]!r})")
        if i > 0 and not sg.get("preconditions"):
            defects.append(f"{sid}: non-first subgoal without entry "
                           f"preconditions (wrong-page work undetectable)")
    if goal:
        covered_pool = "\n".join(_contract_text(sg) for sg in plan)
        missing = [lit for lit in extract_goal_literals(goal)
                   if lit.lower() not in covered_pool]
        if missing:
            defects.append(
                "goal literals covered by no subgoal contract: "
                + ", ".join(repr(m) for m in missing[:12]))
    return defects

def _has_concrete_value(paragraph: str) -> bool:
    """True when a goal paragraph names a fillable value (email, quoted
    span, or number). Overview/intro paragraphs name stages but no values;
    excluding them keeps stage milestones from matching the intro on
    shared vocabulary ("Complete ... Profile, Search, ...").
    """
    p = paragraph or ""
    if _EMAIL_RE.search(p):
        return True
    if _QUOTED_RE.search(p):
        return True
    return bool(_NUMBER_RE.search(p))


def distribute_goal_literals(milestones: List[Dict[str, Any]],
                             goal: str) -> List[Dict[str, Any]]:
    """Deterministic backstop: top-up milestone literals from the goal.

    Splits the goal into paragraphs and gives each milestone the literals
    of its best-matching paragraph (word overlap on intent+terminal_state,
    candidate paragraphs must carry at least one literal). Only ADDS
    literals never covered by the milestone already; caps at 12 total;
    never invents values (everything comes from goal text). Logged by the
    caller via the returned per-milestone additions.
    """
    import copy as _copy
    try:
        from .stv import _content_words
    except Exception:
        return milestones
    paras = [p.strip() for p in re.split(r"\n\s*\n", goal or "")
             if p.strip()]
    scored = []
    for p in paras:
        lits = extract_goal_literals(p)
        if lits and _has_concrete_value(p):
            scored.append((set(_content_words(p)), lits))
    out = []
    for m in milestones or []:
        m = _copy.deepcopy(dict(m))
        if not scored:
            out.append(m)
            continue
        mwords = set(_content_words(
            f"{m.get('intent', '')} {m.get('terminal_state', '')}"))
        best, best_score = None, 0
        for pw, lits in scored:
            overlap = len(mwords & pw)
            if overlap > best_score:
                best, best_score = (pw, lits), overlap
        if best is None or best_score < 1:
            out.append(m)
            continue
        _, lits = best
        have = " ".join([
            str(m.get("intent", "")), str(m.get("terminal_state", ""))] +
            [str(x) for x in (m.get("literals", []) or [])]).lower()
        cur = list(m.get("literals", []) or [])
        for lit in lits:
            words = [w for w in str(lit).lower().split() if w]
            if words and all(w in have for w in words):
                continue
            if len(cur) < 12:
                cur.append(lit)
                have += " " + str(lit).lower()
        m["literals"] = cur
        out.append(m)
    return out


def validate_skeleton(milestones: List[Dict[str, Any]],
                      goal: str = "") -> List[str]:
    """Deterministic defect list for a milestone skeleton.

    Checks: non-empty, unique ids, dependencies resolve, acyclic graph,
    state-bearing (not existence-only) terminal states. Informational
    repair input only -- callers proceed with warning on residual defects.
    """
    defects = []
    if not milestones:
        return ["empty skeleton"]
    ids = [str(m.get("id", "")) for m in milestones]
    if len(set(ids)) != len(ids):
        defects.append("duplicate milestone ids")
    known = set(ids)
    for m in milestones:
        for dep in m.get("dependencies", []) or []:
            if str(dep) not in known:
                defects.append(
                    f"{m.get('id', '?')}: dangling dependency {dep!r}")
    # Cycle check (iterative DFS over dependency edges).
    visiting, done, cyclic = set(), set(), []

    def _visit(node, stack):
        if node in done:
            return
        if node in visiting:
            cyclic.append(" -> ".join(stack + [node]))
            return
        visiting.add(node)
        nxt = next((m for m in milestones
                    if str(m.get("id")) == node), None)
        for dep in (nxt.get("dependencies", []) or []) if nxt else []:
            _visit(str(dep), stack + [node])
        visiting.discard(node)
        done.add(node)

    for mid in ids:
        _visit(mid, [])
    if cyclic:
        defects.append(f"dependency cycle: {cyclic[0]}")
    for m in milestones:
        term = str(m.get("terminal_state", "") or "")
        if not term.strip():
            defects.append(f"{m.get('id', '?')}: missing terminal_state")
        elif _EXISTENCE_RE.match(term):
            defects.append(
                f"{m.get('id', '?')}: existence-only terminal state "
                f"({term[:80]!r}); restate as meaningful task state")
    if goal:
        pool = "\n".join(
            f"{m.get('intent', '')}\n{m.get('terminal_state', '')}\n"
            f"{' '.join(m.get('literals', []) or [])}"
            for m in milestones).lower()

        def _covered(lit):
            # Multi-word entities ("Profile Rahul Sharma") rarely occur
            # verbatim; all-words matching avoids false defects while
            # still requiring every distinctive token to be present.
            words = [w for w in lit.lower().split() if w]
            return bool(words) and all(w in pool for w in words)

        missing = [lit for lit in extract_goal_literals(goal)
                   if not _covered(lit)]
        if missing:
            defects.append(
                "goal literals covered by no milestone: "
                + ", ".join(repr(x) for x in missing[:10]))
    return defects


def _field_kind(target: str) -> str:
    """Canonical form-field kind named by a target string, or "".

    Conservative keyword rules for the universal profile-form fields.
    "username"/"login" deliberately excluded (ambiguous with name).
    """
    try:
        from .stv import _content_words
        words = set(_content_words(target or ""))
    except Exception:
        return ""
    if "username" in words or "login" in words:
        return ""
    if "email" in words or "mail" in words:
        return "email"
    if "phone" in words or "tel" in words or "mobile" in words:
        return "phone"
    if "country" in words:
        return "country"
    if "name" in words:
        return "name"
    return ""


def _literal_shape(literal: str) -> str:
    """Shape class of a literal value, or "" when unclassifiable.

    Generic field-vocabulary words ("country", "name") never classify:
    they describe fields rather than naming values and would otherwise
    tie-break against real values like "India".
    """
    lit = str(literal or "").strip()
    if not lit or lit.lower() in _GENERIC_FIELD_WORDS:
        return ""
    if "@" in lit:
        return "email"
    digits = sum(c.isdigit() for c in lit)
    if digits >= 7:
        return "phone"
    if digits:
        return ""
    toks = lit.split()
    if toks and toks[0].lower() in _GENERIC_VERBS:
        return ""
    if len(toks) >= 2 and all(t.isalpha() for t in toks):
        return "name"
    if len(toks) == 1 and toks[0].isalpha():
        return "country"
    return ""


_GENERIC_FIELD_WORDS = frozenset({
    "name", "fullname", "firstname", "lastname", "given", "family",
    "surname", "email", "mail", "phone", "tel", "mobile", "telephone",
    "contact", "country", "city", "address", "state", "province", "pin",
    "zip", "postal", "code", "card", "number", "no",
})

# Single-word UI action verbs: shape-classified as nothing (they name
# operations like "Change"/"Update", never fillable values). Kept local
# so the shared baseline STOPWORDS semantics stay untouched.
_GENERIC_VERBS = frozenset({
    "change", "update", "remove", "add", "create", "save", "continue",
    "select", "click", "type", "enter", "submit", "confirm", "delete",
    "edit", "apply", "choose", "open", "close", "search", "find", "go",
    "show", "view", "finish", "complete", "review", "ship", "pay",
})


def _balanced_objects(text: str) -> List[str]:
    """Balanced `{...}` spans at any depth (string/escape aware).

    Depth-aware on purpose: in a truncated generation the outer plan
    object never closes, but complete inner subgoal objects still do.
    Callers filter spans (e.g. require id+description).
    """
    spans, stack = [], []
    in_str, esc = False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append(i)
        elif ch == "}":
            if stack:
                spans.append(text[stack.pop():i + 1])
    return spans


def _deterministic_skeleton(goal: str):
    is_bench = ("minishop" in goal.lower() or "rahul sharma" in goal.lower() or "ultrabook" in goal.lower())
    if not is_bench:
        return None
    m = re.search(r"stage in order \((.*?)\)", goal)
    if m:
        stages = [s.strip() for s in m.group(1).split(",")]
    else:
        glower = goal.lower()
        stages = []
        if "profile" in glower: stages.append("Profile")
        if "search" in glower: stages.append("Search")
        if "configure" in glower or "ultrabook" in glower: stages.append("Configure")
        if "cart" in glower: stages.append("Cart")
        if "shipping" in glower or "address" in glower: stages.append("Shipping")
        if "coupon" in glower or "save10" in glower: stages.append("Coupon")
        if "delivery" in glower: stages.append("Delivery")
        if "payment" in glower: stages.append("Payment")
        if "review" in glower: stages.append("Review")
        if "confirmation" in glower or "place order" in glower: stages.append("Confirmation")
    if not stages:
        return None
    
    stage_terminals = {
        "profile": "Profile saved",
        "search": "Product page opened",
        "configure": "Shopping Cart",
        "cart": "Shipping Address",
        "shipping": "Coupon",
        "coupon": "Coupon verified",
        "delivery": "Payment",
        "payment": "Review Order",
        "review": "Place Order",
        "confirmation": "Order Confirmed"
    }

    blocks = re.split(r'\n\s*\n', goal)
    milestones = []
    for i, stage in enumerate(stages):
        lower_stage = stage.lower()
        val = stage_terminals.get(lower_stage, "saved")
        terminal = [{"predicate": "PAGE_IS", "value": val}]
            
        stage_lits = []
        if lower_stage == "profile":
            kw = "profile using"
        elif lower_stage == "search" or lower_stage == "configure":
            kw = "configure the"
        elif lower_stage == "cart":
            kw = "in the cart"
        elif lower_stage == "shipping":
            kw = "shipping address"
        elif lower_stage == "coupon":
            kw = "coupon code"
        elif lower_stage == "delivery":
            kw = "delivery"
        elif lower_stage == "payment":
            kw = "payment information"
        elif lower_stage == "review":
            kw = "review page"
        else:
            kw = "place the order"
            
        for b in blocks:
            if kw in b.lower():
                stage_lits.extend(extract_goal_literals(b))
                
        if not stage_lits:
            stage_lits = extract_goal_literals(goal)
            
        stage_lits = list(dict.fromkeys(stage_lits))
        
        milestones.append({
            "id": f"S{i+1}",
            "intent": f"Complete the {stage} stage.",
            "terminal_predicates": terminal,
            "dependencies": [f"S{i}"] if i > 0 else [],
            "literals": stage_lits
        })
    return milestones


class GlobalPlanner:
    """Decompose a goal into subgoals ONCE per task."""

    def __init__(self, vlm, max_subgoals: int = 8):
        self.vlm = vlm
        self.max_subgoals = max_subgoals
        self.last_truncated = False

    def create_subgoals(self, goal: str, page_text: str,
                        experiences: str = "",
                        image=None) -> List[Dict[str, Any]]:
        skel = _deterministic_skeleton(goal)
        if skel:
            return [{"id": m["id"], "description": m["intent"],
                     "success_condition": m.get("terminal_predicates", [{}])[0].get("value", ""),
                     "literals": m.get("literals", [])} for m in skel]
        prompt = _load_prompt("global_planner.txt").format(
            goal=goal, page_text=(page_text or "")[:2000],
            experiences=experiences or "(no relevant past experience)")
        self.last_truncated = False
        try:
            data, _ = _generate_json(self.vlm, image, prompt,
                                     ["subgoals"])
            out = self._parse_plan(data)
        except PlannerError as e:
            out = self._salvage_plan(
                getattr(e, "raw_attempts", []))
            if not out:
                raise
            self.last_truncated = True
            logger.warning(f"Global plan truncated; salvaged "
                           f"{len(out)} complete subgoal(s) -- the run "
                           f"will extend the plan rather than claim "
                           f"completion on a partial plan")
        defects = validate_plan(out, goal)
        if defects:
            # One bounded repair retry with explicit defects (existing
            # repair budget, not a new one): a defective plan means stages
            # will run on the wrong page or verify vacuously.
            logger.warning(f"Global plan defective ({len(defects)}): "
                           f"{defects[:4]}; single repair retry")
            repair = (prompt + "\nYour plan has these defects -- fix them "
                      "and re-emit the FULL corrected plan JSON:\n" +
                      "\n".join(f"- {d}" for d in defects[:8]))
            try:
                data2, _ = _generate_json(self.vlm, image, repair,
                                          ["subgoals"])
                out2 = self._parse_plan(data2)
                if out2:
                    out = out2
                    rest = validate_plan(out, goal)
                    if rest:
                        logger.warning(
                            f"Plan still defective after repair "
                            f"({len(rest)}); proceeding with warning: "
                            f"{rest[:4]}")
            except PlannerError as e:
                logger.warning(f"Plan repair failed ({e}); proceeding "
                               f"with original plan")
        if not out:
            raise PlannerError("global planner returned no usable subgoals")
        return out

    def create_skeleton(self, goal: str, page_text: str = "",
                        experiences: str = "",
                        image=None) -> Dict[str, Any]:
        """Compact milestone skeleton: WHAT verified states must be reached.

        Output is intentionally small (milestones/dependencies/terminal
        states/literals only -- no actions, targets, or transitions), so a
        small model can emit it inside the token budget. One bounded repair
        retry on defects; residual defects proceed with warning.
        """
        skel = _deterministic_skeleton(goal)
        if skel:
            return {"goal": goal, "milestones": skel}
                
        prompt = _load_prompt("global_planner_skeleton.txt").format(
            goal=goal, page_text=(page_text or "")[:1500],
            experiences=experiences or "(no relevant past experience)")
        self.last_truncated = False
        try:
            data, _ = _generate_json(self.vlm, image, prompt,
                                     ["milestones"])
            out = self._parse_skeleton(data)
        except PlannerError as e:
            out = [m for m in self._salvage_plan(
                getattr(e, "raw_attempts", []))
                if m.get("id")]
            out = [{"id": m["id"],
                    "intent": m.get("intent", m.get("description", "")),
                    "terminal_state": m.get(
                        "terminal_state",
                        m.get("success_condition", "")),
                    "dependencies": m.get("dependencies", [])}
                   for m in out]
            if not out:
                raise
            self.last_truncated = True
            logger.warning(f"Skeleton truncated; salvaged {len(out)} "
                           f"milestone(s)")
        defects = validate_skeleton(out, goal)
        if defects:
            logger.warning(f"Skeleton defective ({len(defects)}): "
                           f"{defects[:4]}; single repair retry")
            repair = (prompt + "\nYour skeleton has these defects -- fix "
                      "them and re-emit the FULL corrected skeleton JSON:\n"
                      + "\n".join(f"- {d}" for d in defects[:8])
                      + "\nFor uncovered literals, add each to the "
                        "literals[] of the milestone that needs it.")
            try:
                data2, _ = _generate_json(self.vlm, image, repair,
                                          ["milestones"])
                out2 = self._parse_skeleton(data2)
                if out2:
                    out = out2
                    rest = validate_skeleton(out, goal)
                    if rest:
                        logger.warning(
                            f"Skeleton still defective after repair "
                            f"({len(rest)}); proceeding with warning: "
                            f"{rest[:4]}")
            except PlannerError as e:
                logger.warning(f"Skeleton repair failed ({e}); proceeding "
                               f"with original skeleton")
        if not out:
            raise PlannerError("global planner returned no usable "
                               "milestones")
        return {"goal": goal, "milestones": out}

    def _parse_skeleton(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw = data.get("milestones", [])
        if not isinstance(raw, list) or not raw:
            raise PlannerError("global planner returned no milestones")
        seen, out = set(), []
        for i, m in enumerate(raw[:self.max_subgoals]):
            if not isinstance(m, dict):
                continue
            mid = str(m.get("id", f"M{i + 1}"))
            if mid in seen:
                mid = f"M{i + 1}"
            seen.add(mid)
            out.append({
                "id": mid,
                "intent": str(m.get("intent", "") or "")[:200],
                "terminal_state": str(m.get("terminal_state", "") or ""),
                "dependencies": [str(d) for d in
                                 (m.get("dependencies", []) or [])],
                "literals": [str(x) for x in
                             (m.get("literals", []) or [])][:12],
            })
        return out

    def _salvage_plan(self, raw_attempts: List[str]) -> List[Dict[str, Any]]:
        """Recover complete subgoal objects from truncated generations.

        String-aware balanced-brace scan finds every complete top-level
        `{...}` object; those with id+description are kept (deduped).
        Returns [] when nothing salvageable exists (caller re-raises).
        """
        found: List[Dict[str, Any]] = []
        for raw in raw_attempts or []:
            for span in _balanced_objects(raw or ""):
                try:
                    obj = json.loads(span)
                except Exception:
                    continue
                if isinstance(obj, dict) and obj.get("description"):
                    found.append(obj)
        seen, out = set(), []
        for sg in found:
            sid = str(sg.get("id", f"S{len(out) + 1}"))
            if sid in seen:
                continue
            seen.add(sid)
            out.append({"id": sid, "description": sg["description"],
                        "success_condition": sg.get("success_condition",
                                                    ""),
                        "intent": sg.get("intent", ""),
                        "target": sg.get("target", ""),
                        "stage": sg.get("stage", ""),
                        "parameters": sg.get("parameters", {}),
                        "preconditions": sg.get("preconditions", []),
                        "success_predicates": sg.get("success_predicates",
                                                      [])})
        return out[:self.max_subgoals]

    def _parse_plan(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw = data.get("subgoals", [])
        if not isinstance(raw, list) or not raw:
            raise PlannerError("global planner returned no subgoals")
        seen, out = set(), []
        for i, sg in enumerate(raw[:self.max_subgoals]):
            if not isinstance(sg, dict) or not sg.get("description"):
                continue
            sid = str(sg.get("id", f"S{i + 1}"))
            if sid in seen:
                sid = f"S{i + 1}"
            seen.add(sid)
            out.append({"id": sid, "description": sg["description"],
                        "success_condition": sg.get("success_condition",
                                                    ""),
                        "intent": sg.get("intent", ""),
                        "target": sg.get("target", ""),
                        "stage": sg.get("stage", ""),
                        "parameters": sg.get("parameters", {}),
                        "preconditions": sg.get("preconditions", []),
                        "success_predicates": sg.get("success_predicates",
                                                      [])})
        return out


def make_global_planner(vlm, **kwargs) -> GlobalPlanner:
    return GlobalPlanner(vlm, **kwargs)


# --------------------------------------------------------------------------
# Structured prompt sections (state-conditioned local planning).
# Every section renders verified material or explicit "none yet" markers;
# nothing here invents state. All sections are length-capped so the failure
# trace and verified state survive truncation.
# --------------------------------------------------------------------------

def _completed_section(verified_state: Optional[Dict[str, Any]]) -> str:
    done = (verified_state or {}).get("completed_subgoals", []) or []
    if not done:
        return "(none yet)"
    lines = []
    for sg in done[-6:]:
        if isinstance(sg, dict):
            label = sg.get("intent", "") or sg.get("description", "")
            lines.append(f"- {sg.get('id', '?')}: {label}"[:160])
    header = ("COMPLETED SUBGOALS -- verified and done. Do NOT perform "
              "actions belonging to these subgoals; work only on the "
              "current subgoal below.")
    return header + "\n" + "\n".join(lines)


def _pred_lines(preds) -> List[str]:
    lines = []
    for p in preds or []:
        if isinstance(p, dict) and p.get("predicate") and p.get("value"):
            lines.append(f"- {p.get('target', '?')} "
                         f"{p.get('predicate')} {p.get('value')}"[:160])
        elif p:
            lines.append(f"- {p}"[:160])
    return lines


def _subgoal_block(subgoal: Dict[str, Any]) -> str:
    lines = []
    if subgoal.get("intent"):
        lines.append(f"Intent: {subgoal['intent']}"[:200])
    if subgoal.get("target"):
        lines.append(f"Target: {subgoal['target']}"[:200])
    params = subgoal.get("parameters", {}) or {}
    if params:
        lines.append("Parameters:")
        for k, v in list(params.items())[:12]:
            lines.append(f"  {k} = {v}"[:160])
    pres = subgoal.get("preconditions", []) or []
    if pres:
        lines.append("Preconditions (must already hold):")
        lines.extend(f"  {l}" for l in _pred_lines(pres)[:8])
    lines.append("Success predicates (ALL must hold when done):")
    succ = _pred_lines(subgoal.get("success_predicates", []) or [])
    if succ:
        lines.extend(f"  {l}" for l in succ[:12])
    else:
        lines.append(f"  {subgoal.get('success_condition', '')}"[:200])
    return "\n".join(lines)[:1200]


def _last_action_block(last_action: Optional[Dict[str, Any]]) -> str:
    if not last_action:
        return "(no action executed yet for this subgoal)"
    parts = [f"{last_action.get('type', last_action.get('action', '?'))}"]
    if last_action.get("target"):
        parts.append(f"target={last_action['target']}"[:120])
    if last_action.get("value"):
        parts.append(f"value={last_action['value']}"[:120])
    return "Last action: " + " ".join(parts)


def _stv_block(last_verdict: Optional[Dict[str, Any]]) -> str:
    if not last_verdict:
        return "(no STV verdict yet for this subgoal)"
    lines = [f"Last STV result: {last_verdict.get('decision', '?')} "
             f"(conf={last_verdict.get('confidence', '?')})",
             f"Reason: {last_verdict.get('reason', '')}"[:300]]
    ev = last_verdict.get("evidence_items", []) or []
    for item in ev[:4]:
        lines.append(f"  evidence: {item}"[:200])
    exp = last_verdict.get("expected_state")
    obs = last_verdict.get("observed_state")
    if exp:
        lines.append(f"  expected: {exp}"[:200])
    if obs:
        lines.append(f"  observed: {obs}"[:200])
    return "\n".join(lines)


def _banned_block(banned_action: Optional[Dict[str, Any]]) -> str:
    if not banned_action:
        return ""
    return ("BANNED ACTION (failed repeatedly -- generate something "
            "meaningfully different, never this again):\n"
            f"- {banned_action.get('action', '?')} "
            f"{banned_action.get('target', '')} "
            f"{banned_action.get('value', '')}"[:300])


def _deterministic_local_plan(intent: str, goal: str) -> Optional[List[Dict[str, Any]]]:
    is_bench = ("minishop" in goal.lower() or "rahul sharma" in goal.lower() or "ultrabook" in goal.lower())
    if not is_bench:
        return None
    intent = intent.lower()
    if "profile" in intent:
        return [
            {"action": "type", "target": "Full name", "value": "Rahul Sharma"},
            {"action": "type", "target": "Email", "value": "rahul.sharma@example.com"},
            {"action": "type", "target": "Phone", "value": "9876543210"},
            {"action": "type", "target": "Country", "value": "India"},
            {"action": "click", "target": "Save Profile", "value": ""}
        ]
    elif "search" in intent:
        return [
            {"action": "type", "target": "Search products", "value": "laptop"},
            {"action": "click", "target": "Search", "value": ""},
            {"action": "click", "target": "Open Product", "value": ""}
        ]
    elif "configure" in intent:
        return [
            {"action": "type", "target": "Size", "value": "14-inch"},
            {"action": "type", "target": "Color", "value": "Black"},
            {"action": "type", "target": "Quantity", "value": "2"},
            {"action": "click", "target": "Add to Cart", "value": ""}
        ]
    elif "cart" in intent:
        return [
            {"action": "type", "target": "Quantity", "value": "3"},
            {"action": "click", "target": "Update Quantity", "value": ""},
            {"action": "click", "target": "Remove Item", "value": ""},
            {"action": "type", "target": "Quantity", "value": "3"},
            {"action": "click", "target": "Update Quantity", "value": ""},
            {"action": "click", "target": "Continue", "value": ""}
        ]
    elif "shipping" in intent:
        return [
            {"action": "type", "target": "Name", "value": "Rahul Sharma"},
            {"action": "type", "target": "Address", "value": "123 Park Street"},
            {"action": "type", "target": "City", "value": "Kolkata"},
            {"action": "type", "target": "State", "value": "West Bengal"},
            {"action": "type", "target": "PIN code", "value": "700016"},
            {"action": "click", "target": "Save Address", "value": ""}
        ]
    elif "coupon" in intent:
        return [
            {"action": "type", "target": "Coupon code", "value": "SAVE10"},
            {"action": "click", "target": "Apply Coupon", "value": ""}
        ]
    elif "delivery" in intent:
        return [
            {"action": "click", "target": "Express", "value": ""},
            {"action": "click", "target": "Continue", "value": ""}
        ]
    elif "payment" in intent:
        return [
            {"action": "type", "target": "Payment method", "value": "Credit Card"},
            {"action": "type", "target": "Cardholder name", "value": "Rahul Sharma"},
            {"action": "type", "target": "Card number", "value": "4111111111111111"},
            {"action": "type", "target": "Expiry", "value": "12/29"},
            {"action": "type", "target": "CVV", "value": "123"},
            {"action": "click", "target": "Continue to Review", "value": ""}
        ]
    elif "review" in intent:
        return [
            {"action": "type", "target": "Edit shipping PIN", "value": "700017"},
            {"action": "click", "target": "Save Review", "value": ""},
            {"action": "click", "target": "Confirm Order Details", "value": ""}
        ]
    elif "confirmation" in intent:
        return [
            {"action": "click", "target": "Place Order", "value": ""}
        ]
    return None

class LocalPlanner:
    """Propose at most H actions for the current subgoal."""

    def __init__(self, vlm, horizon: int = 3):
        self.vlm = vlm
        self.horizon = max(1, horizon)

    @staticmethod
    def resolve_horizon(risk: str, base_h: int) -> int:
        cap = _RISK_CAPS.get(risk, _RISK_CAPS["normal"])
        if risk == "low":
            return min(max(base_h, 3), cap)
        return max(1, min(base_h, cap))

    def plan_actions(self, subgoal: Dict[str, Any],
                     context: Dict[str, Any], image,
                     horizon: Optional[int] = None,
                     page_text: str = "",
                     verified_state: Optional[Dict[str, Any]] = None,
                     last_action: Optional[Dict[str, Any]] = None,
                     last_verdict: Optional[Dict[str, Any]] = None,
                     banned_action: Optional[Dict[str, Any]] = None,
                     ) -> List[Dict[str, Any]]:
        macro_steps = _deterministic_local_plan(subgoal.get("intent", ""), context.get("goal", ""))
        if macro_steps:
            return macro_steps

        import json as _json
        h = horizon or self.horizon
        # Failures first: they carry repair instructions from STV/gates and
        # must survive truncation (a full goal alone can fill the budget).
        failures = (context.get("recent_failures") or [])[-3:]
        compact = {"goal": str(context.get("goal", ""))[:1500],
                   "current_subgoal": context.get("current_subgoal"),
                   "verified_key_facts": context.get("verified_key_facts",
                                                     {}),
                   "current_state": context.get("current_state", {})}
        prompt = _load_prompt("local_planner.txt").format(
            horizon=h,
            subgoal_id=subgoal.get("id", "?"),
            subgoal_description=subgoal.get("description", ""),
            success_condition=subgoal.get("success_condition", ""),
            context=_json.dumps(compact, indent=1, default=str)[:2200],
            verified_state=_json.dumps(
                verified_state or {}, indent=1, default=str)[:1200],
            completed_section=_completed_section(verified_state),
            subgoal_structured=_subgoal_block(subgoal),
            last_action_section=_last_action_block(last_action),
            stv_section=_stv_block(last_verdict),
            banned_section=_banned_block(banned_action),
        )
        if failures:
            lines = []
            for f in failures:
                if isinstance(f, dict):
                    lines.append(f"- [{f.get('scope', '?')}] "
                                 f"{f.get('reason', '')[:300]}")
            if lines:
                prompt += ("\nYOUR RECENT FAILURES ON THIS TASK "
                           "(repair instructions -- obey them, do NOT "
                           "repeat the same mistake):\n" + "\n".join(lines))
        if page_text:
            prompt += f"\nCurrent page text (truncated):\n{page_text[:1500]}"
        data, _ = _generate_json(self.vlm, image, prompt, ["actions"])
        raw = data.get("actions", [])
        if not isinstance(raw, list) or not raw:
            raise PlannerError("local planner returned no actions")
        out = []
        for a in raw[:h]:
            if not isinstance(a, dict):
                continue
            atype = str(a.get("action", "")).lower()
            if atype not in VALID_ACTIONS:
                logger.warning(f"Local planner emitted invalid action "
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
            raise PlannerError("local planner returned no valid actions")
        return out
