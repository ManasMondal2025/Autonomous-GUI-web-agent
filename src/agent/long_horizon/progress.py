"""Milestone distance / progress: deterministic planning signals.

distance(): required vs verified requirements of one milestone.
progress_delta(): before/after comparison of distance snapshots.

Planning/controller signals ONLY -- never a substitute for STV. A
positive delta suggests motion toward the milestone; STV verdicts remain
the sole authority for commits and advancement.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def _verified_fact_texts(memory) -> set:
    texts = set()
    try:
        facts = memory.key_facts or {}
    except Exception:
        return texts
    for name, f in facts.items():
        if not isinstance(f, dict) or f.get("status") != "VERIFIED":
            continue
        texts.add(str(name).lower())
        try:
            if f.get("value") not in (None, True):
                texts.add(str(f.get("value")).lower())
        except Exception:
            pass
    return texts


def _pred_label(pred: Any) -> str:
    if isinstance(pred, dict) and pred.get("predicate"):
        return (f"{pred.get('target', '?')} {pred.get('predicate')} "
                f"{pred.get('value', '')}").strip()[:160]
    return str(pred or "")[:160]


def outstanding_literals(literals, fact_texts) -> List[str]:
    """Milestone literals that name fillable values yet are unverified.

    Used when a milestone carries no terminal/success predicates (compact
    skeleton): its literals double as the required set. Only literals with
    a recognizable shape (name/email/phone/place), a number, or @ count --
    bare label words ("Change", "Update") are chrome, not requirements.
    """
    try:
        from .subgoal_planner import _literal_shape
    except Exception:
        return []
    facts = {str(t).lower() for t in fact_texts or []}
    out = []
    for lit in literals or []:
        s = str(lit or "").strip()
        if not s or s.lower() in facts:
            continue
        shaped = False
        try:
            shaped = bool(_literal_shape(s))
        except Exception:
            pass
        numeric = any(c.isdigit() for c in s)
        if shaped or numeric or "@" in s:
            out.append(s[:160])
    return out


def _pred_satisfied(pred: Any, fact_texts: set,
                    observation=None) -> bool:
    """A requirement counts as satisfied when its value already appears
    among VERIFIED fact names/values, or -- with a live observation --
    when its predicate verifies deterministically right now."""
    if isinstance(pred, dict) and pred.get("predicate"):
        value = str(pred.get("value", "") or "").strip().lower()
        if value and (value in fact_texts or any(
                value in t for t in fact_texts if len(value) > 3)):
            return True
        if observation is not None:
            try:
                from . import expectation as _expm
                target = pred.get("target", "")
                norm = {"target": target.split()
                        if isinstance(target, str) else
                        list(target or []),
                        "predicate": str(pred.get("predicate", "")),
                        "value": pred.get("value", ""),
                        "raw": str(pred)}
                pool = ""
                if isinstance(observation, dict):
                    from .stv import _evidence_pool
                    pool = _evidence_pool(observation, None)
                ok, _ = _expm.verify_item(norm, pool, pool,
                                          (observation.get("dom_elements",
                                                           []) or [])
                                          if isinstance(observation, dict)
                                          else [])
                return bool(ok)
            except Exception as e:
                logger.warning(f"Progress predicate check failed: {e}")
                return False
        return False
    text = str(pred or "").strip().lower()
    return bool(text) and text in fact_texts


def distance(memory, sid: str, observation=None) -> Dict[str, Any]:
    """Required vs verified requirements for one milestone."""
    sg = {}
    try:
        sg = (memory.verified_subgoals or {}).get(sid, {}) or {}
    except Exception:
        pass
    required: List[str] = []
    for p in (sg.get("terminal_predicates", []) or []
              or sg.get("success_predicates", []) or []):
        required.append(_pred_label(p))
    for r in sg.get("required_facts", []) or []:
        label = f"fact: {r}"[:160]
        if label not in required:
            required.append(label)
    fact_texts = _verified_fact_texts(memory)
    verified, remaining = [], []
    preds = (sg.get("terminal_predicates", []) or []
             or sg.get("success_predicates", []) or [])
    if not preds:
        # Compact skeleton milestones carry literals instead of predicates:
        # unfilled literals are the outstanding requirements.
        for lit in outstanding_literals(sg.get("literals", []), fact_texts):
            remaining.append(f"fill: {lit}")
    req_facts = [str(r) for r in (sg.get("required_facts", []) or [])]
    for p in preds:
        (verified if _pred_satisfied(p, fact_texts, observation)
         else remaining).append(_pred_label(p))
    for r in req_facts:
        hit = r.lower() in fact_texts or any(
            r.lower() in t for t in fact_texts) if r else False
        (verified if hit else remaining).append(f"fact: {r}"[:160])
    try:
        stage = memory.verified_stage
    except Exception:
        stage = "unknown"
    return {"sid": sid, "stage": stage, "required": required,
            "verified": verified, "remaining": remaining,
            "remaining_requirements": len(remaining)}


def progress_delta(before: Dict[str, Any], after: Dict[str, Any],
                   unexpected: int = 0,
                   regressions: int = 0) -> int:
    """Before/after distance comparison.

    newly_verified_required_facts_and_predicates
    + correct_page_transition
    - unexpected_changes - regressions.

    A page/stage transition counts only when remaining requirements also
    shrank (bare URL wobble is not progress).
    """
    before, after = before or {}, after or {}
    b_ver = set(before.get("verified", []) or [])
    a_ver = set(after.get("verified", []) or [])
    newly = len(a_ver - b_ver)
    page_transition = 0
    if before.get("stage") != after.get("stage") and \
            len(after.get("remaining", []) or []) < \
            len(before.get("remaining", []) or []):
        page_transition = 1
    try:
        unexpected = int(unexpected or 0)
    except Exception:
        unexpected = 0
    try:
        regressions = int(regressions or 0)
    except Exception:
        regressions = 0
    return newly + page_transition - unexpected - regressions
