"""State Transition Verification (STV): did the INTENDED transition occur?

STV compares the OBSERVED after-state against the PREDICTED transition
(Expected(S_t+1)) plus invariants. It never judges by raw pixel difference
alone and never trusts execution status as progress. Only STV PASS results
may promote facts to VERIFIED (via TaskMemory.commit_verified_fact).

Evidence hierarchy (cheapest confident signal wins):
  1. Deterministic DOM/URL/rule checks (free).
  2. Visual State Signature diff (OmniParser + OCR, no new VLM).
  3. Qwen fallback (VLM call) ONLY when layers 1-2 are inconclusive.

Decisions: PASS | RETRY | REPAIR | REPLAN | WAIT | ROLLBACK.
"""

import io
import logging
import re
from typing import Dict, Any, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

DECISIONS = ("PASS", "RETRY", "REPAIR", "REPLAN", "WAIT", "ROLLBACK")

_LOADING_MARKERS = ("loading", "please wait", "wait...", "spinner",
                    "processing", "one moment")


def _norm_text(t: Any) -> str:
    return " ".join(str(t or "").lower().split())


_UI_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "for", "with", "into", "from", "that",
    "this", "then", "than", "to", "at", "by", "of", "in", "on",
    "is", "are", "was", "were", "be", "been", "being",
    "please", "your", "you", "its", "it",
})


def _words_overlap(set_a: set, set_b: set) -> bool:
    if set_a & set_b:
        return True
    for a in set_a:
        for b in set_b:
            if (len(a) >= 3 and len(b) >= 3) and (a in b or b in a):
                return True
            if (a in ("qty", "quantity") and b in ("qty", "quantity")):
                return True
            if (a in ("pin", "pincode") and b in ("pin", "pincode")):
                return True
    return False


def _content_words(text: str) -> List[str]:
    # Split camelCase: cartQty -> cart Qty
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', str(text or ''))
    raw = [w.lower() for w in re.findall(r"[a-zA-Z0-9]+", text)]
    words = [w for w in raw if len(w) >= 2 and w not in _UI_STOPWORDS and w not in _META_WORDS]
    return words


# Closed set of requirement-phrasing glue (see _content_words). Deliberately
# NOT added to verification.goal_verifier.STOPWORDS: baseline goal checking
# semantics stay untouched; only STV matching ignores them.
_META_WORDS = frozenset({
    "contains", "contain", "containing", "visible", "visibly",
    "shows", "show", "showing", "shown", "displays", "display",
    "appears", "appear", "present", "says", "reads", "message",
    "text", "labeled", "labelled", "titled",
    "dropdown", "option", "options",
    "button", "buttons", "field", "fields", "input", "inputs",
    "choose", "select", "click", "clicked",
})


def _evidence_pool(obs: Dict[str, Any],
                   signature: Optional[Dict[str, Any]] = None) -> str:
    """Searchable text: URL + visible text + DOM fields + signature texts."""
    parts = [obs.get("url", ""), obs.get("visible_text", "")]
    for el in obs.get("dom_elements", []) or []:
        for key in ("text", "ariaLabel", "placeholder", "title", "value",
                    "name"):
            v = el.get(key, "")
            if v:
                parts.append(str(v))
    if signature:
        parts.extend(signature.get("texts", []))
    return "\n".join(p for p in parts if p)


def _dom_field_values(obs: Dict[str, Any]) -> str:
    """Observable form state: non-empty input values + checked/selected
    markers from DOM elements. Typing into an input changes no visible
    text, so without this channel every form-fill action is unverifiable
    (screenshot shows the value, visible_text does not)."""
    vals = []
    for el in obs.get("dom_elements", []) or []:
        if not isinstance(el, dict):
            continue
        v = str(el.get("value", "") or "").strip()
        if v:
            vals.append(v)
        if el.get("checked"):
            label = str(el.get("text", "") or el.get("value", "")
                        or el.get("name", "")).strip()
            vals.append(f"[checked] {label}".strip())
    return " ".join(vals)


def _keywords_hit(requirement: str, evidence: str) -> bool:
    words = _content_words(requirement)
    if not words:
        return True  # vacuous requirement; never blocks
    ev = _norm_text(evidence)
    return all(w in ev for w in words)


def _forbidden_hit(forbidden: str, evidence: str, url: str) -> bool:
    target = _norm_text(f"{url}\n{evidence}")
    words = _content_words(forbidden)
    if not words:
        return False
    return all(w in target for w in words)


def build_failure_trace(decision: str, reason: str,
                        expected_state: Any,
                        observed_state: Any,
                        failed_action: Any,
                        evidence_items: Any) -> Dict[str, Any]:
    """Structured STV failure feedback for state-conditioned replanning.

    Shape fits the existing verdict format; every field is plain
    JSON-serializable data (never model prose). Callers store it under the
    `trace` key of a recent_failures entry, keeping the string `reason`
    for backward compatibility.
    """
    def _short(v: Any, n: int = 300) -> Any:
        try:
            s = v if isinstance(v, (dict, list)) else str(v or "")
        except Exception:
            s = ""
        if isinstance(s, str):
            return s[:n]
        return s
    ev = evidence_items if isinstance(evidence_items, list) \
        else [evidence_items]
    return {"decision": str(decision or ""),
            "reason": str(reason or "")[:500],
            "expected_state": _short(expected_state),
            "observed_state": _short(observed_state),
            "failed_action": _short(failed_action),
            "evidence": [_short(e, 200) for e in ev[:6]]}


def _loading_markers_present(obs: Dict[str, Any]) -> bool:
    blob = _norm_text(f"{obs.get('visible_text', '')} "
                      f"{obs.get('url', '')}").lower()
    if any(m in blob for m in _LOADING_MARKERS):
        return True
    for el in obs.get("dom_elements", []) or []:
        if str(el.get("role", "")).lower() in ("progressbar", "status"):
            return True
        if "loading" in _norm_text(el.get("text", "")):
            return True
    return False


def _point_in_elements(x: float, y: float,
                       elements: List[Dict[str, Any]], pad: float = 6.0,
                       ) -> bool:
    """Is (x, y) inside any known interactable box (DOM or detector)?"""
    for el in elements or []:
        bb = el.get("bbox", {}) or {}
        try:
            ex, ey = float(bb.get("x", 0)), float(bb.get("y", 0))
            ew, eh = float(bb.get("width", 0)), float(bb.get("height", 0))
        except (TypeError, ValueError):
            continue
        if ex - pad <= x <= ex + ew + pad and ey - pad <= y <= ey + eh:
            return True
    return False


class STV:
    """Deterministic-first state transition verifier."""

    def __init__(self, detector=None, ocr=None, vlm=None,
                 fallback_threshold: float = 0.6,
                 visual_fallback: bool = True):
        self.detector = detector
        self.ocr = ocr
        self.vlm = vlm
        self.fallback_threshold = fallback_threshold
        self.visual_fallback = visual_fallback

    # -- public API -----------------------------------------------------
    def decide(self, before_obs: Dict[str, Any], action: Any,
               after_obs: Dict[str, Any],
               expected_transition: Optional[Dict[str, Any]],
               memory=None, target: str = "") -> Dict[str, Any]:
        """Return spec-§9 verdict dict. Never raises on bad inputs.

        `target` is the planner's natural-language target ("Email field").
        It powers field attribution: the clicked element's own identifiers
        must overlap the target words, otherwise a right-value-wrong-field
        action would PASS on page-global keyword matching."""
        exp = dict(expected_transition or {})
        for key in ("must_change", "must_be_true", "must_remain_true",
                    "must_not_happen"):
            vals = exp.get(key, [])
            exp[key] = [vals] if isinstance(vals, str) else list(vals or [])
        evidence, sig = self._gather_evidence(before_obs, after_obs)
        exec_ok = bool(after_obs.get("exec_success", True))
        if after_obs.get("exec_error"):
            return self._verdict(
                "RETRY", 0.7, evidence,
                reason=f"execution layer reported error "
                       f"({after_obs.get('exec_error')}); action may not "
                       f"have executed",
                state_changes={}, unexpected_changes=[],
                memory_updates=[])

        # Harmful/forbidden outcomes first (ROLLBACK iff a checkpoint to
        # restore exists and is known via memory, else REPLAN).
        for item in exp["must_not_happen"]:
            if _forbidden_hit(item, evidence["after_text"],
                              after_obs.get("url", "")):
                has_cp = bool(memory is not None and getattr(
                    memory, "latest_checkpoint_ref", lambda: None)())
                return self._verdict(
                    "ROLLBACK" if has_cp else "REPLAN", 0.8, evidence,
                    reason=f"forbidden outcome observed: {item!r}",
                    state_changes={}, unexpected_changes=[item],
                    memory_updates=[])

        must_change_ok, change_ev = self._check_items(
            exp["must_change"], before_obs, after_obs, evidence)
        must_be_ok, be_ev = self._check_items(
            exp["must_be_true"], before_obs, after_obs, evidence)
        remain_violations = self._check_remain(
            exp["must_remain_true"], evidence)
        loading = _loading_markers_present(after_obs)

        state_changes = {**change_ev, **be_ev}
        memory_updates = [
            {"fact": k, "value": True, "evidence": v}
            for k, v in state_changes.items()]
        # Typed values are first-class verified facts: a value observed in
        # the after-state is committed literally, so later stages (planner
        # context, final review) can reuse exact strings like names/codes.
        typed_value = self._action_value(action)
        if (typed_value and typed_value.strip()
                and _keywords_hit(typed_value, evidence["after_text"])):
            memory_updates.append(
                {"fact": typed_value.strip()[:200], "value": "entered",
                 "evidence": "typed value observed in after-state"})

        if remain_violations:
            return self._verdict(
                "REPLAN", 0.75, evidence,
                reason="invariant broken: "
                       + "; ".join(remain_violations),
                state_changes=state_changes,
                unexpected_changes=list(remain_violations),
                memory_updates=[])
        # Field attribution (type actions): the value must have landed in
        # the DOM element under the action coordinates. Page-global keyword
        # matching cannot tell fields apart -- without this, typing an email
        # into the phone box still "observes" the email and PASSES. A
        # misdirected action commits nothing (not even the typed value).
        field_fault = self._wrong_field_reason(action, after_obs, target)
        if field_fault is not None:
            return self._verdict(
                "REPAIR", 0.7, evidence, reason=field_fault,
                state_changes=state_changes, unexpected_changes=[],
                memory_updates=[])
        if must_change_ok and must_be_ok:
            conf = self._confidence(exp, True, True, loading)
            return self._verdict("PASS", conf, evidence,
                                 reason="predicted transition observed; "
                                        "invariants hold",
                                 state_changes=state_changes,
                                 unexpected_changes=[],
                                 memory_updates=memory_updates)
        if loading:
            return self._verdict(
                "WAIT", 0.6, evidence,
                reason="loading markers present; UI may still be settling",
                state_changes=state_changes, unexpected_changes=[],
                memory_updates=[])
        # No transition observed: distinguish miss vs no-response.
        acted = self._action_coordinates(action)
        if acted is not None:
            x, y = acted
            elements = list(before_obs.get("dom_elements", []) or [])
            elements += sig.get("ui_elements_boxes", []) if isinstance(
                sig, dict) else []
            if not _point_in_elements(x, y, self._as_bboxes(elements)):
                return self._verdict(
                    "REPAIR", 0.65, evidence,
                    reason=f"click/type at ({x:.0f},{y:.0f}) hit no known "
                           f"interactable element: grounding likely missed",
                    state_changes=state_changes, unexpected_changes=[],
                    memory_updates=[])
        return self._verdict(
            "RETRY", 0.6, evidence,
            reason="no predicted change observed and no error shown; "
                   "action may not have registered",
            state_changes=state_changes, unexpected_changes=[],
            memory_updates=[])

    def verify_subgoal(self, subgoal: Dict[str, Any],
                       observation: Dict[str, Any],
                       signature: Optional[Dict[str, Any]] = None,
                       ) -> Dict[str, Any]:
        """Independently check a whole subgoal against live evidence.

        Structured contracts win: `terminal_predicates` first, else
        `success_predicates`; when either is present and non-empty, EVERY
        predicate must verify (via the validated semantic layer) and the
        legacy string condition is display-only. Otherwise the legacy path
        applies: the success_condition is the contract when present, else
        the description.
        """
        sig = signature or {}
        preds = subgoal.get("terminal_predicates", []) or \
            subgoal.get("success_predicates", []) or []
        if preds:
            from . import expectation as _expm
            pool = _evidence_pool(observation, sig)
            dom = observation.get("dom_elements", []) or []
            checks, ok_all, evidence = [], True, []
            for p in preds:
                if isinstance(p, dict) and p.get("predicate"):
                    target = p.get("target", "")
                    norm = {"target": target.split()
                            if isinstance(target, str) else
                            list(target or []),
                            "predicate": str(p.get("predicate", "")),
                            "value": p.get("value", ""),
                            "raw": str(p)}
                else:
                    norm = _expm.normalize_expectation(str(p or ""))
                ok, note = _expm.verify_item(norm, pool, pool, dom)
                label = str(p if isinstance(p, str) else
                            p.get("value", p))[:120]
                checks.append({"requirement": label, "satisfied": ok})
                ok_all = ok_all and ok
                if ok:
                    evidence.append(f"observed: {label} ({note})"[:160])
            return {"satisfied": ok_all, "evidence": evidence,
                    "checks": checks}
        pool = _evidence_pool(observation, sig)
        cond = str(subgoal.get("success_condition", "") or "").strip()
        targets = [cond] if cond else [
            str(subgoal.get("description", "") or "")]
        checks = []
        ok_all, evidence = True, []
        for t in targets:
            if not str(t).strip():
                continue
            hit = _keywords_hit(t, pool)
            checks.append({"requirement": t, "satisfied": hit})
            ok_all = ok_all and hit
            if hit:
                evidence.append(f"observed: {t[:120]}")
        return {"satisfied": ok_all, "evidence": evidence, "checks": checks}

    def verify_final_goal(self, task: Dict[str, Any],
                           success_conditions: List[str],
                           observation: Dict[str, Any],
                           signature: Optional[Dict[str, Any]] = None,
                           ) -> Dict[str, Any]:
        """STV-gated DONE: every success predicate must hold live."""
        sig = signature or {}
        pool = _evidence_pool(observation, sig)
        missing = [c for c in success_conditions
                   if c and not _keywords_hit(c, pool)]
        return {"satisfied": not missing, "missing": missing,
                "evidence": [f"predicate holds: {c[:100]}"
                             for c in success_conditions if c not in missing]}
    
    def check_live(self, requirement: str,
                   observation: Dict[str, Any],
                   signature: Optional[Dict[str, Any]] = None) -> bool:
        """Single public keyword probe: does `requirement` hold in the live
        observation (page + DOM values + visual texts)? Used by the
        controller's final check for the terminal marker."""
        if not str(requirement or "").strip():
            return True
        try:
            sig = signature or {}
            if not sig:
                _, sig = self._gather_evidence(observation, observation)
            pool = _evidence_pool(observation, sig)
            return _keywords_hit(requirement, pool)
        except Exception as e:
            logger.warning(f"Live check crashed ({e}); treating as unmet")
            return False

    # -- internals --------------------------------------------------------
    def _gather_evidence(self, before_obs, after_obs):
        """DOM/URL evidence always; visual signature when configured and
        useful (DOM-scarce or URL-unchanged). Returns (evidence, sig)."""
        evidence = {
            "before_url": before_obs.get("url", ""),
            "after_url": after_obs.get("url", ""),
            "url_changed": before_obs.get("url", "")
            != after_obs.get("url", ""),
            "before_text": _norm_text(
                f"{before_obs.get('visible_text', '')}\n"
                f"{_dom_field_values(before_obs)}"),
            "after_text": _norm_text(
                f"{after_obs.get('visible_text', '')}\n"
                f"{_dom_field_values(after_obs)}"),
        }
        sig = {}
        dom_n = len(before_obs.get("dom_elements", []) or []) + len(
            after_obs.get("dom_elements", []) or [])
        if self.visual_fallback and (dom_n < 10 or not evidence["url_changed"]):
            try:
                from .state_extractor import extract_signature
                img = Image.open(io.BytesIO(
                    after_obs["screenshot_bytes"]))
                sig = extract_signature(
                    img, after_obs.get("dom_elements"),
                    detector=self.detector, ocr=self.ocr)
                evidence["signature_texts"] = list(sig.get("texts", []))
            except Exception as e:
                logger.warning(f"STV visual layer failed: {e}")
        return evidence, sig

    def _check_items(self, items, before_obs, after_obs, evidence):
        """Per-item semantic verification (TARGET+PREDICATE+VALUE).

        Each planner expectation is normalized and checked DOM-first, page
        second. Empty items never block (no requirement stated);
        non-empty but unresolvable items FAIL the check (uncertainty never
        passes -- the recovery ladder, not a guess, handles it).
        """
        from . import expectation as _expm
        ok, ev = True, {}
        for item in items:
            if not str(item or "").strip():
                continue
            norm = _expm.normalize_expectation(item)
            hit, note = _expm.verify_item(
                norm, evidence.get("before_text", ""),
                evidence.get("after_text", ""),
                (after_obs.get("dom_elements", []) or [])
                if isinstance(after_obs, dict) else [])
            ok = ok and hit
            if hit:
                ev[item] = note
        return ok, ev

    @staticmethod
    def _check_remain(items, evidence):
        violations = []
        for item in items:
            was = _keywords_hit(item, evidence["before_text"])
            still = _keywords_hit(item, evidence["after_text"])
            if was and not still:
                violations.append(f"{item!r} held before but broke after")
        return violations

    @staticmethod
    def _confidence(exp, change_ok, be_ok, loading):
        conf = 0.5
        conf += min(0.3, 0.1 * len(exp.get("must_change", []))) \
            if change_ok else 0.0
        conf += 0.1 if be_ok and exp.get("must_be_true") else 0.0
        if loading:
            conf -= 0.1
        return round(max(0.05, min(0.95, conf)), 2)

    @staticmethod
    def _verdict(decision, confidence, evidence, reason,
                 state_changes, unexpected_changes, memory_updates):
        assert decision in DECISIONS, decision
        return {"decision": decision, "confidence": confidence,
                "evidence": evidence if isinstance(evidence, dict)
                else {"note": str(evidence)},
                "state_changes": state_changes,
                "unexpected_changes": unexpected_changes,
                "memory_updates": memory_updates, "reason": reason}

    @staticmethod
    def _action_value(action) -> str:
        try:
            v = action.get("value") if isinstance(action, dict) \
                else getattr(action, "value", "")
            return str(v or "")
        except Exception:
            return ""

    @staticmethod
    def _element_at(x: float, y: float,
                    elements: List[Dict[str, Any]],
                    pad: float = 6.0) -> Optional[Dict[str, Any]]:
        """First DOM element whose bbox contains (x, y), else None."""
        for el in elements or []:
            if not isinstance(el, dict):
                continue
            bb = el.get("bbox", {}) or {}
            try:
                ex, ey = float(bb.get("x", 0)), float(bb.get("y", 0))
                ew, eh = float(bb.get("width", 0)), float(bb.get("height", 0))
            except (TypeError, ValueError):
                continue
            if ex - pad <= x <= ex + ew + pad and ey - pad <= y <= ey + eh + pad:
                return el
        return None

    def _wrong_field_reason(self, action, after_obs,
                              target: str = "") -> Optional[str]:
        """None when the action hit the intended field (or when attribution
        is impossible: no coords, no DOM, element without identifiers).
        Otherwise a REPAIR reason string. Two complementary checks:

        1. value presence: a typed value must be readable in the element
           under the action coordinates;
        2. label agreement: that element's own identifiers (text, label,
           placeholder, name, id, input type) must share at least one
           content word with the planner's target -- otherwise a right
           value in the wrong field (email typed into the phone box)
           would PASS on page-global matching.
        """
        coords = self._action_coordinates(action)
        if coords is None:
            return None
        x, y = coords
        try:
            hit = self._element_at(
                x, y, after_obs.get("dom_elements", []) or [])
        except Exception:
            return None
        if hit is None:
            return None  # no-element miss is reported by the caller
        typed = self._action_value(action)
        if typed.strip():
            got = str(hit.get("value", "") or "") \
                if "value" in hit else None
            if got is not None and \
                    typed.strip().lower() not in got.strip().lower():
                return (f"typed {typed[:40]!r} at ({x:.0f},{y:.0f}) but the "
                        f"field there shows {got[:40]!r}: right value, "
                        f"wrong field")
        want = set(_content_words(target or ""))
        if not want:
            return None  # vague target ("click it"): cannot judge labels
        have = set()
        for key in ("text", "ariaLabel", "label", "placeholder", "name", "htmlId", "id",
                    "value"):
            have.update(_content_words(str(hit.get(key, "") or "")))
        # Input types only when meaningful: "select-one"/"submit"/"text"
        # contribute junk words ("one") that cause false mismatches.
        if str(hit.get("type", "")).lower() in (
                "email", "tel", "number", "password", "radio",
                "checkbox", "url", "date", "search"):
            have.update(_content_words(str(hit.get("type", ""))))
        # The typed value itself is not an identifier (it is what we wrote).
        have -= set(_content_words(typed))
        if have and not _words_overlap(want, have):
            label = str(hit.get("text", "") or hit.get("ariaLabel", "")
                        or hit.get("label", "")
                        or hit.get("placeholder", "")
                        or hit.get("name", "") or hit.get("htmlId", "") or hit.get("type", ""))[:60]
            return (f"action at ({x:.0f},{y:.0f}) hit the "
                    f"{label or hit.get('tag', 'element')!r} element but "
                    f"the target was {target[:60]!r}: wrong element")
        return None

    @staticmethod
    def _action_coordinates(action):
        try:
            x = action.x if hasattr(action, "x") else action.get("x")
            y = action.y if hasattr(action, "y") else action.get("y")
            if x is None or y is None:
                return None
            return float(x), float(y)
        except (TypeError, ValueError, AttributeError):
            return None

    @staticmethod
    def _as_bboxes(elements):
        out = []
        for el in elements or []:
            if isinstance(el, dict) and "box" in el and "label" in el:
                # state-extractor ui_elements use normalized xyxy boxes
                try:
                    x1, y1, x2, y2 = el["box"]
                    out.append({"bbox": {"x": x1, "y": y1,
                                         "width": x2 - x1,
                                         "height": y2 - y1}})
                    continue
                except (TypeError, ValueError):
                    pass
            out.append(el)
        return out

    # -- Qwen fallback (last resort only) -----------------------------------
    def fallback_verdict(self, before_obs, action, after_obs,
                         expected_transition, evidence) -> Optional[Dict]:
        """Ask Qwen whether the transition occurred. Called by the
        controller only when deterministic confidence is low."""
        if self.vlm is None:
            return None
        try:
            from PIL import Image as _Image
            from pathlib import Path
            prompt = Path(__file__).resolve().parent.parent / "prompts" \
                / "stv_fallback.txt"
            import json as _json
            text = prompt.read_text().format(
                action=_json.dumps(action if isinstance(action, dict)
                                   else getattr(action, "__dict__", {}),
                                   default=str)[:800],
                expected_transition=_json.dumps(expected_transition,
                                                default=str)[:800],
                before_evidence=_json.dumps(
                    {k: v for k, v in evidence.items()
                     if k != "signature_texts"}, default=str)[:1500],
                after_evidence=_json.dumps(
                    evidence.get("signature_texts", [])[:20], default=str))
            img = _Image.open(io.BytesIO(after_obs["screenshot_bytes"]))
            raw = self.vlm.generate(img, text)
            import re
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            data = _json.loads(m.group(0) if m else raw)
            dec = str(data.get("decision", "")).upper()
            if dec not in ("PASS", "RETRY", "REPAIR", "REPLAN", "WAIT"):
                logger.warning(f"STV fallback returned bad decision {dec!r}")
                return None
            try:
                conf = float(data.get("confidence", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
            return {"decision": dec, "confidence": max(0.05, min(0.95, conf)),
                    "evidence": {"fallback": True,
                                 "items": data.get("evidence", [])},
                    "state_changes": {}, "unexpected_changes": [],
                    "memory_updates": [],
                    "reason": str(data.get("reason", ""))[:300]}
        except Exception as e:
            logger.warning(f"STV fallback failed: {e}")
            return None

    def verify_popup_dismissal(self, before_obs: Dict[str, Any],
                               after_obs: Dict[str, Any],
                               popup_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Verify that an obstructing popup/modal/banner is no longer visible."""
        b_text = str(before_obs.get("visible_text", "") or "").lower()
        a_text = str(after_obs.get("visible_text", "") or "").lower()
        b_dom = before_obs.get("dom_elements", []) or []
        a_dom = after_obs.get("dom_elements", []) or []

        banner_id = (popup_info or {}).get("banner_id", "")
        # Check if dialog container in after_obs is hidden or absent
        visible_in_after = False
        for el in a_dom:
            if not isinstance(el, dict):
                continue
            el_id = el.get("id") or el.get("htmlId")
            if banner_id and el_id == banner_id:
                if el.get("visible") is not False:
                    visible_in_after = True
                    break
            if el.get("role") in ("dialog", "alertdialog") and el.get("visible") is not False:
                # Check if it was in before
                visible_in_after = True
                break

        # Check keyword markers: cookie preferences, session notice
        if "session notice" in a_text or "cookie preferences" in a_text:
            # Check if backdrop or banner is visible
            for el in a_dom:
                if el.get("id") in ("cookieBanner", "blockingModal") and el.get("visible") is not False:
                    visible_in_after = True

        success = not visible_in_after
        decision = "PASS" if success else "FAIL"
        reason = "Popup/dialog obstruction verified removed." if success else "Popup/dialog still visible after dismissal attempt."
        return {
            "success": success,
            "decision": decision,
            "confidence": 1.0 if success else 0.8,
            "reason": reason,
            "evidence": [reason]
        }

    def verify_date_selection(self, observation: Dict[str, Any],
                              expected_date_iso: str,
                              target_field: str = "appointmentDate") -> Dict[str, Any]:
        """Verify that the date field holds exactly the expected ISO date."""
        dom = observation.get("dom_elements", []) or []
        val_found = None
        for el in dom:
            if not isinstance(el, dict):
                continue
            fid = str(el.get("id") or el.get("htmlId") or el.get("name") or "")
            if target_field.lower() in fid.lower():
                val_found = str(el.get("value", "") or "").strip()
                break

        pool = _evidence_pool(observation)
        if val_found is None:
            # Fallback to search in pool
            if expected_date_iso in pool:
                val_found = expected_date_iso

        success = (val_found == expected_date_iso)
        decision = "PASS" if success else "FAIL"
        reason = f"Date verified: {val_found} == {expected_date_iso}" if success else f"Date mismatch: found {val_found!r}, expected {expected_date_iso!r}"
        return {
            "success": success,
            "decision": decision,
            "confidence": 1.0 if success else 0.9,
            "observed_value": val_found,
            "expected_value": expected_date_iso,
            "reason": reason,
            "evidence": [reason]
        }

    def verify_async_result(self, observation: Dict[str, Any],
                            expected_marker: str,
                            forbidden_error: str = "failed") -> Dict[str, Any]:
        """Verify that an asynchronous operation finished successfully."""
        pool = _evidence_pool(observation).lower()
        exp = expected_marker.lower()
        has_expected = exp in pool
        has_error = forbidden_error.lower() in pool if forbidden_error else False

        success = has_expected and not has_error
        decision = "PASS" if success else "FAIL"
        if has_error:
            reason = f"Asynchronous operation reported error matching '{forbidden_error}'."
        elif has_expected:
            reason = f"Observable async completion confirmed: found '{expected_marker}'."
        else:
            reason = f"Expected async marker '{expected_marker}' not observed in page state."

        return {
            "success": success,
            "decision": decision,
            "confidence": 0.95 if success else 0.85,
            "reason": reason,
            "evidence": [reason]
        }
