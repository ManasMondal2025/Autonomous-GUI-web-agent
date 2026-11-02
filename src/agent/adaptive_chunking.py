"""Adaptive Action Chunk Controller (ACC).

Opt-in experimental feature (default OFF, activated via
``--adaptive-chunking`` together with ``--action-chunking``). When enabled,
ACC deterministically decides how many proposed GUI actions may execute in
one chunk before the next required observation / verification boundary.

Design constraints (deliberate):
- ACC never generates actions, never calls a model, and never touches the
  browser. The existing micro-planner still plans; the existing
  ActionExecutor still executes; the existing StateVerifier / STV still
  verify; the existing recovery still recovers.
- ACC only consumes signals the agent already collects: DOM observations,
  grounded targets, execution results, verifier outcomes, and its own
  bounded history of recent chunk outcomes.
- Sizing policy (spec-conservative):
    Rule A (conservative): allowed = 1 when targets are unknown/low
        confidence, DOM is dynamic/unknown, or a relevant recent failure
        is pending.
    Rule B (navigation/boundary): a boundary action may execute, then the
        chunk ends and remaining actions are dropped for fresh observation.
    Rule C (ordinary): up to 2 actions when every target in the prefix is
        validated against the current DOM and the state is stable.
    Rule D (larger): up to the configured cap (default 3) additionally
        requires explicit, populated per-action confidence (never invented
        by ACC).
    Rule E (recovery): any failure resets the recovery streak to 0
        (preferred size back to 1); each successful verified execution
        grows the preferred size by exactly one step, up to the cap.
- effective limit: min(configured hard cap, existing limit, state limit,
  recovery preference, number of proposed actions). Missing or invalid
  signals are treated conservatively; they never enlarge a chunk.

Sizing runs in two modes: proposal mode (controllers pass their concrete
proposal; target-confidence Rules A/C/D apply) and horizon mode
(``proposed_actions=None`` — no proposal exists yet, so only
stability/recovery rules apply and per-action target validation happens in
``constrain_actions`` at execution time).

State verification reuse (false-success fixes):
- ``verify_expected_transition`` enforces the planner's
  ``expected_transition`` contract (must_change / must_be_true /
  must_remain_true / must_not_happen) by composing the existing
  long-horizon STV evidence machinery (``expectation.normalize_expectation``
  / ``verify_item`` and ``stv._forbidden_hit``). It distinguishes
  ``passed`` / ``failed`` / ``unknown`` / ``no_contract`` and never
  converts an unknown result into verified success. Hardened semantics:
  a ``must_change`` property item must hold AFTER the action and NOT have
  held before it (an unrelated page mutation or an already-true property
  never satisfies it), and a ``must_not_happen`` item only passes when the
  forbidden condition was reliably observable in its absence.

Goal verification (fail-closed completion):
- ``classify_goal_check`` classifies one existing GoalVerifier result
  (or a check() exception) into ``verified_success`` / ``verified_failure``
  / ``unknown`` / ``unavailable`` / ``verification_error``. Anything that
  is not an explicit, well-formed "achieved" answer — including a missing
  verifier — is never ``verified_success``; callers reject unverified
  terminal ``done`` claims through their existing recovery paths.

Stability decay: an ``unknown`` outcome (verification could not be
established) decays the recovery streak by one step instead of leaving it
untouched (spec §6.7: uncertain verification must reset or decay). It never
grows the streak and never counts as verified success.
"""

import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Supported chunk-size envelope (spec: initial controller supports preferred
# sizes 1..3; the default hard cap never exceeds 3; larger caps are only
# accepted from explicit experiment configuration and are validated).
ACC_MIN_CHUNK_SIZE = 1
ACC_DEFAULT_MAX_CHUNK_SIZE = 3
ACC_MAX_SUPPORTED_CAP = 6
ACC_INITIAL_PREFERRED_SIZE = 1
ACC_MAX_STATE_LIMIT = 3

# Explicit per-action confidence required before Rule D may grant the full
# cap. This score must come from the model/grounding output; ACC never
# invents a number when the field is absent.
ACC_CONFIDENCE_THRESHOLD = 0.7

# Bounded outcome history (spec §14): small, deterministic, testable.
ACC_MAX_HISTORY = 20

# DOM element-count delta that marks a page as dynamic between two chunk
# starts (relative to the larger of the two counts, floored at 8).
ACC_DOM_DELTA_FLOOR = 8
ACC_DOM_DELTA_RATIO = 0.25

# A must_not_happen absence verdict needs a material page-evidence space.
# With no DOM elements and less than this many characters of visible text,
# "the forbidden string is absent" is not reliable observability (§3.3).
ACC_MNH_EVIDENCE_FLOOR = 8

# verify_item notes that mean "could not evaluate" rather than "disproven".
_UNRESOLVABLE_MARKERS = ("unparseable expectation", "no resolvable",
                         "cannot verify", "cannot be observed")


def _is_unresolvable_note(note: str) -> bool:
    low = str(note or "").lower()
    return any(marker in low for marker in _UNRESOLVABLE_MARKERS)


# Goal-verification verdicts (fail-closed completion, spec §2). Only
# ``verified_success`` may permit a terminal completion.
GCH_VERIFIED_SUCCESS = "verified_success"
GCH_VERIFIED_FAILURE = "verified_failure"
GCH_UNKNOWN = "unknown"
GCH_UNAVAILABLE = "unavailable"
GCH_VERIFICATION_ERROR = "verification_error"


def classify_goal_check(result: Any = None,
                        exception: Any = None,
                        is_intermediate: bool = False) -> Dict[str, Any]:
    """Classify one EXISTING GoalVerifier result into an ACC verdict.

    A small adapter only: the verifier and its result semantics are
    untouched. Fail-closed by construction — anything that is not an
    explicit, well-formed "achieved" answer is never ``verified_success``:

    - ``exception is not None``  -> ``verification_error``
    - ``result is None``         -> ``unknown`` (no result to trust)
    - malformed result / missing verdict -> ``unknown``
    - ``achieved`` falsy         -> ``verified_failure`` (explicit negative)
    - ``achieved`` truthy        -> ``verified_success`` (unless disqualified
                                    as an intermediate heuristic match)

    Terminal-success correctness invariant:
      terminal_success => goal_check_verdict == verified_success.
    An intermediate action must NOT terminate the entire task merely
    because local state matches a heuristic (e.g., input values containing
    entity strings or partial DOM mutations without an explicit terminal
    signal). In such intermediate cases, ``verdict`` is ``verified_failure``
    so that the chunk / execution sequence can run to completion.

    Callers treat every verdict other than ``verified_success`` as
    "unverified"; a missing verifier maps to ``unavailable`` at the call
    site (there is nothing to consult).
    """
    if exception is not None:
        return {"verdict": GCH_VERIFICATION_ERROR, "achieved": None,
                "done": False,
                "reason": f"goal check error: {exception}", "signals": {}}
    if result is None:
        return {"verdict": GCH_UNKNOWN, "achieved": None, "done": False,
                "reason": "goal check returned no result", "signals": {}}
    if not isinstance(result, dict):
        return {"verdict": GCH_UNKNOWN, "achieved": None, "done": False,
                "reason": (f"malformed goal check result type "
                           f"{type(result).__name__}"),
                "signals": {}}
    achieved = result.get("achieved", result.get("done"))
    if achieved is None:
        return {"verdict": GCH_UNKNOWN, "achieved": None, "done": False,
                "reason": str(result.get("reason", "") or
                              "goal check result carried no verdict"),
                "signals": dict(result.get("signals") or {})}
    ok = bool(achieved)
    signals = dict(result.get("signals") or {})

    # Intermediate non-terminal safeguard:
    # An intermediate action must never claim verified success based on heuristic
    # partial state (such as input field matches or unverified DOM changes).
    # Only an explicit terminal action (done) or genuinely verified terminal proof
    # may permit terminal success.
    if ok and is_intermediate:
        has_input_match = bool(signals.get("input_match"))
        is_heuristic = (bool(signals.get("heuristic_reason")) or
                        "heuristic" in str(result.get("reason", "")).lower() or
                        not signals.get("explicit_verified"))
        if has_input_match or is_heuristic:
            return {
                "verdict": GCH_VERIFIED_FAILURE,
                "achieved": False,
                "done": False,
                "reason": ("intermediate action heuristic match cannot claim "
                           "terminal goal completion"),
                "signals": signals,
            }

    return {"verdict": (GCH_VERIFIED_SUCCESS if ok else GCH_VERIFIED_FAILURE),
            "achieved": ok, "done": ok,
            "reason": str(result.get("reason", "") or (
                "Goal already achieved" if ok else "Goal not achieved")),
            "signals": signals}


class AdaptiveChunkController:
    """Deterministic chunk-size governor for the existing chunk executors.

    The controller is intentionally small: assess signals -> choose a size
    -> constrain the accepted prefix -> record outcomes. Enforcement stays
    with the existing controllers (they slice/execute/verify; ACC advises
    with structured decisions).
    """

    def __init__(self, enabled: bool = False,
                 hard_cap: int = ACC_DEFAULT_MAX_CHUNK_SIZE):
        self.enabled = bool(enabled)
        cap = int(hard_cap) if hard_cap is not None else ACC_DEFAULT_MAX_CHUNK_SIZE
        if not (ACC_MIN_CHUNK_SIZE <= cap <= ACC_MAX_SUPPORTED_CAP):
            raise ValueError(
                f"adaptive chunk hard cap {cap} outside supported range "
                f"[{ACC_MIN_CHUNK_SIZE}, {ACC_MAX_SUPPORTED_CAP}]")
        self.hard_cap = cap
        self._consecutive_verified_successes = 0
        self._outcomes: deque = deque(maxlen=ACC_MAX_HISTORY)
        # Cumulative counters for logging/evaluation (never reset by
        # reset_if_needed; task state lives in the streak + history only).
        self.stats: Dict[str, int] = {
            "chunks_planned": 0, "actions_accepted": 0,
            "actions_dropped": 0, "verified_successes": 0,
            "failures": 0, "unknown_verifications": 0, "resets": 0,
        }
        # Structured tallies (additive observability; §7). Controllers attach
        # structured fields to record_outcome calls; zero when a field is
        # never reported. Never treated as Rule E evidence.
        self.contract_results: Dict[str, int] = {
            "passed": 0, "failed": 0, "unknown": 0, "no_contract": 0}
        self.goal_check_verdicts: Dict[str, int] = {}
        self.executed_sizes: Dict[int, int] = {}
        self.interruptions: Dict[str, int] = {}

    # -- state assessment (spec §6) ---------------------------------------
    def assess_state(self, observation: Optional[Dict[str, Any]],
                     prev_observation: Optional[Dict[str, Any]] = None,
                     ) -> Dict[str, Any]:
        """Evaluate DOM/browser stability and recent-failure signals.

        Uses only existing observation dictionaries. Missing evidence is
        classified as ``unknown`` (conservative), never as stable.
        """
        obs = observation if isinstance(observation, dict) else {}
        prev = prev_observation if isinstance(prev_observation, dict) else {}
        url = str(obs.get("url", "") or "")
        dom = obs.get("dom_elements") or []
        purl = str(prev.get("url", "") or "")
        pdom = prev.get("dom_elements") or []

        if not url and not dom:
            stability = "unknown"
        elif purl and url != purl:
            # navigation since the previous chunk start: targets may be stale
            stability = "dynamic"
        elif pdom and abs(len(dom) - len(pdom)) > self._dom_delta_limit(
                len(dom), len(pdom)):
            stability = "dynamic"
        else:
            stability = "stable"

        recent_failure = None
        if self._consecutive_verified_successes == 0 and self._outcomes:
            last = self._outcomes[-1]
            if isinstance(last, dict) and last.get("kind") == "failure":
                recent_failure = str(last.get("reason", ""))[:160]

        return {
            "dom_stability": stability,
            "recent_failure": recent_failure,
            "recovery_streak": self._consecutive_verified_successes,
        }

    @staticmethod
    def _dom_delta_limit(n_after: int, n_before: int) -> int:
        larger = max(int(n_after or 0), int(n_before or 0), 1)
        return max(ACC_DOM_DELTA_FLOOR, int(ACC_DOM_DELTA_RATIO * larger))

    def _target_status(self, act: Any,
                       dom_elements: Optional[List[Any]]) -> str:
        """Classify one proposed target: ``validated`` | ``unknown``.

        Deterministic, evidence-based only:
        - target-free actions (wait/scroll/press and terminal/hitl) count as
          validated (they do not depend on a clickable target).
        - coordinate actions carry their own target.
        - SoM ``target_id`` must exist in the current DOM snapshot.
        - text targets must be found in the current DOM element fields.
        Unknown never enlarges a chunk and never joins a multi-action chunk.
        """
        if not isinstance(act, dict):
            return "unknown"
        atype = str(act.get("action_type") or act.get("action") or "").lower()
        if atype in ("wait", "scroll", "press", "done", "fail", "ask_user"):
            return "validated"
        x, y = act.get("x"), act.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return "validated"
        dom = dom_elements or []
        tid = act.get("target_id")
        if tid is not None:
            try:
                for el in dom:
                    if isinstance(el, dict) and el.get("id") == tid:
                        return "validated"
            except Exception:
                pass
            return "unknown"  # marker not confirmed in the current DOM
        from .long_horizon.stv import _content_words, _norm_text
        text = str(act.get("target") or "").strip()
        if not text:
            return "unknown"
        t_words = _content_words(text)
        if not t_words:
            return "unknown"
        t_norm = _norm_text(text)
        for el in dom:
            if not isinstance(el, dict):
                continue
            blob = _norm_text(" ".join(
                str(el.get(k, "") or "") for k in (
                    "text", "ariaLabel", "placeholder", "title", "name",
                    "id", "value")))
            if blob and (t_norm in blob or all(w in blob for w in t_words)):
                return "validated"
        return "unknown"

    def _classify_confidence(self, proposed_actions,
                             dom_elements) -> Tuple[str, int]:
        """(confidence level, length of the validated prefix).

        The evaluation window covers the configured hard cap (the controller
        truncates to ``allowed_chunk_size`` afterwards in any case).
        """
        proposed = [a for a in (proposed_actions or []) if isinstance(a, dict)]
        if not proposed:
            return "unknown", 0
        window = ACC_MAX_STATE_LIMIT if self.hard_cap <= ACC_MAX_STATE_LIMIT \
            else self.hard_cap
        statuses = []
        confidences = []
        for act in proposed[:window]:
            statuses.append(self._target_status(act, dom_elements))
            c = act.get("confidence")
            confidences.append(float(c) if isinstance(c, (int, float))
                               else None)
        cut = len(statuses)
        for i, s in enumerate(statuses):
            if s == "unknown":
                cut = i
                break
        if cut == 0:
            return "unknown", 0
        prefix_conf = confidences[:cut]
        if any(c is not None and c < ACC_CONFIDENCE_THRESHOLD
               for c in prefix_conf):
            return "low", cut
        if all(s == "validated" for s in statuses[:cut]):
            if all(c is not None and c >= ACC_CONFIDENCE_THRESHOLD
                   for c in prefix_conf):
                return "high", cut
            return "validated", cut
        return "unknown", cut

    # -- sizing (spec §7) ---------------------------------------------------
    def choose_chunk_size(self, signals: Dict[str, Any],
                          existing_limit: Optional[int] = None,
                          proposed_actions: Optional[List[Any]] = None,
                          dom_elements: Optional[List[Any]] = None,
                          ) -> Dict[str, Any]:
        """Return the deterministic execution limit decision.

        allowed = min(configured hard cap, existing limit, state-based
        limit, recovery preference, number of proposed actions).
        ``proposed_actions=None`` (planner not yet run, LH horizon mode)
        skips the count clamp; the count clamp then happens downstream in
        constrain_actions.
        """
        signals = signals or {}
        cap = self.hard_cap  # default 3; explicit validated configs may go up to 6
        stop: List[str] = []
        conf, _ = self._classify_confidence(proposed_actions, dom_elements)
        stability = str(signals.get("dom_stability", "unknown"))
        recent_failure = signals.get("recent_failure")

        state_limit = cap
        if recent_failure:
            state_limit = min(state_limit, 1)
            stop.append("recent_failure")
        if stability in ("dynamic", "unknown"):
            state_limit = min(state_limit, 1)
            stop.append(f"dom_{stability}")
        # Target confidence rules apply only when a concrete proposal exists
        # (planner already ran). Horizon-mode sizing (proposed_actions=None)
        # leaves target checking to the per-action truncation in
        # constrain_actions, so recovery-streak growth (Rule E) is preserved.
        if proposed_actions is not None:
            if conf == "low":
                state_limit = min(state_limit, 1)
                stop.append("low_target_confidence")
            elif conf == "unknown":
                state_limit = min(state_limit, 1)
                stop.append("target_confidence_unknown")
            if state_limit == cap:  # no conservative trigger fired
                if conf == "high":
                    stop.append("rule_d_confident")
                elif conf == "validated":
                    state_limit = min(state_limit, 2)
                    stop.append("rule_c_validated")

        # Rule E recovery: preferred size grows one verified success at a
        # time and always starts back at 1 after a failure.
        preferred = min(cap, ACC_INITIAL_PREFERRED_SIZE
                        + self._consecutive_verified_successes)
        existing = int(existing_limit) if existing_limit else cap
        count_limit = (len(proposed_actions)
                       if proposed_actions is not None else cap)
        allowed = min(cap, existing, state_limit, preferred, count_limit)

        seen: List[str] = []
        for reason in stop:
            if reason not in seen:
                seen.append(reason)
        decision = {
            "preferred_chunk_size": preferred,
            "allowed_chunk_size": max(ACC_MIN_CHUNK_SIZE, allowed),
            "planned_chunk_size": None,
            "executed_chunk_size": None,
            "stop_reason": ";".join(seen[:3]),
            "signals": {
                "dom_stability": stability,
                "target_confidence": conf,
                "recent_failure": recent_failure,
            },
        }
        self.stats["chunks_planned"] += 1
        return decision

    # -- chunk constraining (spec §8, §9) -----------------------------------
    def constrain_actions(self, signals: Optional[Dict[str, Any]],
                          proposed_actions: List[Any],
                          decision: Dict[str, Any],
                          dom_elements: Optional[List[Any]] = None,
                          boundary_checker=None,
                          ) -> Tuple[List[Any], Dict[str, Any]]:
        """Return the accepted ordered prefix of proposed actions.

        Rules applied (deterministic, order preserved, nothing reordered):
        - never more than ``allowed_chunk_size`` actions;
        - an action whose target cannot be validated against the current
          DOM ends the chunk BEFORE it (its result is needed first) unless
          it would leave the chunk empty (that single action still runs
          alone so existing grounding/verification handles it);
        - a boundary action (existing ActionChunker boundary policy passed
          in by the controller) may execute but must be the LAST accepted
          action (Rule B: navigation / submit / terminal / HITL);
        - truncation never deletes required task steps from the overall
          plan: dropped actions re-enter the next planning cycle.
        """
        decision = dict(decision or {})
        proposed = list(proposed_actions or [])
        allowed = int(decision.get("allowed_chunk_size") or 0)
        if allowed <= 0:
            allowed = len(proposed)
        stop = str(decision.get("stop_reason", "") or "")
        accepted: List[Any] = []
        for act in proposed[:allowed]:
            status = self._target_status(act, dom_elements)
            if status == "unknown" and accepted:
                stop = (stop + ";unverifiable_target") if stop \
                    else "unverifiable_target"
                decision["unverified_tail"] = True
                break
            accepted.append(act)
            if boundary_checker is not None:
                try:
                    is_boundary, name = boundary_checker(act)
                except Exception:
                    is_boundary, name = False, ""
                if is_boundary:
                    # Rule B: this action may execute; the chunk ends after
                    # it so the controllers re-observe / re-verify.
                    stop = (stop + f";boundary:{name}") if stop \
                        else f"boundary:{name}"
                    decision["boundary_end"] = str(name or "boundary")
                    break
        decision["planned_chunk_size"] = len(accepted)
        decision["stop_reason"] = stop
        decision["actions_dropped"] = max(0, len(proposed) - len(accepted))
        self.stats["actions_accepted"] += len(accepted)
        self.stats["actions_dropped"] += int(decision["actions_dropped"])
        return accepted, decision

    # -- outcome recording (spec §14) ----------------------------------------
    def record_outcome(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Update the bounded history and the recovery streak.

        Outcome kinds used by the controllers:
        - ``failure``: execution / verification / contract failure,
          unverified terminal-done rejection (streak resets to 0);
        - ``success``: a successfully executed AND verified action (streak
          +1, preferred size may grow by one step);
        - ``unknown``: verification could not be established (streak DECAYS
          by one preferred-size step — spec §6.7; never treated as verified
          success and never left unchanged to grow later);
        - anything else (``navigation``, ``boundary``, ``ask_user``,
          ``terminal``, ``info``): recorded for observability only.

        Structured, additive fields a controller may attach:
        - ``contract``: expected-transition result (passed / failed /
          unknown / no_contract) tallied into ``contract_results``;
        - ``goal_check_verdict``: a ``classify_goal_check`` verdict tallied
          into ``goal_check_verdicts``.
        """
        rec = dict(outcome or {})
        rec["kind"] = str(rec.get("kind", "info"))
        self._outcomes.append(rec)
        contract = str(rec.get("contract", "") or "")
        if contract in self.contract_results:
            self.contract_results[contract] += 1
        gc_verdict = str(rec.get("goal_check_verdict", "") or "")
        if gc_verdict:
            self.goal_check_verdicts[gc_verdict] = \
                self.goal_check_verdicts.get(gc_verdict, 0) + 1
        if rec["kind"] in ("navigation", "boundary"):
            name = rec["kind"] if not rec.get("reason") else \
                f"{rec['kind']}:{rec['reason']}"
            self.interruptions[name] = self.interruptions.get(name, 0) + 1
        if rec["kind"] == "failure":
            self._consecutive_verified_successes = 0
            self.stats["failures"] += 1
        elif rec["kind"] == "success":
            self._consecutive_verified_successes += 1
            self.stats["verified_successes"] += 1
        elif rec["kind"] == "unknown":
            # §6.7: uncertain verification must reset or decay stability.
            # Decay conservatively by exactly one preferred-size step.
            if self._consecutive_verified_successes > 0:
                self._consecutive_verified_successes -= 1
            self.stats["unknown_verifications"] += 1
        return rec

    def record_executed_size(self, size: Any) -> None:
        """Record one post-execution chunk-size observation (additive)."""
        try:
            key = max(0, int(size))
        except (TypeError, ValueError):
            return
        self.executed_sizes[key] = self.executed_sizes.get(key, 0) + 1

    def acc_metrics(self) -> Dict[str, Any]:
        """Additive ACC observability snapshot (opt-in path only).

        Contains only actually-recorded event counts; an empty section
        means "no such events occurred", never a fabricated zero-filled
        success story. Consumed by the scenario benchmark and by terminal
        outcome records.
        """
        return {
            "enabled": self.enabled,
            "hard_cap": self.hard_cap,
            "stats": dict(self.stats),
            "contract_results": dict(self.contract_results),
            "goal_check_verdicts": dict(self.goal_check_verdicts),
            "executed_chunk_sizes": dict(self.executed_sizes),
            "interruptions": dict(self.interruptions),
            "recovery_streak": self._consecutive_verified_successes,
        }

    def reset_if_needed(self) -> None:
        """Clear per-task adaptive state (called at task boundaries)."""
        self._consecutive_verified_successes = 0
        self._outcomes.clear()
        self.stats["resets"] += 1

    @property
    def consecutive_verified_successes(self) -> int:
        return self._consecutive_verified_successes


# -- expected-transition contract (Priority 2) --------------------------------

def _observable_page_evidence(after_obs: Any) -> bool:
    """True when the after-observation carries a material page-evidence
    space for absence claims (visible text of at least the floor length).

    DOM elements are checked separately; a URL alone never counts as page
    evidence for a must_not_happen claim.
    """
    obs = after_obs if isinstance(after_obs, dict) else {}
    txt = str(obs.get("visible_text", "") or "").strip()
    return len(txt) >= ACC_MNH_EVIDENCE_FLOOR


def _must_change_property_changed(norm: Dict[str, Any], before_pool: str,
                                  before_dom: List[Any],
                                  note_after: str) -> Tuple[bool, str]:
    """A must_change PROPERTY item must hold after AND not have held before.

    Uses the same ``expectation.verify_item`` machinery with the
    before-pool substituted for the after-pool, so no new verification
    logic is introduced. The caller guarantees the before-state has
    observable evidence (otherwise it returns unknown upstream).
    """
    try:
        before_ok, _ = _expm_verify_fn(norm, before_pool, before_pool,
                                       before_dom)
    except Exception as e:  # pragma: no cover - evidence machinery issue
        return False, f"unknown: before-state check cannot be observed: {e}"
    if before_ok:
        return False, ("property already held before the action; no "
                       "property change observed")
    return True, (f"{note_after}; property not present before the action")


def _expm_verify_fn(norm, pool_a, pool_b, dom):
    """Indirection so both verify calls share one import point."""
    from .long_horizon.expectation import verify_item as _vi
    return _vi(norm, pool_a, pool_b, dom)

def verify_expected_transition(before_obs: Optional[Dict[str, Any]],
                               after_obs: Optional[Dict[str, Any]],
                               transition: Any,
                               action: Any = None) -> Dict[str, Any]:
    """Verify a planner ``expected_transition`` contract deterministically.

    Composes the EXISTING long-horizon STV evidence machinery (no new LLM
    calls, no duplicated verification logic):
    - ``expectation.normalize_expectation`` / ``verify_item`` for
      must_change / must_be_true / must_remain_true items, DOM-first with
      page-text fallback, exactly like ``STV.decide`` consumes these;
    - ``stv._forbidden_hit`` for must_not_happen items.

    Returns ``{"result": passed|failed|unknown|no_contract,
    "failed_item", "note", "checks"}``:
    - ``no_contract``: no contract present (or every field empty) — the
      caller defers to the existing StateVerifier verdict (behavior
      identical to before this feature).
    - ``failed``: a concrete requirement is disproven by observed evidence.
    - ``unknown``: evidence is missing / a predicate cannot be evaluated /
      the contract is malformed — never converted into verified success.

    Hardened semantics (no existing helper is modified):
    - ``must_change`` PROPERTY items must hold after the action AND not
      have held before it (checked with the same ``verify_item`` against
      the before-state). The generic whole-page CHANGED item keeps its
      documented before-after inequality semantics. An unrelated page
      mutation or an already-true property therefore never satisfies a
      specific ``must_change`` requirement.
    - ``must_not_happen`` only passes when the forbidden condition was
      reliably observable in its absence (DOM elements present, or at
      least ``ACC_MNH_EVIDENCE_FLOOR`` characters of visible text). A
      forbidden phrase with no observable content, or absence claimed from
      a thin evidence space, yields ``unknown``.
    - Evidence association is per-action: callers pass that action's own
      pre/post observations, so a later transition can never satisfy an
      earlier action's contract.
    """
    result = {"result": "no_contract", "failed_item": "", "note": "",
              "checks": []}
    if transition is None or transition == {} or (
            isinstance(transition, (list, tuple)) and not transition):
        return result
    if not isinstance(transition, dict):
        result["result"] = "unknown"
        result["note"] = (f"malformed expected_transition type "
                          f"{type(transition).__name__}")
        return result

    exp: Dict[str, List[Any]] = {}
    for key in ("must_change", "must_be_true", "must_remain_true",
                "must_not_happen"):
        vals = transition.get(key, [])
        if isinstance(vals, str):
            vals = [vals]  # existing convention: strings wrap into lists
        elif isinstance(vals, (list, tuple)):
            vals = list(vals)
        elif vals is None:
            vals = []
        else:
            result["result"] = "unknown"
            result["note"] = (f"malformed transition field {key!r} "
                              f"({type(vals).__name__})")
            return result
        exp[key] = vals

    items: List[Tuple[str, Any]] = []
    for key in ("must_change", "must_be_true", "must_remain_true",
                "must_not_happen"):
        for item in exp[key]:
            if isinstance(item, dict) or str(item or "").strip():
                items.append((key, item))
    if not items:
        return result  # structurally present but empty: no contract

    try:
        from .long_horizon import stv as _stv
        from .long_horizon import expectation as _expm
    except Exception as e:  # pragma: no cover - environment issue
        result["result"] = "unknown"
        result["note"] = f"STV evidence machinery unavailable: {e}"
        return result

    before_pool = _stv._evidence_pool(before_obs or {})
    before_dom = (list(before_obs.get("dom_elements") or [])
                  if isinstance(before_obs, dict) else [])
    after_obs = after_obs if isinstance(after_obs, dict) else {}
    after_pool = _stv._evidence_pool(after_obs)
    dom = after_obs.get("dom_elements") or []
    if not str(after_pool or "").strip():
        result["result"] = "unknown"
        result["note"] = "no after-state evidence (URL/DOM/text all empty)"
        return result

    checks: List[Dict[str, Any]] = []
    for key, item in items:
        if isinstance(item, dict) and item.get("predicate"):
            target = item.get("target", "")
            norm = {"target": (target.split() if isinstance(target, str)
                               else list(target or [])),
                    "predicate": str(item.get("predicate", "")),
                    "value": item.get("value", ""), "raw": str(item)}
            label = str(item.get("value", item))[:120]
        else:
            norm = _expm.normalize_expectation(str(item or ""))
            label = str(item)[:120]
        if key == "must_not_happen":
            forbidden = str(item if not isinstance(item, dict)
                            else (item.get("value", "") or str(item)))
            hit = _stv._forbidden_hit(forbidden, after_pool,
                                      after_obs.get("url", ""))
            if hit:
                ok, note = False, "forbidden outcome observed"
            elif not _stv._content_words(str(forbidden or "")):
                # Absence of nothing cannot be evidence: a forbidden
                # requirement that names no observable content is
                # unresolvable, never a silent pass (§3.3).
                ok, note = False, ("unknown: forbidden condition cannot be "
                                   "observed (no observable content named)")
            elif len(dom) == 0 and not _observable_page_evidence(after_obs):
                # Neither DOM enumeration nor page text exists, so the
                # relevant condition was NOT reliably observable; absence
                # alone does not prove it did not occur (§3.3).
                ok, note = False, ("unknown: forbidden condition cannot be "
                                   "observed in the available evidence")
            else:
                ok, note = True, ("forbidden outcome not observed "
                                  "(relevant evidence space present)")
        else:
            ok, note = _expm.verify_item(norm, before_pool, after_pool, dom)
            if ok and key == "must_change" and norm is not None \
                    and norm.get("predicate") != "CHANGED":
                if not str(before_pool or "").strip() and not before_dom:
                    # The before-state cannot be observed, so "did the
                    # property actually change?" is unanswerable (§3.1).
                    ok, note = False, ("unknown: before-state cannot be "
                                       "observed; cannot confirm the "
                                       "property changed")
                else:
                    ok, note = _must_change_property_changed(
                        norm, before_pool, before_dom, note)
        unknown = (not ok) and _is_unresolvable_note(str(note))
        checks.append({"requirement": label, "contract": key,
                       "result": ("passed" if ok else
                                  "unknown" if unknown else "failed"),
                       "note": str(note)[:160]})
        if ok:
            continue
        result["result"] = "unknown" if unknown else "failed"
        result["failed_item"] = label
        result["note"] = f"{key}: {note}"[:200]
        result["checks"] = checks
        return result

    result["result"] = "passed"
    result["checks"] = checks
    return result
