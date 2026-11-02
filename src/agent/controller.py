"""Agent controller: observe → ground → validate → execute → verify → recover/finish."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image
import io
import time

from .memory import AgentMemory
from .recovery import RecoveryManager
from .action_chunking import ActionChunker, DEFAULT_CHUNK_SIZE
from .adaptive_chunking import (
    AdaptiveChunkController,
    ACC_DEFAULT_MAX_CHUNK_SIZE,
    verify_expected_transition,
    classify_goal_check,
    GCH_VERIFIED_SUCCESS,
    GCH_UNAVAILABLE,
)

logger = logging.getLogger(__name__)


class AgentController:
    def __init__(
        self,
        browser,
        observer,
        grounder,
        executor,
        verifier=None,
        memory: Optional[AgentMemory] = None,
        recovery: Optional[RecoveryManager] = None,
        trajectory_logger=None,
        max_steps: int = 20,
        task: str = "",
        goal_verifier=None,
        enable_goal_check: bool = True,
        ask_handler=None,
        max_asks: int = 3,
        action_chunking: bool = False,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        # Experimental Adaptive Chunk Controller (opt-in CLI feature; OFF
        # by default so every existing behavior is unchanged). Only the
        # chunk execution path consults ACC; the baseline loop ignores it.
        adaptive_chunking: bool = False,
        adaptive_chunk_max_size: int = ACC_DEFAULT_MAX_CHUNK_SIZE,
    ):
        self.browser = browser
        self.observer = observer
        self.grounder = grounder
        self.executor = executor
        self.verifier = verifier
        self.memory = memory or AgentMemory()
        self.recovery = recovery or RecoveryManager()
        self.trajectory_logger = trajectory_logger
        self.max_steps = max_steps
        self.task = task
        self.ask_handler = ask_handler
        self.max_asks = max_asks
        self._ask_count = 0
        self.action_chunking = bool(action_chunking)
        self.chunk_size = max(2, min(int(chunk_size or DEFAULT_CHUNK_SIZE), 6)) if action_chunking else 3
        self.chunker = ActionChunker(enabled=self.action_chunking, chunk_size=self.chunk_size)
        self.adaptive_chunking = bool(adaptive_chunking)
        self.acc = AdaptiveChunkController(
            enabled=self.adaptive_chunking, hard_cap=adaptive_chunk_max_size)
        # Most recent chunk-start observation, used for ACC stability
        # signals and goal verification context (chunk mode only).
        self._last_chunk_start_obs: Optional[Dict[str, Any]] = None
        self._run_start_obs: Optional[Dict[str, Any]] = None
        self.actions_count = 0
        # noVNC takeover HITL (default off): when True, ask_user turns wait
        # for human browser action instead of blocking on stdin. Env
        # HITL_TAKEOVER=1 also enables it (CLI --browser-takeover).
        self.takeover_mode = False
        self.takeover_timeout_s = 120.0
        self.takeover_poll_s = 2.0
        # Per-question HITL transport: None (= resolve HITL_ASK_ROUTE env,
        # else "terminal"), or explicit "terminal" / "takeover" / "auto"
        # (credentials -> takeover, rest -> terminal).
        self.ask_route = None
        self.intent_gate = None  # set via set_intent_gate(); OFF by default
        self.hybrid_gate = None  # set via set_hybrid_gate(); OFF by default
        # CAPTCHA safety gate (standalone detector -> existing HITL).
        # None = disabled (default): behavior identical to before.
        self.captcha_gate = None  # set via set_captcha_gate()
        self.captcha_max_rounds = 3
        # Download-completion detection: count of browser downloads already
        # accounted for. A NEW download on a download-type task ends the run.
        self._downloads_seen = self._download_count()
        if goal_verifier is None and enable_goal_check:
            from verification.goal_verifier import GoalVerifier
            goal_verifier = GoalVerifier()
        self.goal_verifier = goal_verifier if enable_goal_check else None

    def run(self, task: Optional[str] = None) -> Dict[str, Any]:
        task = task or self.task
        # Adaptive chunking (opt-in): per-task adaptive state starts clean;
        # no effect when the feature is disabled.
        if self.adaptive_chunking and self.acc is not None:
            self.acc.reset_if_needed()
        logger.info(f"Starting task: {task} (max_steps={self.max_steps})")
        if self.trajectory_logger:
            self.trajectory_logger.set_metadata({
                "task": task,
                "grounding_method": getattr(self.grounder, "name", "unknown"),
                "max_steps": self.max_steps,
            })

        observation = self.observer.observe(save_screenshot=False)
        self._run_start_obs = observation
        start_url = observation.get("url", "")
        for step in range(1, self.max_steps + 1):
            result = self._step(step, task, observation, start_url)
            observation = result["next_observation"]
            if result.get("finished"):
                return result["outcome"]
            if observation is None:
                break

        outcome = {"success": False, "reason": "Step budget exhausted",
                   "steps": self.max_steps, "termination": "budget_exhausted"}
        if self.trajectory_logger:
            self.trajectory_logger.log_result(outcome)
        return outcome

    def _step(self, step: int, task: str, observation: Dict[str, Any],
              start_url: str = "") -> Dict[str, Any]:
        # 0. Early check for delayed downloads from previous steps
        if "download" in (task or "").lower():
            count = self._download_count()
            if count > self._downloads_seen:
                new = list(getattr(self.browser, "downloads", []))[self._downloads_seen:]
                self._downloads_seen = count
                names = [d.get("filename", "?") for d in new]
                logger.info(f"Step {step}: delayed file download detected before grounding: {names}")
                try:
                    saved = self.browser.save_downloads()
                except Exception as e:
                    logger.warning(f"saving downloads failed: {e}")
                    saved = []
                return self._finish(
                    step, True,
                    f"Goal achieved: downloaded {', '.join(names)}"
                    + (f" -> {saved[0]}" if saved else ""),
                    observation,
                    termination="download_completed"
                )

        from actions.action_schema import parse_action
        from pydantic import ValidationError

        image = Image.open(io.BytesIO(observation["screenshot_bytes"]))
        history = self.memory.all()

        # 1. Ground
        # Adaptive chunking (opt-in): measure planning time / VLM calls for
        # ChunkMetrics (populated only on this path; legacy metrics 0).
        _acc_t0 = (time.time()
                   if (self.action_chunking and self.adaptive_chunking)
                   else None)
        _acc_vlm0 = (getattr(self.grounder, "vlm_calls", None)
                     if _acc_t0 is not None else None)
        try:
            prediction = self.grounder.ground(
                image, task, observation.get("dom_elements", []), history=history,
            )
        except Exception as e:
            logger.error(f"Grounding crashed at step {step}: {e}")
            return self._finish(step, False, f"Grounder error: {e}", observation,
                                termination="grounder_error")

        if _acc_t0 is not None:
            self.chunker.metrics.time_planning += time.time() - _acc_t0
            _acc_vlm1 = getattr(self.grounder, "vlm_calls", None)
            if isinstance(_acc_vlm0, int) and isinstance(_acc_vlm1, int):
                self.chunker.metrics.vlm_calls += max(0, _acc_vlm1 - _acc_vlm0)

        if hasattr(self.grounder, "advance") and not self.action_chunking:
            try:
                self.grounder.advance(1)
            except Exception:
                pass

        if self.action_chunking:
            raw_acts = None
            if isinstance(prediction, dict) and "actions" in prediction and isinstance(prediction["actions"], list):
                raw_acts = prediction["actions"]
            elif isinstance(prediction, list):
                raw_acts = prediction
            if raw_acts:
                return self._execute_chunk(step, task, observation, raw_acts[:self.chunk_size], start_url)
            elif isinstance(prediction, dict) and ("action" in prediction or "action_type" in prediction):
                return self._execute_chunk(step, task, observation, [prediction], start_url)

        # 2. SoM → resolve marker to coords
        if hasattr(self.grounder, "to_action"):
            try:
                action_dict = self.grounder.to_action(prediction)
            except Exception as e:
                return self._finish(step, False, f"to_action error: {e}", observation,
                                termination="grounder_error")
        else:
            action_dict = prediction

        # annotated screenshot for logging (SoM)
        annotated = None
        if isinstance(prediction, dict):
            annotated = prediction.get("annotated_screenshot")

        # 3. Validate via Pydantic
        try:
            action = parse_action(action_dict)
        except ValidationError as e:
            logger.warning(f"Step {step}: malformed action rejected: {e}")
            verification = {"success": False, "reason": f"Invalid action: {e}"}
            self.memory.add(action_dict, {"success": False, "error": str(e)}, observation.get("url", ""))
            self._log(step, observation, action_dict, verification, annotated)
            if self.recovery.should_recover(verification):
                self.recovery.record_recovery("re-observe after invalid action")
                return {"next_observation": self.observer.observe(save_screenshot=False)}
            return self._finish(step, False, "Invalid action, recovery exhausted", observation,
                                termination="action_exhausted")

        # 3b. Human-in-the-loop: intercept BEFORE browser/STV.
        # ask_user never touches the browser: no executor call, no
        # re-observe screenshot, no STV DOM/URL diff. Success is hardcoded
        # so recovery budgets are untouched (consumes a step, not a retry).
        is_ask = getattr(action, "is_ask_user", None)
        try:
            is_ask = bool(is_ask()) if callable(is_ask) else bool(is_ask)
        except Exception:
            at = getattr(action, "action_type", "")
            is_ask = at == "ask_user"
        if is_ask:
            return self._handle_ask_user(step, observation, action,
                                         action_dict, annotated)

        # 3b2. CAPTCHA safety gate (default OFF). A blocking challenge
        # pauses autonomous execution via the existing HITL path BEFORE
        # any further grounding/execution; cleared state re-enters the
        # normal flow (no action is repeated).
        gated = self._maybe_captcha_gate(step, observation, annotated)
        if gated is not None:
            return gated

        # 3c. EXPERIMENTAL intent gate (default OFF). Consults a text-only
        # judge for potentially information-requiring actions BEFORE the
        # browser. ASK verdicts funnel into the existing HITL path above.
        gated = self._maybe_intent_gate(step, task, observation, action,
                                        action_dict, annotated)
        if gated is not None:
            return gated

        # 4. Terminal?
        if action.is_terminal():
            ok = action.is_success()
            verification = {"success": True, "reason": f"Agent emitted {action.action_type}"}
            self.memory.add(action_dict, {"success": True}, observation.get("url", ""))
            # Confirm a "done" claim against the actual page state (recorded,
            # never overrides the agent's own termination decision).
            done_check = None
            if ok and self.goal_verifier is not None:
                try:
                    done_check = self.goal_verifier.check(
                        task, observation, start_url,
                        prev_observation=observation, verification=verification)
                except Exception as e:
                    done_check = {"achieved": False, "done": False,
                                  "reason": f"goal check error: {e}"}
            self._log(step, observation, action_dict, verification, annotated,
                      goal_check=done_check)
            term = "agent_done" if ok else "agent_fail"
            return self._finish(step, ok, f"Agent done: {action.reason or action.action_type}",
                                observation, verification=verification,
                                termination=term)

        # 5. Execute
        if "download" in (task or "").lower():
            done = self._maybe_download_done(step, task, observation, action_dict, annotated)
            if done is not None:
                return done

        self.actions_count += 1
        exec_result = self.executor.execute(action)

        # 6. Re-observe
        try:
            next_obs = self.observer.observe(save_screenshot=False)
        except Exception as e:
            return self._finish(step, False, f"Re-observe failed: {e}", observation,
                                termination="browser_error")
        next_obs["exec_success"] = exec_result.get("success", False)
        next_obs["exec_error"] = exec_result.get("error")

        # 6b. CAPTCHA safety gate (default OFF). A challenge that
        # appeared as a result of the just-executed action routes to the
        # existing HITL path BEFORE verification; on clearance the fresh
        # post-human observation flows into normal STV verification.
        gated = self._maybe_captcha_gate(step, next_obs, annotated)
        if gated is not None:
            if gated.get("finished"):
                return gated
            next_obs = gated.get("next_observation", next_obs)

        # 6c. Download completion: a file download triggered by this action
        # on a download task is a hard success signal (no visible page
        # change is expected, so STV/goal heuristics cannot see it).
        done = self._maybe_download_done(step, task, next_obs, action_dict,
                                         annotated)
        if done is not None:
            return done

        # 7. Verify
        if self.verifier:
            verification = self.verifier.verify(observation, next_obs, action_dict, task)
        else:
            verification = {"success": exec_result.get("success", False),
                            "reason": "no verifier (exec status only)"}
        verification["exec_success"] = exec_result.get("success", False)

        # 7b. Goal check — stop early if the goal is already achieved,
        # before spending another VLM call.
        goal_check: Dict[str, Any] = {"achieved": False, "done": False,
                                      "reason": "goal check disabled"}
        if self.goal_verifier is not None:
            try:
                goal_check = self.goal_verifier.check(
                    task, next_obs, start_url,
                    prev_observation=observation, verification=verification)
            except Exception as e:
                logger.warning(f"Goal check failed at step {step}: {e}")
                goal_check = {"achieved": False, "done": False,
                              "reason": f"goal check error: {e}"}

        self.memory.add({**action_dict, "reason": action.reason},
                        {"success": verification["success"],
                         "error": verification.get("reason"),
                         "verification": verification["success"]},
                        next_obs.get("url", ""),
                        dom_count_before=len(observation.get("dom_elements", [])))
        self._log(step, next_obs, action_dict, verification, annotated,
                  goal_check=goal_check)

        if goal_check.get("achieved"):
            logger.info(f"Step {step}: goal achieved — terminating successfully")
            return self._finish(
                step, True, f"Goal achieved: {goal_check.get('reason')}",
                next_obs, verification=verification,
                termination="goal_achieved_terminated")

        # 8. Loop detection
        loop = self.memory.detect_loop()
        if loop:
            logger.warning(f"Step {step}: loop detected {loop}")
            if self.recovery.should_recover({"success": False}):
                self.recovery.record_recovery("loop → re-observe and re-ground")
                return {"next_observation": next_obs}
            return self._finish(step, False, "Loop detected, recovery exhausted", next_obs,
                                termination="loop_exhausted")

        # 9. Recovery on failure
        if not verification["success"]:
            if self.recovery.should_recover(verification):
                self.recovery.record_recovery(f"re-ground after: {verification.get('reason')}")
                self.recovery.reset_step_counters()
                return {"next_observation": next_obs}
            return self._finish(step, False,
                                f"Verification failed: {verification.get('reason')}", next_obs,
                                termination="verification_exhausted")

        self.recovery.reset_step_counters()
        return {"next_observation": next_obs}

    def _handle_ask_user(self, step, observation, action, action_dict,
                           annotated):
        """Shared HITL turn: prompt, record Q&A, reuse observation.

        Used both for VLM-emitted `ask_user` and for experimental
        intent-gate ASK verdicts. Never touches the browser or STV;
        never increments recovery budgets (shares max_asks instead).
        """
        from .hitl import resolve_ask_user, MAX_ASKS_DEFAULT
        limit = self.max_asks if self.max_asks is not None else MAX_ASKS_DEFAULT
        if self._ask_count >= limit:
            return self._finish(
                step, False,
                f"ask_user budget exhausted ({self._ask_count}/{limit})",
                observation, termination="ask_exhausted")
        self._ask_count += 1
        # Dashboard hook: handlers exposing for_action(action_dict) get the
        # action context (e.g. CAPTCHA typing); plain callables are used
        # as-is. Backwards compatible.
        _handler = self.ask_handler
        _for_action = getattr(_handler, "for_action", None)
        if callable(_for_action):
            try:
                _handler = _for_action(action_dict) or _handler
            except Exception as e:
                logger.warning("ask for_action hook failed (%s); "
                               "using base handler", e)
        if self._ask_transport(action_dict) == "takeover":
            taken = self._handle_browser_takeover(
                step, observation, action_dict, annotated)
            if taken is not None:
                return taken
            # Timeout with no page change: fall back to terminal ask so the
            # run never hangs silently.
            logger.info("Takeover produced no change; "
                        "falling back to terminal ask")
        resolved = resolve_ask_user(action, action_dict, _handler)
        verification = resolved["verification"]
        answered_dict = resolved["action_dict"]
        self.memory.add(
            answered_dict, {"success": True}, observation.get("url", ""))
        self._log(step, observation, answered_dict, verification,
                  annotated)
        # Reset step counters like a success so a prior failure streak
        # does not leak into the post-answer turn; do NOT record_recovery.
        self.recovery.reset_step_counters()
        try:
            fresh_obs = self.observer.observe(save_screenshot=False)
        except Exception:
            fresh_obs = observation
        return {"next_observation": fresh_obs}

    def _takeover_on(self) -> bool:
        if getattr(self, "takeover_mode", False):
            return True
        try:
            import os as _os
            return _os.getenv("HITL_TAKEOVER", "").strip().lower() in (
                "1", "true", "yes", "on")
        except Exception:
            return False

    def _ask_transport(self, action_dict) -> str:
        """Per-question transport: legacy takeover flag wins; else route."""
        if self._takeover_on():
            return "takeover"
        try:
            from .hitl import route_ask_transport
            route = getattr(self, "ask_route", None) or ""
            if not route:
                import os as _os
                route = _os.getenv("HITL_ASK_ROUTE", "terminal")
            d = action_dict if isinstance(action_dict, dict) else {}
            return route_ask_transport(d.get("question", ""),
                                       d.get("reason", ""), route)
        except Exception:
            return "terminal"

    def _handle_browser_takeover(self, step, observation, action_dict,
                                 annotated):
        """noVNC takeover turn: wait for human browser action, no stdin.

        Returns a step-result dict on human action, or None on timeout
        (caller falls back to terminal ask). Unlike terminal HITL, the
        FRESH observation is returned so grounding sees the human's work.
        Memory/trajectory record the change summary only -- never typed
        secrets.
        """
        if getattr(self, "observer", None) is None:
            logger.warning("Takeover requested with no observer; "
                           "falling back to terminal ask")
            return None
        from .hitl import (wait_for_browser_takeover,
                            TAKEOVER_TIMEOUT_S_DEFAULT,
                            TAKEOVER_POLL_S_DEFAULT)
        try:
            timeout_s = float(getattr(self, "takeover_timeout_s",
                                      TAKEOVER_TIMEOUT_S_DEFAULT))
            poll_s = float(getattr(self, "takeover_poll_s",
                                   TAKEOVER_POLL_S_DEFAULT))
        except (TypeError, ValueError):
            logger.warning("Bad takeover timing config; using defaults")
            timeout_s, poll_s = (TAKEOVER_TIMEOUT_S_DEFAULT,
                                 TAKEOVER_POLL_S_DEFAULT)
        question = (action_dict or {}).get("question", "")
        result = wait_for_browser_takeover(
            str(question or "Human input needed in the browser"),
            self.observer, observation,
            timeout_s=timeout_s, poll_interval_s=poll_s)
        if not result.get("resumed"):
            return None
        fresh = result.get("observation") or observation
        answered_dict = dict(action_dict or {})
        answered_dict["human_answer"] = "<acted in browser via noVNC>"
        answered_dict["reason"] = (
            f"Human acted in browser (noVNC): "
            f"{result.get('change_summary', 'page changed')[:200]}")
        verification = {"success": True,
                        "reason": "takeover: human acted in browser",
                        "takeover": True}
        try:
            url = fresh.get("url", observation.get("url", ""))
        except Exception:
            url = ""
        self.memory.add(answered_dict, {"success": True}, url)
        self._log(step, fresh, answered_dict, verification, annotated)
        self.recovery.reset_step_counters()
        return {"next_observation": fresh}

    def set_intent_gate(self, gate) -> None:
        """Attach the EXPERIMENTAL intent gate (None = disabled, default)."""
        self.intent_gate = gate

    def set_hybrid_gate(self, gate) -> None:
        """Attach the EXPERIMENT-2 hybrid gate (None = disabled, default)."""
        self.hybrid_gate = gate

    def set_captcha_gate(self, gate) -> None:
        """Attach the CAPTCHA safety gate (None = disabled, default).

        When attached, a BLOCKING challenge pauses the turn via the
        existing HITL path (`reason=captcha_detected`); anything else
        flows through unchanged.
        """
        self.captcha_gate = gate

    def _maybe_captcha_gate(self, step, observation, annotated):
        """CAPTCHA hook. Returns None to proceed normally, a
        ``{"next_observation": fresh}`` dict when a blocking challenge
        was human-cleared, or a finished outcome when it was not.
        Never raises: detector/page failures fail open (logged).
        """
        gate = getattr(self, "captcha_gate", None)
        if gate is None or not getattr(gate, "enabled", True):
            return None
        max_rounds = getattr(gate, "max_rounds", None)
        try:
            max_rounds = max(1, int(
                getattr(self, "captcha_max_rounds", 0) or max_rounds or 1))
        except (TypeError, ValueError):
            max_rounds = 1
        saw_block = False
        for _ in range(max_rounds):
            try:
                page = self.browser.get_page()
            except Exception as e:
                logger.error("[CAPTCHA_GATE] no live page (%s); "
                             "continuing autonomous", e)
                return None
            res = gate.check(page)
            status = getattr(res, "status", "clear")
            if status in ("clear", "disabled"):
                if not saw_block:
                    return None
                # A block was seen and human-cleared: the action grounded
                # on the pre-CAPTCHA observation is stale, so end the step
                # with the fresh observation (next step re-grounds).
                # Never resume by executing the stale action.
                logger.info("[CAPTCHA_GATE] cleared after human "
                            "intervention; resuming with fresh observation")
                return {"next_observation": observation}
            if status == "error":
                # Distinguishable from "no CAPTCHA": logged, autonomous
                # execution continues (fail-open by design).
                logger.error("[CAPTCHA_GATE] check error (%s); "
                             "continuing autonomous",
                             getattr(res, "error", "?"))
                return None
            # BLOCKING: pause via the existing HITL machinery.
            saw_block = True
            from actions.action_schema import parse_action
            ask_dict = gate.build_ask_dict(getattr(res, "result", None))
            try:
                ask_action = parse_action(ask_dict)
            except Exception as e:
                logger.error("[CAPTCHA_GATE] bad ask dict (%s); "
                             "continuing autonomous", e)
                return None
            ret = self._handle_ask_user(step, observation, ask_action,
                                        ask_dict, annotated)
            if ret.get("finished"):
                return ret
            observation = ret.get("next_observation", observation)
            # The human solved in the browser; the pre-HITL observation
            # is stale, so re-observe before re-detecting. The same live
            # page/session is reused (never closed/reloaded by the gate).
            try:
                observation = self.observer.observe(save_screenshot=False)
            except Exception as e:
                return self._finish(
                    step, False,
                    f"Re-observe failed after CAPTCHA HITL: {e}",
                    observation, termination="browser_error")
            logger.info("[CAPTCHA_GATE] Human intervention completed; "
                        "rechecking CAPTCHA...")
        logger.info("[CAPTCHA_GATE] CAPTCHA still blocking after human "
                    "intervention; not resuming autonomous execution")
        return self._finish(
            step, False,
            "CAPTCHA still blocking after human intervention "
            "(captcha_unresolved)",
            observation, termination="captcha_unresolved")

    def _intent_gate_on(self) -> bool:
        if self.intent_gate is None:
            return False
        try:
            from .hybrid_gate import get_intent_gate_mode
            if get_intent_gate_mode() == "llm":
                return True
            from .intent_gate import is_intent_gate_enabled
            return bool(is_intent_gate_enabled())
        except Exception:
            return False

    def _maybe_intent_gate(self, step, task, observation, action,
                           action_dict, annotated):
        """EXPERIMENTAL hook. Returns a step-result dict to short-circuit,
        or None to proceed normally. Never raises: failures fall to ALLOW."""
        try:
            from .hybrid_gate import get_intent_gate_mode
            mode = get_intent_gate_mode()
        except Exception:
            mode = "off"
        if mode == "hybrid":
            if self.hybrid_gate is None:
                return None
            return self._hybrid_check(step, task, observation, action)
        if not self._intent_gate_on():
            return None
        try:
            from .intent_gate import (should_consult_gate, build_dom_context,
                                      find_action_target)
            dom = observation.get("dom_elements", [])
            consult, why, target = should_consult_gate(
                action, action_dict, dom)
            if target is not None:
                logger.info(
                    "[INTENT-GATE] action=%s target_role=%s target_label=%.40s",
                    getattr(action, "action_type", "?"),
                    (target.get("role") or target.get("tag") or "?"),
                    (target.get("text") or target.get("ariaLabel")
                     or target.get("placeholder") or ""))
            if not consult:
                logger.info("[INTENT-GATE] bypass action=%s reason=%s",
                            getattr(action, "action_type", "?"), why)
                return None
            logger.info("[INTENT-GATE] enabled=True action=%s reason=%s",
                        getattr(action, "action_type", "?"), why)
            verdict = self.intent_gate.validate_action_intent(
                task, self.memory.all(), action,
                build_dom_context(target, dom))
            if verdict.get("decision") != "ASK":
                logger.info("[INTENT-GATE] decision=ALLOW")
                return None
            logger.info("[INTENT-GATE] decision=ASK missing_field=%s",
                        verdict.get("missing_field", ""))
            logger.info("[INTENT-GATE] question=%.120s",
                        verdict.get("question", ""))
            from actions.action_schema import parse_action
            ask_dict = {"action_type": "ask_user",
                        "question": verdict.get("question", ""),
                        "reason": ("experimental intent gate: missing "
                                   f"{verdict.get('missing_field', '?')}")}
            ask_action = parse_action(ask_dict)
            return self._handle_ask_user(step, observation, ask_action,
                                         ask_dict, annotated)
        except Exception as e:
            logger.warning("[INTENT-GATE] ERROR %s", e)
            logger.info("[INTENT-GATE] fallback=ALLOW")
            return None

    def _hybrid_check(self, step, task, observation, action):
        """EXPERIMENT-2 hook: deterministic-first hybrid gate."""
        try:
            verdict = self.hybrid_gate.decide(
                task, self.memory.all(), action,
                observation.get("dom_elements", []))
        except Exception as e:
            logger.warning("[HYBRID-GATE] ERROR %s", e)
            logger.info("[HYBRID-GATE] fallback=ALLOW")
            return None
        if verdict.get("decision") != "ASK":
            return None
        from actions.action_schema import parse_action
        ask_dict = {"action_type": "ask_user",
                    "question": verdict.get("question", ""),
                    "reason": ("experimental hybrid gate: missing "
                               f"{verdict.get('missing_field', '?')}")}
        try:
            ask_action = parse_action(ask_dict)
        except Exception as e:
            logger.warning("[HYBRID-GATE] ERROR bad question: %s", e)
            return None
        return self._handle_ask_user(step, observation, ask_action,
                                     ask_dict, None)

    def _log(self, step, observation, action_dict, verification, annotated,
             goal_check=None) -> None:
        if self.trajectory_logger:
            extra: Dict[str, Any] = {
                "annotated_screenshot": annotated,
                "url": observation.get("url", ""),
            }
            if goal_check is not None:
                extra["goal_check"] = goal_check
            self.trajectory_logger.log_step(
                step, observation["screenshot_bytes"], action_dict, verification,
                extra=extra,
            )

    def _download_count(self) -> int:
        fn = getattr(self.browser, "download_count", None)
        try:
            return int(fn()) if callable(fn) else 0
        except Exception:
            return 0

    def _maybe_download_done(self, step, task, next_obs, action_dict,
                             annotated):
        count = self._download_count()
        if count <= self._downloads_seen:
            return None
        new = list(getattr(self.browser, "downloads", []))[self._downloads_seen:]
        self._downloads_seen = count
        names = [d.get("filename", "?") for d in new]
        logger.info(f"Step {step}: file download detected: {names}")
        if "download" not in (task or "").lower():
            return None  # incidental download; keep going
        saved = []
        try:
            saved = self.browser.save_downloads()
        except Exception as e:
            logger.warning(f"saving downloads failed: {e}")
        verification = {"success": True,
                        "reason": f"File downloaded: {', '.join(names)}",
                        "downloads": saved or names}
        self.memory.add({**action_dict, "reason": "download completed"},
                        {"success": True}, next_obs.get("url", ""))
        self._log(step, next_obs, action_dict, verification, annotated)
        return self._finish(
            step, True,
            f"Goal achieved: downloaded {', '.join(names)}"
            + (f" -> {saved[0]}" if saved else ""),
            next_obs, verification=verification,
            termination="download_completed")

    def _finish(self, step, success, reason, observation, verification=None,
               termination: str = "unknown",
               goal_check: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        outcome = {"success": success, "reason": reason, "steps": step,
                   "actions_executed": self.actions_count,
                   "termination": termination, "history": self.memory.all()}
        # Priority 4 (opt-in path): record the goal-verification outcome in
        # the terminal log entry when a chunk terminal `done` was checked.
        if goal_check is not None:
            outcome["goal_check"] = goal_check
        if self.action_chunking:
            outcome["action_chunking"] = True
            outcome["chunk_metrics"] = self.chunker.metrics.to_dict()
            # Additive ACC observability snapshot (only on the opt-in path;
            # absent when adaptive chunking is disabled).
            if self.adaptive_chunking and self.acc is not None:
                outcome["adaptive_chunking"] = self.acc.acc_metrics()
        if self.trajectory_logger:
            if verification is None:
                verification = {"success": success, "reason": reason}
            try:
                _finish_extra = ({"goal_check": goal_check}
                                 if goal_check is not None else None)
                self.trajectory_logger.log_step(
                    step, observation["screenshot_bytes"], {"action": "finish"},
                    verification, extra=_finish_extra)
            except Exception:
                pass
            self.trajectory_logger.log_result(outcome)
        logger.info(f"Finished step {step}: success={success} ({reason})")
        return {"next_observation": observation, "finished": True, "outcome": outcome}

    def _execute_chunk(self, step: int, task: str, observation: Dict[str, Any],
                       raw_actions: List[Dict[str, Any]], start_url: str = "") -> Dict[str, Any]:
        """Experimental Action Chunking: execute a sequence of actions with interruption checks.

        When adaptive chunking is enabled (opt-in), chunk sizing, target
        validation, navigation/boundary endings, the expected-transition
        contract check, and the goal-verification of terminal `done` are
        governed by the AdaptiveChunkController. With adaptive chunking off
        (default), this method behaves exactly as before.
        """
        from actions.action_schema import parse_action
        from pydantic import ValidationError

        # ── Adaptive chunking (opt-in): assess state and constrain the
        # accepted sequence BEFORE execution. Legacy behavior below is
        # unchanged when the feature is disabled.
        acc = self.acc if self.adaptive_chunking else None
        acc_signals = None
        acc_decision = None
        if acc is not None and acc.enabled:
            acc_signals = acc.assess_state(observation,
                                           self._last_chunk_start_obs)
        if self._run_start_obs is None:
            self._run_start_obs = observation
        self._last_chunk_start_obs = observation
        acc_dom = (observation.get("dom_elements")
                   if isinstance(observation, dict) else None)

        chunk_id = self.chunker.next_chunk_id()
        raw_count = len(raw_actions)
        if acc is not None and acc.enabled:
            acc_decision = acc.choose_chunk_size(
                acc_signals, existing_limit=self.chunk_size,
                proposed_actions=raw_actions, dom_elements=acc_dom)
            raw_actions, acc_decision = acc.constrain_actions(
                acc_signals, raw_actions, acc_decision,
                dom_elements=acc_dom,
                boundary_checker=self.chunker.is_boundary_action)
            logger.info(
                "[ACC] chunk_id=%s proposed=%d allowed=%d planned=%s "
                "preferred=%d stop=%s signals=%s",
                chunk_id, raw_count, acc_decision["allowed_chunk_size"],
                acc_decision.get("planned_chunk_size"),
                acc_decision["preferred_chunk_size"],
                acc_decision.get("stop_reason", ""),
                acc_decision.get("signals"))

        proposed_count = len(raw_actions)
        self.chunker.metrics.chunks_proposed += 1
        self.chunker.metrics.actions_proposed += proposed_count
        self.chunker.log_chunk_start(chunk_id, proposed_count)

        if acc is not None and acc.enabled and not raw_actions:
            # Adaptive path only (the legacy path is intentionally left
            # unchanged): an empty proposal must never crash on the `idx`
            # reference below; end the turn so the existing planner /
            # replanning flow continues with a fresh observation.
            acc.record_outcome({"kind": "info", "reason": "empty_chunk"})
            self.chunker.metrics.record_interruption("empty_chunk")
            self.chunker.log_chunk_result(0, True, "empty_chunk")
            return {"next_observation": observation}

        executed_count = 0
        interrupted = False
        interruption_reason = ""
        curr_obs = observation
        last_result = None

        for idx, act_dict in enumerate(raw_actions):
            # Adaptive (opt-in) §4.3 stale-target protection: beyond the
            # first action, a target that can no longer be confirmed in the
            # CURRENT (post-action) DOM ends the chunk before it so the
            # next planning cycle re-observes and re-grounds. Only for
            # idx > 0: the first action was validated at chunk start
            # against this same observation. Runs BEFORE the SoM marker
            # resolution, so a stale marker id cannot silently resolve to
            # yesterday's coordinates.
            if acc is not None and acc.enabled and idx > 0:
                _dom_now = (curr_obs.get("dom_elements")
                            if isinstance(curr_obs, dict) else None)
                if acc._target_status(act_dict, _dom_now) == "unknown":
                    interrupted = True
                    interruption_reason = "stale_target"
                    acc.record_outcome({
                        "kind": "unknown",
                        "reason": ("stale_target: target cannot be "
                                   "confirmed in the current DOM; chunk "
                                   "ends for fresh re-grounding")})
                    break

            # Resolve SoM if applicable
            if hasattr(self.grounder, "to_action") and ("target_id" in act_dict or "target" in act_dict):
                try:
                    act_dict = self.grounder.to_action(act_dict)
                except Exception:
                    pass

            # Adaptive (opt-in): reuse the existing ActionChunker validator
            # before schema parsing (parity with the LH chunk loop).
            if acc is not None and acc.enabled:
                _valid, _verr = self.chunker.validate_action(act_dict)
                if not _valid:
                    interrupted = True
                    interruption_reason = f"invalid_action: {_verr}"
                    acc.record_outcome({"kind": "failure",
                                        "reason": f"invalid_action: {_verr}"})
                    break

            # Pre-execution validation
            try:
                action = parse_action(act_dict)
            except ValidationError as e:
                interrupted = True
                interruption_reason = f"malformed_action: {e}"
                if acc is not None and acc.enabled:
                    acc.record_outcome({"kind": "failure",
                                        "reason": f"malformed_action: {e}"})
                break

            # Boundary check
            is_boundary, boundary_name = self.chunker.is_boundary_action(act_dict)

            # HITL check
            if getattr(action, "is_ask_user", None) and action.is_ask_user():
                interrupted = True
                interruption_reason = "hitl_ask_user"
                if acc is not None and acc.enabled:
                    acc.record_outcome({"kind": "ask_user",
                                        "reason": "hitl_ask_user"})
                last_result = self._handle_ask_user(step, curr_obs, action, act_dict, None)
                break

            # Terminal check (with the adaptive goal-verification safeguard;
            # legacy branch below is unchanged when adaptive chunking is off)
            if action.is_terminal():
                ok = action.is_success()
                done_check = None
                if acc is not None and acc.enabled and ok:
                    done_check = self._adaptive_goal_check(
                        task, curr_obs, start_url)
                    if done_check.get("verdict") != GCH_VERIFIED_SUCCESS:
                        # Priority 1 (fail-closed): never finalize success
                        # on the model's word alone — a verified failure,
                        # unknown, unavailable, or errored goal check all
                        # reject the done and hand control to the existing
                        # recovery / replanning flow (bounded).
                        _verdict = str(done_check.get("verdict", "unknown"))
                        acc.record_outcome({
                            "kind": "failure",
                            "reason": ("done_not_verified: "
                                       f"{_verdict}: "
                                       + str(done_check.get("reason", "")))[
                                           :220],
                            "goal_check_verdict": _verdict})
                        self.memory.add(
                            {**act_dict, "reason": action.reason},
                            {"success": False,
                             "error": ("done rejected: "
                                       f"{_verdict} "
                                       f"({done_check.get('reason', '')})")},
                            curr_obs.get("url", ""))
                        if self.recovery.should_recover({"success": False}):
                            self.recovery.record_recovery(
                                "adaptive chunk: done rejected by goal "
                                f"verification ({_verdict})")
                            interrupted = True
                            interruption_reason = "done_not_verified"
                            break
                        # Bounded safeguard: no recovery budget left and
                        # the goal is NOT verified — the task fails honestly
                        # instead of claiming an unverified completion.
                        last_result = self._finish(
                            step, False,
                            "done rejected: goal not verified "
                            f"({_verdict}: "
                            f"{done_check.get('reason', '')}); "
                            "recovery exhausted",
                            curr_obs,
                            verification={"success": False,
                                          "reason": str(done_check.get(
                                              "reason", "goal not verified"))},
                            termination=("goal_verification_unavailable"
                                         if _verdict == GCH_UNAVAILABLE else
                                         "verification_exhausted"),
                            goal_check=done_check)
                        interrupted = True
                        interruption_reason = "done_not_verified"
                        break
                executed_count += 1
                self.chunker.metrics.actions_executed += 1
                interruption_reason = "terminal_action"
                if done_check is not None:
                    # Only an explicit verified success reaches here: the
                    # goal verifier confirmed completion.
                    acc.record_outcome({
                        "kind": "terminal", "reason": "done_verified",
                        "goal_check_verdict": str(done_check.get(
                            "verdict", ""))})
                    last_result = self._finish(
                        step, True,
                        f"Agent done: {action.reason or action.action_type}",
                        curr_obs,
                        verification={
                            "success": True,
                            "reason": "Agent done (goal verified)"},
                        termination="agent_done", goal_check=done_check)
                    break
                term = "agent_done" if ok else "agent_fail"
                last_result = self._finish(step, ok, f"Agent done: {action.reason or action.action_type}",
                                           curr_obs, termination=term)
                break

            # Execute
            pre_url = self.browser.get_url() if hasattr(self.browser, "get_url") else curr_obs.get("url", "")
            t_exec_start = time.time()
            self.actions_count += 1
            exec_res = self.executor.execute(action)
            self.chunker.metrics.time_executing += (time.time() - t_exec_start)

            if not exec_res.get("success", False):
                interrupted = True
                interruption_reason = f"execution_failed: {exec_res.get('error')}"
                if acc is not None and acc.enabled:
                    acc.record_outcome({
                        "kind": "failure",
                        "reason": f"execution_failed: {exec_res.get('error')}"})
                break

            try:
                next_obs = self.observer.observe(save_screenshot=False)
            except Exception as e:
                interrupted = True
                interruption_reason = f"reobserve_failed: {e}"
                last_result = self._finish(step, False, f"Re-observe failed: {e}", curr_obs, termination="browser_error")
                if acc is not None and acc.enabled:
                    acc.record_outcome({"kind": "failure",
                                        "reason": f"reobserve_failed: {e}"})
                break

            # Verify
            t_v_start = time.time()
            if self.verifier:
                verification = self.verifier.verify(curr_obs, next_obs, act_dict, task)
            else:
                verification = {"success": True, "reason": "exec status ok"}
            self.chunker.metrics.time_verifying += (time.time() - t_v_start)

            self.memory.add({**act_dict, "reason": action.reason},
                            {"success": verification.get("success", True),
                             "error": verification.get("reason"),
                             "verification": verification.get("success", True)},
                            next_obs.get("url", ""))

            if not verification.get("success", False):
                interrupted = True
                interruption_reason = f"verification_failed: {verification.get('reason')}"
                if acc is not None and acc.enabled:
                    acc.record_outcome({
                        "kind": "failure",
                        "reason": (f"verification_failed: "
                                   f"{verification.get('reason')}")})
                curr_obs = next_obs
                break

            # Adaptive (opt-in) Priority 2: enforce the planner's
            # expected_transition contract with the existing STV evidence
            # machinery. Failures stop the chunk; unknown results are
            # recorded but never treated as verified success; when no
            # contract is present, the StateVerifier verdict above remains
            # the only authority (behavior identical to before).
            if acc is not None and acc.enabled:
                _transition = (act_dict.get("expected_transition")
                               if isinstance(act_dict, dict) else None)
                contract = verify_expected_transition(
                    curr_obs, next_obs, _transition, action=act_dict)
                if contract["result"] == "failed":
                    interrupted = True
                    interruption_reason = (
                        f"expected_transition_failed: "
                        f"{contract.get('failed_item', '')} "
                        f"({contract.get('note', '')})")
                    acc.record_outcome({"kind": "failure",
                                        "reason": interruption_reason,
                                        "contract": "failed"})
                    curr_obs = next_obs
                    break
                if contract["result"] == "unknown":
                    acc.record_outcome({"kind": "unknown",
                                        "reason": contract.get("note", ""),
                                        "contract": "unknown"})
                elif self.verifier is None:
                    # No state verifier: the only evidence is the
                    # executor's own success status — that is never
                    # "appropriately verified" execution (§6.6), so the
                    # outcome is recorded as unknown (streak decays; §6.7)
                    # and no unverified success is claimed.
                    acc.record_outcome({
                        "kind": "unknown",
                        "reason": ("exec_status_only: no state verifier "
                                   "configured"),
                        "contract": contract["result"]})
                else:  # passed or no_contract, with real verification
                    acc.record_outcome({"kind": "success",
                                        "reason": contract["result"],
                                        "contract": contract["result"]})

            executed_count += 1
            self.chunker.metrics.actions_executed += 1

            # Goal check
            if self.goal_verifier is not None:
                try:
                    goal_check = self.goal_verifier.check(task, next_obs, start_url, prev_observation=curr_obs, verification=verification)
                    if acc is not None and acc.enabled:
                        classified = classify_goal_check(goal_check, is_intermediate=True)
                        if classified.get("verdict") == GCH_VERIFIED_SUCCESS:
                            interruption_reason = "goal_achieved"
                            last_result = self._finish(step, True, f"Goal achieved: {goal_check.get('reason')}", next_obs, verification=verification, termination="goal_achieved_terminated", goal_check=classified)
                            break
                    elif goal_check.get("achieved"):
                        interruption_reason = "goal_achieved"
                        last_result = self._finish(step, True, f"Goal achieved: {goal_check.get('reason')}", next_obs, verification=verification, termination="goal_achieved_terminated")
                        break
                except Exception:
                    pass

            curr_obs = next_obs

            # Unexpected navigation check
            post_url = self.browser.get_url() if hasattr(self.browser, "get_url") else next_obs.get("url", "")
            if post_url != pre_url and (idx < proposed_count - 1):
                interrupted = True
                interruption_reason = "navigation"
                if acc is not None and acc.enabled:
                    acc.record_outcome({"kind": "navigation",
                                        "reason": "navigation"})
                break

            # Boundary action check
            if is_boundary and (idx < proposed_count - 1):
                interrupted = True
                interruption_reason = boundary_name
                if acc is not None and acc.enabled:
                    acc.record_outcome({"kind": "boundary",
                                        "reason": boundary_name})
                break

        actions_consumed = (idx + 1) if interrupted else proposed_count
        if hasattr(self.grounder, "advance"):
            try:
                self.grounder.advance(actions_consumed)
            except Exception:
                pass

        self.chunker.metrics.chunks_executed += 1
        if interrupted:
            self.chunker.metrics.record_interruption(interruption_reason)
            self.chunker.metrics.replanning_calls += 1

        self.chunker.log_chunk_result(executed_count, interrupted, interruption_reason)

        if acc is not None and acc.enabled and acc_decision is not None:
            # §13 distinction: executed_chunk_size is recorded only now,
            # after execution actually happened.
            acc_decision["executed_chunk_size"] = executed_count
            acc.record_executed_size(executed_count)
            logger.info("[ACC] executed=%d planned=%s allowed=%d stop=%s",
                        executed_count, acc_decision.get("planned_chunk_size"),
                        acc_decision.get("allowed_chunk_size"),
                        acc_decision.get("stop_reason", ""))

        if last_result is not None:
            return last_result
        return {"next_observation": curr_obs}

    def _adaptive_goal_check(self, task: str, curr_obs: Dict[str, Any],
                             start_url: str) -> Dict[str, Any]:
        """Goal verification for a chunk-mode terminal `done` (Priority 1).

        Reuses the existing GoalVerifier with the baseline call convention
        (chunk-start observation as the previous state, so progress made
        during the chunk counts). The raw result is classified by the
        shared ``classify_goal_check`` adapter; every verdict other than
        ``verified_success`` fails closed. A missing verifier is
        ``unavailable`` — never implicitly "achieved": terminal completion
        must be independently verified (spec §2), so unverified done
        claims are rejected through the existing recovery path.
        """
        if self.goal_verifier is None:
            # No verifier configured: the safeguard has nothing to consult.
            # Never substitute the model's done word for actual goal
            # verification: the terminal done is rejected (fail-closed).
            logger.warning(
                "[ACC] goal verification unavailable: no goal verifier is "
                "configured; terminal done will be rejected (fail-closed)")
            return {"verdict": GCH_UNAVAILABLE, "achieved": None,
                    "done": False,
                    "reason": "no goal verifier configured (fail-closed)",
                    "signals": {"acc_safeguard_off": True}}
        try:
            result = self.goal_verifier.check(
                task, curr_obs, start_url,
                prev_observation=self._run_start_obs or self._last_chunk_start_obs or curr_obs,
                verification={"success": True})
        except Exception as e:
            logger.warning(f"[ACC] goal check crashed: {e}")
            return classify_goal_check(None, exception=e)
        classified = classify_goal_check(result)
        logger.info("[ACC] terminal goal verdict=%s reason=%s",
                    classified.get("verdict"),
                    str(classified.get("reason", ""))[:160])
        return classified
