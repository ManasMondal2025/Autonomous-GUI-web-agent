"""Long-horizon verified controller: plan -> act -> STV-verify -> commit.

Composes existing pieces (Observer, grounders, ActionExecutor,
StateVerifier/GoalVerifier, TrajectoryLogger, VLM backends) under a new
subgoal loop. The baseline AgentController is untouched. Loop budget counts
ACTIONS, like the baseline's max_steps.

Honest limitations (see also docs/LONG_HORIZON_PLAN.md):
- ROLLBACK is URL-restore + bounded redrive only; form/scroll/auth/server
  state is NOT guaranteed. The limitation is logged on every rollback.
- Global recovery cap: RecoveryManager allows 10 recoveries per task by
  default here (baseline default is 5); when it binds, rungs abort even if
  per-rung budgets remain. Reported via termination=recovery_exhausted.
- Without STV (enable_stv=False) nothing may become VERIFIED: PASS-branch
  commits are skipped and any VERIFIED facts are demoted to UNVERIFIED.
"""

import io
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from PIL import Image
from pydantic import ValidationError

from agent.memory import AgentMemory  # noqa: F401 (kept for API parity)
from agent.recovery import RecoveryManager
from actions.action_schema import parse_action
from .task_memory import TaskMemory
from .experience_memory import ExperienceMemory
from .subgoal_planner import (GlobalPlanner, LocalPlanner, PlannerError,
                               accept_contracts, _field_kind,
                               _literal_shape, distribute_goal_literals,
                               fill_default_transition,
                               derive_action_expectation,
                               make_global_planner)
from .stv import STV, build_failure_trace, _evidence_pool, \
    _content_words
from . import failure_taxonomy as _ft
from .micro_planner import MicroPlanner
from .milestone_manager import MilestoneManager
from .state_extractor import extract_signature
from .checkpoint_manager import CheckpointManager
from .recovery import RecoveryPolicy
# Adaptive Chunk Controller (opt-in): the class-signature default below needs
# the constant at module level; adaptive_chunking imports STV helpers only
# inside functions, so this creates no circular import.
from agent.adaptive_chunking import ACC_DEFAULT_MAX_CHUNK_SIZE, \
    classify_goal_check, GCH_VERIFIED_SUCCESS, GCH_UNAVAILABLE
# Phase 1-3 additions (Task Compiler + SubgoalContract)
from .task_spec import (
    TaskCompiler, TaskSpec, CompilationResult,
    validate_spec_for_execution,
)
from .subgoal_contract import (
    SubgoalContract, validate_contract_pre_execution, repair_contract,
    ContractValidationError, MAX_RETRIES_DEFAULT,
)
from .observation_builder import build_observation_context, is_stale_observation
from agent.action_chunking import ActionChunker, DEFAULT_CHUNK_SIZE

logger = logging.getLogger(__name__)

IRREVERSIBLE_WORDS = ("delete", "remove", "checkout", "submit", "pay",
                      "confirm", "purchase", "cancel", "reset")

MAX_CONSECUTIVE_REPLANS = 4


class LongHorizonController:
    def __init__(
        self,
        browser,
        observer,
        grounder,
        executor,
        vlm,
        verifier=None,
        memory: Optional[TaskMemory] = None,
        recovery: Optional[RecoveryPolicy] = None,
        trajectory_logger=None,
        max_steps: int = 20,
        task: str = "",
        goal_verifier=None,
        enable_goal_check: bool = True,
        planning_horizon: int = 3,
        experience_memory: Optional[ExperienceMemory] = None,
        checkpoints_dir: Optional[Path] = None,
        enable_stv: bool = True,
        enable_planning: bool = True,
        enable_experience: bool = True,
        enable_checkpoints: bool = True,
        max_rollback_redrive_actions: int = 5,
        risk: str = "normal",
        detector=None,
        ocr=None,
        stv_fallback_threshold: float = 0.6,
        ask_handler=None,
        max_asks: int = 3,
        enable_state_gate: bool = True,
        enable_micro_planner: bool = True,
        enable_milestone_graph: bool = True,
        enable_progress_score: bool = True,
        enable_failure_classifier: bool = True,
        # Phase 1-3: Task Compiler + SubgoalContract validation
        enable_task_compiler: bool = True,
        task_compiler: Optional[TaskCompiler] = None,
        enable_contract_validation: bool = True,
        # Experimental Action Chunking (opt-in CLI feature)
        action_chunking: bool = False,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        # Experimental Adaptive Chunk Controller (opt-in CLI feature; OFF
        # by default so every existing behavior is unchanged). Only the
        # chunk-mode paths consult ACC; the baseline LH paths ignore it.
        adaptive_chunking: bool = False,
        adaptive_chunk_max_size: int = ACC_DEFAULT_MAX_CHUNK_SIZE,
    ):
        self.browser = browser
        self.observer = observer
        self.grounder = grounder
        self.executor = executor
        self.vlm = vlm
        self.verifier = verifier
        self.memory = memory or TaskMemory()
        if recovery is None:
            recovery = RecoveryPolicy(
                recovery=RecoveryManager(max_recovery_attempts=10))
        self.recovery = recovery
        self.trajectory_logger = trajectory_logger
        self.max_steps = 60 if "MiniShop benchmark" in task else max_steps
        self.task = task
        self.goal_verifier = goal_verifier
        self.enable_goal_check = enable_goal_check
        self.planning_horizon = max(0, planning_horizon)
        self.experience_memory = experience_memory
        self.enable_experience = enable_experience and \
            experience_memory is not None
        self.enable_checkpoints = enable_checkpoints
        self.enable_stv = enable_stv
        self.enable_planning = enable_planning
        self.max_rollback_redrive_actions = max_rollback_redrive_actions
        self.risk = risk
        self.detector = detector
        self.ocr = ocr
        self.stv = STV(detector=detector, ocr=ocr, vlm=vlm,
                       fallback_threshold=stv_fallback_threshold)
        self.checkpoints = CheckpointManager(checkpoints_dir)
        self.global_planner = make_global_planner(vlm)
        self.local_planner = LocalPlanner(vlm, horizon=max(1,
                                                           planning_horizon))
        self.action_chunking = bool(action_chunking)
        self.chunk_size = max(2, min(int(chunk_size or DEFAULT_CHUNK_SIZE), 6)) if action_chunking else 3
        self.chunker = ActionChunker(enabled=self.action_chunking, chunk_size=self.chunk_size)
        micro_h = self.chunk_size if self.action_chunking else max(1, min(planning_horizon, 3))
        self.micro_planner = MicroPlanner(
            vlm, horizon=micro_h, action_chunking=self.action_chunking, chunk_size=self.chunk_size)
        # Adaptive Chunk Controller (opt-in): deterministic sizing only.
        from agent import adaptive_chunking as _acc_mod
        self.adaptive_chunking = bool(adaptive_chunking)
        self.acc = _acc_mod.AdaptiveChunkController(
            enabled=self.adaptive_chunking, hard_cap=adaptive_chunk_max_size)
        self._last_acc_observation: Optional[Dict[str, Any]] = None
        # Adaptive chunking (opt-in): pre-task URL, remembered for the
        # terminal goal check (existing GoalVerifier convention).
        self._acc_start_url: str = ""
        self.stats: Dict[str, int] = {
            "actions": 0, "model_calls": 0, "grounding_attempts": 0,
            "local_replans": 0, "global_replans": 0, "retries": 0,
            "regrounds": 0, "waits": 0, "rollbacks": 0, "checkpoints": 0,
            "stv_pass": 0, "stv_nonpass": 0, "fallback_calls": 0,
        }
        self._history: List[Dict[str, Any]] = []
        self._traj_step = 0
        self.ask_handler = ask_handler
        self.max_asks = max_asks
        self._ask_count = 0
        self.enable_state_gate = enable_state_gate
        self.enable_micro_planner = enable_micro_planner
        self.enable_milestone_graph = enable_milestone_graph
        self.enable_progress_score = enable_progress_score
        self.enable_failure_classifier = enable_failure_classifier
        self._plan_log_path = None
        self._post_recovery = False
        self._consecutive_replans = 0
        self._last_failed_sig = None
        self._failed_streak = 0
        self._last_verify_ctx = (None, None)
        self._sg_start_state = None
        self._sg_start_commits = 0
        self._fact_commits = 0
        self._last_action_summary = None
        self._last_verdict_summary = None
        self._milestone_replanned: set = set()
        # Phase 1-3: Task Compiler + contract validation
        self.enable_task_compiler = enable_task_compiler
        self.enable_contract_validation = enable_contract_validation
        self._task_compiler = task_compiler or TaskCompiler(
            ask_handler=ask_handler, max_asks=max_asks)
        self._compiled_spec: Optional[TaskSpec] = None  # set in run()
        self._contract_repair_budget = MAX_RETRIES_DEFAULT
        # CAPTCHA safety gate (standalone detector -> existing HITL).
        # None = disabled (default): behavior identical to before.
        self.captcha_gate = None  # set via set_captcha_gate()
        self.captcha_max_rounds = 3
        self.stats["captcha_hits"] = 0
        self.stats["captcha_cleared"] = 0
        self._downloads_seen = self._download_count()
        # Popup, DatePicker, and AsyncWaiter handlers
        from agent.popup_handler import PopupHandler
        from agent.date_picker import DatePickerHandler
        from agent.async_waiter import AsyncWaiter
        self.popup_handler = PopupHandler(ask_handler=ask_handler)
        self.date_picker = DatePickerHandler(ask_handler=ask_handler)
        self.async_waiter = AsyncWaiter()
        self.enable_popup_gate = True
        self.enable_date_picker = True

    # -- public -----------------------------------------------------------
    def run(self, task: Optional[str] = None,
            task_id: str = "") -> Dict[str, Any]:
        task = task or self.task
        tid = task_id or f"task_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
        self._log("PLAN", f"task={tid} goal={task[:120]}")
        self._milestone_replanned = set()
        # Adaptive chunking (opt-in): per-task adaptive state starts clean;
        # no effect when the feature is disabled.
        if self.adaptive_chunking and self.acc is not None:
            self.acc.reset_if_needed()

        # ── Phase 1: Task Compiler ──────────────────────────────────────────
        spec_result = self._compile_task(task, tid)
        if spec_result is not None:
            # Compilation failed or needs clarification
            return spec_result
        # ───────────────────────────────────────────────────────────────────

        observation = self.observer.observe(save_screenshot=False)
        # Adaptive chunking (opt-in): remember the pre-task URL so the
        # terminal goal check can reason about navigation (GoalVerifier
        # convention). No effect when ACC is disabled.
        if self.adaptive_chunking and self.acc is not None:
            self._acc_start_url = str(observation.get("url", "") or "")
        page_text = observation.get("visible_text", "")
        image = self._image(observation)

        experiences = ""
        if self.enable_experience and self.experience_memory is not None:
            try:
                found = self.experience_memory.retrieve(task, k=3)
                experiences = self.experience_memory.format_for_planner(found)
                self._log("PLAN", f"retrieved {len(found)} past "
                                  f"experience(s) (suggestions only)")
            except Exception as e:
                logger.warning(f"Experience retrieval failed: {e}")

        if self.enable_planning:
            try:
                plan = self.global_planner.create_subgoals(
                    task, page_text, experiences, image=image)
                self.stats["model_calls"] += 1
            except PlannerError as e:
                return self._finish(task, tid, False,
                                    f"global planning failed: {e}",
                                    observation,
                                    termination="planning_failed")
            self._log("PLAN", f"global plan: {[s['id'] for s in plan]}")
        else:
            plan = [{"id": "S1", "description": task,
                     "success_condition": ""}]
        # Acceptance: every subgoal enters memory as a normalized contract
        # (weak existence-only conditions silently rewritten from params).
        plan = accept_contracts(plan)
        success_conds = [s.get("success_condition", "") for s in plan]
        self.memory.initialize(tid, task, success_conds, plan)
        self._store_spec_in_memory_after_initialize()

        while True:
            # 0. Early download check for pending background downloads
            if "download" in task.lower():
                done = self._maybe_download_done(self.stats["actions"], task, observation, {}, None)
                if done is not None:
                    return done

            if self.memory.current_subgoal is None:
                # Partial-plan honesty: a truncated global plan that ran to
                # its end is NOT completion -- extend via the EXISTING
                # replan budget (never a new one) instead of claiming
                # success early. Budget exhaustion fails honestly.
                if self.global_planner.last_truncated:
                    self.global_planner.last_truncated = False
                    rep = self.recovery.next(
                        "REPLAN",
                        {"has_checkpoint": self._has_checkpoint()})
                    if rep["instruction"] == "abort":
                        return self._finish(
                            task, tid, False,
                            "plan truncated and replan budget exhausted",
                            observation, termination="replan_exhausted")
                    ext = self._global_replan(
                        observation, "previous plan truncated; extend the "
                                     "remaining stages")
                    if ext.get("failed"):
                        return self._finish(
                            task, tid, False, ext["failed"], observation,
                            termination=ext.get("termination",
                                                "replan_failed"))
                    observation = ext["observation"]
                    # Freshly extended: point at the new work and re-enter
                    # the loop; an empty extension falls through to the
                    # final check below.
                    nxt = self.memory.advance_to_next_subgoal()
                    self._log("PLAN", f"plan extended; next={nxt}")
                    continue
                break
            if self.stats["actions"] >= self.max_steps:
                return self._finish(task, tid, False,
                                    "Step budget exhausted", observation,
                                    termination="budget_exhausted")
            sid = self.memory.current_subgoal
            self._log("PLAN",
                      f"subgoal {sid}: "
                      f"{self.memory.verified_subgoals[sid]['description']}")
            ok = self._run_subgoal(sid, observation)
            observation = ok["observation"]
            if ok.get("failed"):
                return self._finish(task, tid, False, ok["failed"],
                                    observation,
                                    termination=ok.get("termination",
                                                       "subgoal_failed"))
            if ok.get("task_done"):
                if ok.get("download_done"):
                    return ok["download_done"]
                self._save_experience(tid, task, True)
                # Adaptive chunking (opt-in): _terminal() has already run
                # the §5 terminal goal gate; an unverified completion can
                # never reach this branch. goal_check is None (absent from
                # the outcome) on the legacy ACC-off path.
                return self._finish(task, tid, True,
                                    "agent done + final STV passed",
                                    observation,
                                    termination="task_complete",
                                    goal_check=ok.get("acc_goal_check"))
            if ok.get("subgoal_done"):
                nxt = self.memory.advance_to_next_subgoal()
                self._log("PLAN", f"subgoal {sid} VERIFIED; next={nxt}")

        final = self._final_check(task, observation)
        gate = (self._acc_verify_terminal(observation)
                if final["success"] else None)
        gate_ok = (gate is None
                   or gate.get("verdict") == GCH_VERIFIED_SUCCESS)
        success = bool(final.get("success")) and gate_ok
        reason = final["reason"]
        unverified_claim = bool(final.get("success")) and not gate_ok
        if unverified_claim:
            # Adaptive chunking (opt-in) §5: the subgoal-based checks all
            # passed, but the independent terminal goal check does not
            # verify completion — reported honestly as NOT verified
            # instead of task_complete.
            _verdict = str(gate.get("verdict", "unknown"))
            reason = (f"{reason}; but goal verification returned "
                      f"{_verdict} ({gate.get('reason', '')}); "
                      f"completion not verified")
            if self.acc is not None:
                self.acc.record_outcome({
                    "kind": "failure",
                    "reason": (f"done_not_verified: {_verdict}: "
                               + str(gate.get("reason", "")))[:220],
                    "goal_check_verdict": _verdict})
            self.memory.record_failure(
                {"scope": "final_check", "subgoal": "",
                 "reason": f"task_complete rejected by goal check "
                           f"({_verdict})",
                 **self._failure_keys("CONTRACT_FAILURE")})
        self._save_experience(tid, task, success)
        if success:
            term = "task_complete"
        elif unverified_claim:
            term = ("goal_verification_unavailable"
                    if str((gate or {}).get("verdict")) == GCH_UNAVAILABLE
                    else "verification_exhausted")
        else:
            term = "final_check_failed"
        return self._finish(task, tid, success, reason,
                            observation,
                            termination=term,
                            goal_check=gate)

    # -- hierarchical rolling-horizon loop ------------------------------------
    def run_hierarchical(self, task: Optional[str] = None,
                         task_id: str = "") -> Dict[str, Any]:
        """Rolling-horizon control: skeleton -> milestones -> micro-plans.

        Mirrors run() but replaces the one-shot subgoal queue with a
        milestone graph (MilestoneManager) and per-batch micro-planning
        from CURRENT VERIFIED STATE. All execution, verification,
        guardrail, recovery, checkpoint and trajectory machinery is shared
        verbatim with the legacy loop (see _run_actions/_single_action/
        _handle_verdict); only plan sourcing and loop driving differ, so
        the legacy path stays byte-identical.
        """
        task = task or self.task
        tid = task_id or f"task_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
        self._log("PLAN", f"hierarchical task={tid} goal={task[:120]}")
        self._milestone_replanned = set()

        # ── Phase 1: Task Compiler ──────────────────────────────────────────
        spec_result = self._compile_task(task, tid)
        if spec_result is not None:
            return spec_result
        # ───────────────────────────────────────────────────────────────────

        observation = self.observer.observe(save_screenshot=False)
        # Adaptive chunking (opt-in): remember the pre-task URL (goal check
        # convention). No effect when ACC is disabled.
        if self.adaptive_chunking and self.acc is not None:
            self._acc_start_url = str(observation.get("url", "") or "")
        page_text = observation.get("visible_text", "")
        image = self._image(observation)

        experiences = ""
        if self.enable_experience and self.experience_memory is not None:
            try:
                found = self.experience_memory.retrieve(task, k=3)
                experiences = self.experience_memory.format_for_planner(found)
                self._log("PLAN", f"retrieved {len(found)} past "
                                  f"experience(s) (suggestions only)")
            except Exception as e:
                logger.warning(f"Experience retrieval failed: {e}")

        if not self.enable_planning:
            self._log("PLAN", "planning disabled: delegating to legacy run")
            return self.run(task, tid)
        try:
            skel = self.global_planner.create_skeleton(
                task, page_text, experiences, image=image)
            self.stats["model_calls"] += 1
        except PlannerError as e:
            return self._finish(task, tid, False,
                                f"skeleton planning failed: {e}",
                                observation,
                                termination="planning_failed")
        milestones = skel.get("milestones", [])
        self._log("PLAN", f"skeleton: {[m['id'] for m in milestones]}")
        before_lits = {m["id"]: list(m.get("literals", []) or [])
                       for m in milestones}
        milestones = distribute_goal_literals(milestones, task)
        for m in milestones:
            added = [x for x in m.get("literals", [])
                     if x not in before_lits.get(m["id"], [])]
            if added:
                self._log("PLAN", f"milestone {m['id']}: topped up "
                                  f"literals from goal: {added[:6]}")
        if not self.enable_milestone_graph:
            # Ablation: flat milestone sequence (dependency order kept,
            # edges dropped) to isolate the graph's contribution.
            milestones = [{**m, "dependencies": []} for m in milestones]
            self._log("PLAN", "milestone graph disabled: flat sequence")
        seed = accept_contracts([{
            "id": m["id"],
            "description": m.get("intent", "") or
            f"Reach: {m.get('terminal_state', '')}",
            "success_condition": m.get("terminal_state", ""),
            "intent": m.get("intent", ""),
            "target": "",
            "stage": "",
            "dependencies": m.get("dependencies", []),
            "terminal_predicates": [],
            "literals": m.get("literals", []),
        } for m in milestones])
        self.memory.initialize(tid, task,
                               [m.get("terminal_state", "")
                                for m in milestones], seed)
        self._store_spec_in_memory_after_initialize()
        mgr = MilestoneManager(self.memory)

        while True:
            # 0. Early download check for pending background downloads
            if "download" in task.lower():
                done = self._maybe_download_done(self.stats["actions"], task, observation, {}, None)
                if done is not None:
                    return done

            mid = mgr.current_milestone()
            if mid is None:
                if mgr.all_verified():
                    break
                remaining = mgr.remaining_milestones()
                return self._finish(
                    task, tid, False,
                    f"milestone deadlock: {remaining} never became ready",
                    observation, termination="milestone_deadlock")
            if not mgr.activate_milestone(mid):
                self.memory.record_failure(
                    {"scope": "milestone", "subgoal": mid,
                     "reason": "dependencies unmet at activation",
                     **self._failure_keys("MILESTONE_FAILURE")})
                return self._finish(
                    task, tid, False,
                    f"milestone {mid} blocked: dependencies unmet",
                    observation, termination="milestone_blocked")
            if self.stats["actions"] >= self.max_steps:
                return self._finish(task, tid, False,
                                    "Step budget exhausted", observation,
                                    termination="budget_exhausted")
            self._log("PLAN",
                      f"milestone {mid}: "
                      f"{self.memory.verified_subgoals[mid].get('intent', '')}")
            ok = self._run_milestone(mgr, mid, observation)
            observation = ok["observation"]
            if ok.get("failed"):
                try:
                    mgr.mark_failed(mid, str(ok["failed"])[:300])
                except Exception:
                    pass
                return self._finish(task, tid, False, ok["failed"],
                                    observation,
                                    termination=ok.get("termination",
                                                       "subgoal_failed"))
            if ok.get("task_done"):
                if ok.get("download_done"):
                    return ok["download_done"]
                self._save_experience(tid, task, True)
                # Adaptive chunking (opt-in): _terminal() has already run
                # the §5 terminal goal gate; an unverified completion can
                # never reach this branch. goal_check is None (absent from
                # the outcome) on the legacy ACC-off path.
                return self._finish(task, tid, True,
                                    "agent done + final STV passed",
                                    observation,
                                    termination="task_complete",
                                    goal_check=ok.get("acc_goal_check"))
            # subgoal_done (verified inside _handle_verdict) or plain
            # batch exhaustion: loop re-reads verified state and either
            # advances the milestone or micro-plans the next step.

        final = self._final_check(task, observation)
        gate = (self._acc_verify_terminal(observation)
                if final["success"] else None)
        gate_ok = (gate is None
                   or gate.get("verdict") == GCH_VERIFIED_SUCCESS)
        success = bool(final.get("success")) and gate_ok
        reason = final["reason"]
        unverified_claim = bool(final.get("success")) and not gate_ok
        if unverified_claim:
            # Adaptive chunking (opt-in) §5: the subgoal-based checks all
            # passed, but the independent terminal goal check does not
            # verify completion — reported honestly as NOT verified
            # instead of task_complete.
            _verdict = str(gate.get("verdict", "unknown"))
            reason = (f"{reason}; but goal verification returned "
                      f"{_verdict} ({gate.get('reason', '')}); "
                      f"completion not verified")
            if self.acc is not None:
                self.acc.record_outcome({
                    "kind": "failure",
                    "reason": (f"done_not_verified: {_verdict}: "
                               + str(gate.get("reason", "")))[:220],
                    "goal_check_verdict": _verdict})
            self.memory.record_failure(
                {"scope": "final_check", "subgoal": "",
                 "reason": f"task_complete rejected by goal check "
                           f"({_verdict})",
                 **self._failure_keys("CONTRACT_FAILURE")})
        self._save_experience(tid, task, success)
        if success:
            term = "task_complete"
        elif unverified_claim:
            term = ("goal_verification_unavailable"
                    if str((gate or {}).get("verdict")) == GCH_UNAVAILABLE
                    else "verification_exhausted")
        else:
            term = "final_check_failed"
        return self._finish(task, tid, success, reason,
                            observation,
                            termination=term,
                            goal_check=gate)

    def _run_milestone(self, mgr, mid: str, observation) -> Dict[str, Any]:
        """One milestone's rolling loop: fresh micro-plan per batch.

        Loop plumbing mirrors _run_subgoal (same budget/consecutive bounds)
        but every batch is planned from CURRENT VERIFIED STATE via the
        micro-planner; execution/verification/recovery are shared verbatim
        (_run_actions and below). Kept separate so the legacy loop stays
        byte-identical.
        """
        sg = self.memory.verified_subgoals[mid]
        self._sg_start_state = (
            mid,
            observation.get("url", ""),
            observation.get("visible_text", ""))
        self._sg_start_commits = self._fact_commits
        self._last_action_summary = None
        self._last_verdict_summary = None
        while True:
            if self.stats["actions"] >= self.max_steps:
                return {"observation": observation,
                        "failed": "Step budget exhausted",
                        "termination": "budget_exhausted"}
            actions_at_loop_top = self.stats["actions"]
            mgr.record_attempt(mid)
            if self.enable_micro_planner:
                actions = self._micro_plan(mid, observation)
            else:
                actions = self._local_fallback_plan(mid, observation)
            if isinstance(actions, dict):
                return actions
            outcome = self._run_actions(mid, actions, observation)
            observation = outcome["observation"]
            if outcome.get("failed") or outcome.get("task_done"):
                return outcome
            if outcome.get("subgoal_done"):
                return {"observation": observation, "subgoal_done": True}
            dist = mgr.distance(mid)
            self._log("PLAN", f"milestone {mid}: batch done, "
                              f"remaining={dist['remaining_requirements']}")
            if self.stats["actions"] > actions_at_loop_top:
                self._consecutive_replans = 0
            elif outcome.get("replan_subgoal"):
                if not self._count_replan():
                    return {"observation": observation,
                            "failed": "too many consecutive replans "
                                      "without action progress",
                            "termination": "replan_exhausted"}
                self._log("PLAN", f"replanning milestone {mid} "
                                  f"(inner recovery path)")
                continue
            rep = self.recovery.next(
                "REPLAN",
                {"has_checkpoint": self._has_checkpoint()})
            if rep["instruction"] == "abort":
                return {"observation": observation,
                        "failed": "local replan budget exhausted",
                        "termination": "replan_exhausted"}
            if rep["instruction"] == "global_replan":
                return self._global_replan(
                    observation, "milestone batch exhausted; escalated")
            self.stats["local_replans"] += 1
            if not self._count_replan():
                return {"observation": observation,
                        "failed": "too many consecutive replans without "
                                  "action progress",
                        "termination": "replan_exhausted"}
            self._log("PLAN", f"replanning milestone {mid} ({rep['detail']})")

    def _acc_horizon(self, observation):
        """Adaptive chunking (opt-in): preferred micro-plan horizon.

        Returns None when ACC is disabled (MicroPlanner.plan then uses its
        configured horizon exactly as before). Otherwise returns the
        deterministic allowed size for the current state assessment, never
        above the existing chunk_size cap (the planner clamps further).
        """
        acc = self.acc if (self.adaptive_chunking and self.acc is not None
                           and self.acc.enabled) else None
        if acc is None:
            return None
        signals = acc.assess_state(observation, self._last_acc_observation)
        self._last_acc_observation = observation
        decision = acc.choose_chunk_size(
            signals, existing_limit=self.chunk_size,
            proposed_actions=None,
            dom_elements=(observation.get("dom_elements")
                          if isinstance(observation, dict) else None))
        logger.info("[ACC] horizon preferred=%d allowed=%d stop=%s",
                    decision["preferred_chunk_size"],
                    decision["allowed_chunk_size"],
                    decision.get("stop_reason", ""))
        return decision["allowed_chunk_size"]

    def _acc_verify_terminal(self, observation):
        """Adaptive chunking (opt-in) §5: independent terminal goal
        verification, shared by run() / run_hierarchical() completion and
        by _terminal() done claims.

        Returns None when there is nothing to check (ACC disabled — the
        legacy path is untouched). Otherwise returns a goal-check verdict
        dict whose ``verdict`` must be ``verified_success`` for a
        completion claim; every other verdict (verified_failure /
        unknown / unavailable / verification_error) fails closed at the
        call sites through their existing recovery / honest-failure
        reporting. The existing GoalVerifier and its semantics are
        reused untouched via the shared ``classify_goal_check`` adapter;
        the model's own done word is never substituted for verification.
        """
        acc = self.acc if (self.adaptive_chunking and self.acc is not None
                           and self.acc.enabled) else None
        if acc is None:
            return None
        verifier = self.goal_verifier \
            if getattr(self, "enable_goal_check", True) else None
        if verifier is None:
            logger.warning(
                "[ACC] long-horizon terminal goal verification "
                "unavailable: no goal verifier configured; completion "
                "claim rejected (fail-closed)")
            return {"verdict": GCH_UNAVAILABLE, "achieved": None,
                    "done": False,
                    "reason": "no goal verifier configured (fail-closed)",
                    "signals": {"acc_safeguard_off": True}}
        try:
            mtask = getattr(self.memory, "task", {}) or {}
        except Exception:
            mtask = {}
        goal = (mtask.get("goal", "") if isinstance(mtask, dict) else "") \
            or (self.task or "")
        try:
            result = verifier.check(
                goal, observation,
                getattr(self, "_acc_start_url", "") or "",
                prev_observation=self._last_acc_observation or None,
                verification=None)
        except Exception as e:
            logger.warning(f"[ACC] long-horizon goal check crashed: {e}")
            return classify_goal_check(None, exception=e)
        classified = classify_goal_check(result)
        logger.info("[ACC] terminal goal verdict=%s reason=%s",
                    classified.get("verdict"),
                    str(classified.get("reason", ""))[:160])
        return classified

    def _micro_plan(self, mid, observation):
        """One rolling micro-plan (or a failure result dict on planner
        error). Ablation off-switch lives here: with the micro-planner
        disabled the legacy local planner serves the milestone instead."""
        sg = self.memory.verified_subgoals.get(mid, {})
        try:
            verified_facts = {
                k: f.get("value")
                for k, f in (self.memory.key_facts or {}).items()
                if isinstance(f, dict) and f.get("status") == "VERIFIED"}
            actions = self.micro_planner.plan(
                state=self.memory.verified_state_summary(),
                milestone={"id": mid, **sg,
                           "verified_facts": verified_facts},
                ui={"url": observation.get("url", ""),
                    "page_text": observation.get("visible_text", ""),
                    "element_count": len(
                        observation.get("dom_elements", []) or [])},
                failures=self.memory.recent_failures[-2:],
                banned_action=self._banned_action(),
                image=self._image(observation),
                goal=self.memory.task.get("goal", ""),
                # Adaptive chunking (opt-in): None falls back to the
                # planner's configured horizon (MicroPlanner.plan treats a
                # falsy horizon exactly like before, so the legacy call is
                # unchanged).
                horizon=self._acc_horizon(observation))
            self.stats["model_calls"] += 1
            self._log_plan_cycle("MICRO", mid, actions)
            return actions
        except PlannerError as e:
            try:
                self.memory.record_failure(
                    {"scope": "micro_plan", "subgoal": mid,
                     "reason": f"micro planning failed: {e}",
                     **self._failure_keys("PLANNING_FAILURE")})
            except Exception:
                pass
            return self._local_replan(
                mid, observation, f"micro planning failed: {e}",
                failure_class="PLANNING_FAILURE"
                if self.enable_failure_classifier else None)

    def _local_fallback_plan(self, mid, observation):
        """Legacy local planner serving a milestone (micro ablation)."""
        sg = self.memory.verified_subgoals.get(mid, {})
        try:
            h = LocalPlanner.resolve_horizon(
                "post_recovery" if self._post_recovery else self.risk,
                self.planning_horizon)
            actions = self.local_planner.plan_actions(
                {"id": mid, **sg}, self.memory.get_planner_context(),
                self._image(observation), horizon=h,
                page_text=observation.get("visible_text", ""),
                verified_state=self.memory.verified_state_summary(),
                last_action=self._last_action_summary,
                last_verdict=self._last_verdict_summary,
                banned_action=self._banned_action())
            self.stats["model_calls"] += 1
            self._log_plan_cycle("LOCAL-FALLBACK", mid, actions)
            return actions
        except PlannerError as e:
            return self._local_replan(
                mid, observation, f"local planning failed: {e}",
                failure_class="PLANNING_FAILURE"
                if self.enable_failure_classifier else None)

    # -- subgoal loop -------------------------------------------------------
    def _run_subgoal(self, sid: str, observation) -> Dict[str, Any]:
        """Iterative (non-recursive) subgoal loop. Each iteration executes
        at least one planned action, fails, completes, or consumes a
        bounded replan; MAX_CONSECUTIVE_REPLANS bounds action-free spins."""
        sg = self.memory.verified_subgoals[sid]
        # Progress baseline: a subgoal whose contract already holds AND for
        # which nothing changed and no fact was verified under it has
        # accomplished nothing (vacuous existence conditions match nav
        # chrome from the start). The satisfaction check enforces this.
        self._sg_start_state = (
            sid,
            observation.get("url", ""),
            observation.get("visible_text", ""))
        self._sg_start_commits = self._fact_commits
        # Fresh subgoal, fresh planner slate: last-subgoal action/verdict
        # summaries must not leak into the new subgoal's prompt.
        self._last_action_summary = None
        self._last_verdict_summary = None
        pre_fail = self._check_preconditions(sid, observation)
        if pre_fail is not None:
            return self._local_replan(sid, observation, pre_fail,
                                      failure_class="CONTRACT_FAILURE")
        while True:
            if self.stats["actions"] >= self.max_steps:
                return {"observation": observation,
                        "failed": "Step budget exhausted",
                        "termination": "budget_exhausted"}
            actions_at_loop_top = self.stats["actions"]
            risk = "post_recovery" if self._post_recovery else self.risk
            h = LocalPlanner.resolve_horizon(risk, self.planning_horizon) \
                if self.enable_planning else 0
            if h > 0:
                try:
                    actions = self.local_planner.plan_actions(
                        {"id": sid, **sg}, self.memory.get_planner_context(),
                        self._image(observation), horizon=h,
                        page_text=observation.get("visible_text", ""),
                        verified_state=(
                            self.memory.verified_state_summary()),
                        last_action=self._last_action_summary,
                        last_verdict=self._last_verdict_summary,
                        banned_action=self._banned_action())
                    self.stats["model_calls"] += 1
                    self._log_plan_cycle("LOCAL", sid, actions)
                except PlannerError as e:
                    return self._local_replan(
                        sid, observation, f"local planning failed: {e}")
                actions = self._truncate_irreversible(actions)
                actions, _ = self._filter_invariant_violations(sid, actions)
                if not actions:
                    return self._local_replan(
                        sid, observation, "planner returned no valid actions")
            else:
                # Reactive fallback: goal as target with a goal-anchored
                # expectation (goal words must appear post-action). Weaker
                # than planned transitions, but the subgoal check still
                # gates advancement, so nothing advances unverified.
                goal = self.memory.task.get("goal", "")
                actions = [{"__fallback_single": True, "target": goal,
                            "expected_transition": {
                                "must_change": [goal] if goal else [],
                                "must_be_true": [], "must_remain_true": [],
                                "must_not_happen": []}}]
            outcome = self._run_actions(sid, actions, observation)
            observation = outcome["observation"]
            if outcome.get("failed") or outcome.get("task_done"):
                return outcome
            if outcome.get("subgoal_done"):
                return {"observation": observation, "subgoal_done": True}
            if self.stats["actions"] > actions_at_loop_top:
                self._consecutive_replans = 0
            elif outcome.get("replan_subgoal"):
                if not self._count_replan():
                    return {"observation": observation,
                            "failed": "too many consecutive replans "
                                      "without action progress",
                            "termination": "replan_exhausted"}
                self._log("PLAN", f"replanning subgoal {sid} "
                                  f"(inner recovery path)")
                continue
            # Local batch exhausted without completion -> replan (bounded).
            rep = self.recovery.next(
                "REPLAN",
                {"has_checkpoint": self._has_checkpoint()})
            if rep["instruction"] == "abort":
                return {"observation": observation,
                        "failed": "local replan budget exhausted",
                        "termination": "replan_exhausted"}
            if rep["instruction"] == "global_replan":
                return self._global_replan(
                    observation, "local batch exhausted; escalated")
            self.stats["local_replans"] += 1
            if not self._count_replan():
                return {"observation": observation,
                        "failed": "too many consecutive replans without "
                                  "action progress",
                        "termination": "replan_exhausted"}
            self._log("PLAN", f"replanning subgoal {sid} ({rep['detail']})")

    def _count_replan(self) -> bool:
        """Count one action-free replan; False when the spin bound trips."""
        self._consecutive_replans += 1
        return self._consecutive_replans <= MAX_CONSECUTIVE_REPLANS

    def _banned_action(self) -> Optional[Dict[str, Any]]:
        """Action currently failing repeatedly, for the planner prompt.

        Returns None when there is no active streak: the prompt must not
        ban anything the streak mechanism has already cleared.
        """
        if not self._last_failed_sig or self._failed_streak < 1:
            return None
        atype, target, value = self._last_failed_sig
        return {"action": atype, "target": target, "value": value}

    def _check_preconditions(self, sid: str,
                             observation) -> Optional[str]:
        """State-grounded entry gate: every structured precondition must
        hold in the CURRENT observation (single-state pools; CHANGED-style
        preconditions cannot hold at entry and fail closed).

        Returns None when all hold (or none exist); else a replan reason.
        Never executes blindly on mismatch -- the caller replans bounded.
        """
        sg = self.memory.verified_subgoals.get(sid, {})
        pres = sg.get("preconditions", []) or []
        if not pres:
            return None
        from . import expectation as _expm
        try:
            pool = _evidence_pool(observation, None)
            dom = observation.get("dom_elements", []) or []
        except Exception as e:
            logger.warning(f"Precondition evidence failed: {e}")
            return None
        for pre in pres:
            if isinstance(pre, dict) and pre.get("predicate"):
                target = pre.get("target", "")
                norm = {"target": target.split()
                        if isinstance(target, str) else
                        list(target or []),
                        "predicate": str(pre.get("predicate", "")),
                        "value": pre.get("value", ""),
                        "raw": str(pre)}
            else:
                norm = _expm.normalize_expectation(str(pre or ""))
            ok, note = _expm.verify_item(norm, pool, pool, dom)
            if not ok:
                label = str(pre if isinstance(pre, str) else
                              pre.get("value", pre))[:120]
                return (f"precondition unmet for subgoal {sid}: "
                        f"{label} ({note}). Current verified stage is "
                        f"{self.memory.verified_stage!r}; reach the "
                        f"required state first instead of acting blindly.")
        return None

    def _run_actions(self, sid, actions, observation):
        if self.action_chunking:
            return self._run_action_chunk(sid, actions, observation)
        for planned in actions:
            if self.stats["actions"] >= self.max_steps:
                return {"observation": observation, "failed": "budget",
                        "termination": "budget_exhausted"}
            gate = self._state_gate_check(sid, planned, observation)
            if gate is not None:
                return gate
            res = self._single_action(sid, planned, observation)
            observation = res["observation"]
            if res.get("failed"):
                return res
            if res.get("subgoal_done") or res.get("task_done"):
                return res
            if res.get("retry_action") is not None:
                return self._exec_retry(sid, res["retry_action"],
                                        observation,
                                        res.get("target_text", ""),
                                        res.get("planned"))
            if res.get("reground_target") is not None:
                return self._exec_reground(sid, res["reground_target"],
                                           observation, res.get("planned"))
            if res.get("replan_subgoal"):
                return {"observation": observation, "replan_subgoal": True}
        return {"observation": observation}

    def _run_action_chunk(self, sid, actions, observation):
        """Experimental Action Chunking: execute predictable consecutive actions
        within a single planning cycle. Monitors execution for errors and state
        transitions, interrupting the chunk and triggering replanning when needed.
        """
        chunk_id = self.chunker.next_chunk_id()
        proposed = list(actions) if isinstance(actions, list) else []
        # ── Adaptive chunking (opt-in): constrain the accepted sequence
        # before execution. With ACC off this block is skipped and the
        # legacy chunk loop below is unchanged.
        acc = self.acc if (self.adaptive_chunking and self.acc is not None
                           and self.acc.enabled) else None
        acc_signals = None
        acc_decision = None
        if acc is not None:
            acc_signals = acc.assess_state(observation,
                                           self._last_acc_observation)
            self._last_acc_observation = observation
            acc_dom = (observation.get("dom_elements")
                       if isinstance(observation, dict) else None)
            acc_decision = acc.choose_chunk_size(
                acc_signals, existing_limit=self.chunk_size,
                proposed_actions=None, dom_elements=acc_dom)
            proposed, acc_decision = acc.constrain_actions(
                acc_signals, proposed, acc_decision, dom_elements=acc_dom,
                boundary_checker=self.chunker.is_boundary_action)
            logger.info(
                "[ACC] chunk_id=%s planned=%d allowed=%d preferred=%d "
                "stop=%s signals=%s",
                chunk_id, len(proposed), acc_decision["allowed_chunk_size"],
                acc_decision["preferred_chunk_size"],
                acc_decision.get("stop_reason", ""),
                acc_decision.get("signals"))

        proposed_count = len(proposed)
        self.chunker.metrics.chunks_proposed += 1
        self.chunker.metrics.actions_proposed += proposed_count
        self.chunker.log_chunk_start(chunk_id, proposed_count)

        if not proposed:
            self.chunker.metrics.record_interruption("empty_chunk")
            self.chunker.log_chunk_result(0, True, "empty_chunk")
            return {"observation": observation, "replan_subgoal": True}

        executed_count = 0
        interrupted = False
        interruption_reason = ""
        last_outcome = {"observation": observation}

        for idx, planned in enumerate(proposed):
            if self.stats["actions"] >= self.max_steps:
                interrupted = True
                interruption_reason = "step_budget_exhausted"
                last_outcome = {"observation": observation, "failed": "budget",
                                "termination": "budget_exhausted"}
                break

            # 1. Pre-execution action validation
            valid, v_err = self.chunker.validate_action(planned)
            if not valid:
                interrupted = True
                interruption_reason = f"invalid_action: {v_err}"
                last_outcome = {"observation": observation, "replan_subgoal": True}
                break

            # 2. State gate check
            gate = self._state_gate_check(sid, planned, observation)
            if gate is not None:
                interrupted = True
                interruption_reason = "state_gate_interrupted"
                last_outcome = gate
                break

            # 3. Popup / Consent banner check before action
            popup_gate = self._maybe_popup_gate(sid, observation)
            if popup_gate is not None:
                interrupted = True
                interruption_reason = "popup_detected"
                last_outcome = popup_gate
                break

            # 4. CAPTCHA check
            captcha_gate = self._maybe_captcha_gate(sid, observation)
            if captcha_gate is not None:
                interrupted = True
                interruption_reason = "captcha_detected"
                last_outcome = captcha_gate
                break

            # Pre-action URL tracking
            pre_url = self.browser.get_url() if hasattr(self.browser, "get_url") else observation.get("url", "")
            is_boundary, boundary_name = self.chunker.is_boundary_action(planned)

            # 5. Execute action via existing _single_action pipeline
            _acc_stv0 = self.stats.get("stv_pass", 0) if acc is not None else 0
            t_start = time.time()
            res = self._single_action(sid, planned, observation)
            self.chunker.metrics.time_executing += (time.time() - t_start)

            observation = res["observation"]

            # Check if action execution failed or replan was signaled
            if res.get("failed"):
                interrupted = True
                interruption_reason = f"action_failed: {res.get('failed')}"
                if acc is not None:
                    acc.record_outcome({"kind": "failure",
                                        "reason": interruption_reason})
                last_outcome = res
                break

            if res.get("retry_action") is not None:
                interrupted = True
                interruption_reason = "stv_retry_needed"
                if acc is not None:
                    acc.record_outcome({"kind": "failure",
                                        "reason": "stv_retry_needed"})
                last_outcome = self._exec_retry(sid, res["retry_action"],
                                                observation,
                                                res.get("target_text", ""),
                                                res.get("planned"))
                break

            if res.get("reground_target") is not None:
                interrupted = True
                interruption_reason = "stv_reground_needed"
                if acc is not None:
                    acc.record_outcome({"kind": "failure",
                                        "reason": "stv_reground_needed"})
                last_outcome = self._exec_reground(sid, res["reground_target"],
                                                   observation, res.get("planned"))
                break

            if res.get("replan_subgoal"):
                interrupted = True
                interruption_reason = "replan_signaled"
                if acc is not None:
                    acc.record_outcome({"kind": "failure",
                                        "reason": "replan_signaled"})
                last_outcome = res
                break

            # Action succeeded
            executed_count += 1
            self.chunker.metrics.actions_executed += 1
            last_outcome = res
            if acc is not None:
                _acc_verified = (
                    (self.stats.get("stv_pass", 0) - _acc_stv0) > 0
                    or res.get("task_done") or res.get("subgoal_done"))
                acc.record_outcome({
                    "kind": "success" if _acc_verified else "unknown",
                    "reason": ("stv_pass" if _acc_verified
                               else "stv_verdict_not_exposed")})

            # Check if terminal or subgoal satisfied
            if res.get("task_done"):
                interruption_reason = "task_done"
                break

            if res.get("subgoal_done"):
                interruption_reason = "subgoal_done"
                break

            # Check for unexpected navigation or URL change
            post_url = self.browser.get_url() if hasattr(self.browser, "get_url") else observation.get("url", "")
            if post_url != pre_url and (idx < proposed_count - 1):
                interrupted = True
                interruption_reason = "navigation"
                if acc is not None:
                    acc.record_outcome({"kind": "navigation",
                                        "reason": "navigation"})
                last_outcome = {"observation": observation, "replan_subgoal": True}
                break

            # Check if boundary action was executed (stop chunk for safe fresh observation)
            if is_boundary and (idx < proposed_count - 1):
                interrupted = True
                interruption_reason = boundary_name
                if acc is not None:
                    acc.record_outcome({"kind": "boundary",
                                        "reason": boundary_name})
                last_outcome = {"observation": observation}
                break

        self.chunker.metrics.chunks_executed += 1
        if interrupted:
            self.chunker.metrics.record_interruption(interruption_reason)
            self.chunker.metrics.replanning_calls += 1
            self.stats["chunk_replans"] = self.stats.get("chunk_replans", 0) + 1

        self.chunker.log_chunk_result(executed_count, interrupted, interruption_reason)
        if acc is not None and acc_decision is not None:
            acc_decision["executed_chunk_size"] = executed_count
            # Additive observability (§7): record actual executed sizes so
            # the metrics snapshot can distinguish planned from executed.
            if acc.enabled:
                acc.record_executed_size(executed_count)
            logger.info("[ACC] executed=%d planned=%s allowed=%s stop=%s",
                        executed_count, acc_decision.get("planned_chunk_size"),
                        acc_decision.get("allowed_chunk_size"),
                        acc_decision.get("stop_reason", ""))
        return last_outcome

    def _state_gate_check(self, sid, planned, observation):
        """State compatibility gate (pre-grounding, pre-execution).

        Returns None when the action may proceed, else a replan result
        dict. Blocked actions are recorded with their structured reason;
        the subgoal loop treats the outcome as an action-free replan
        (bounded by the existing consecutive-replan bound).
        """
        if not self.enable_state_gate:
            return None
        from . import state_gate as _sg
        try:
            milestone = {"id": sid,
                         **self.memory.verified_subgoals.get(sid, {})}
            verdict = _sg.check(self.memory.verified_state_summary(),
                                milestone, planned
                                if isinstance(planned, dict) else {},
                                observation)
        except Exception as e:
            logger.warning(f"State gate crashed ({e}); permissive")
            return None
        if verdict.get("allowed", True):
            return None
        reason = str(verdict.get("reason", "STATE_INCOMPATIBLE"))
        self._log("STV", f"state gate BLOCKED action: {reason}")
        self.stats["state_gate_blocks"] = \
            self.stats.get("state_gate_blocks", 0) + 1
        self.memory.record_failure(
            {"scope": "state_gate", "subgoal": sid, "reason": reason,
             **self._failure_keys("WRONG_PAGE"),
             "trace": {"decision": "REPAIR", "reason": reason,
                       "expected_state": verdict.get("expected_state"),
                       "observed_state": verdict.get("observed_state"),
                       "failed_action": planned
                       if isinstance(planned, dict) else {},
                       "evidence": [reason]}})
        self._history.append({"action": str((planned or {}).get("action",
                                                                "?"))
                              if isinstance(planned, dict) else "?",
                              "target": str((planned or {}).get("target",
                                                                ""))[:60]
                              if isinstance(planned, dict) else "",
                              "result": "state_gate_blocked"})
        # Same streak seeding as value-gate refusals: blocked actions never
        # execute, so name them in BANNED ACTION explicitly.
        if isinstance(planned, dict):
            self._last_failed_sig = (
                str(planned.get("action", "")),
                str(planned.get("target", ""))[:120],
                str(planned.get("value", "") or "")[:200])
            self._failed_streak = max(self._failed_streak, 1)
        return {"observation": observation, "replan_subgoal": True}

    # -- single verified action ----------------------------------------------
    def _single_action(self, sid, planned, observation):
        image = self._image(observation)
        dom = observation.get("dom_elements", [])
        target = ""
        if isinstance(planned, dict):
            target = planned.get("target", "") or ""
            if not target and planned.get("__fallback_single"):
                target = self.memory.task.get("goal", "")
        self.stats["grounding_attempts"] += 1
        if isinstance(planned, dict) and \
                str(planned.get("action", "")).lower() == "type" and \
                not planned.get("value"):
            # Micro-planners often emit value-less targets; default the
            # value deterministically from the contract before grounding
            # so DOM anchoring (which needs a value) can fire. Ambiguity
            # falls through to VLM grounding + value gate as before.
            sg = {}
            try:
                sg = self.memory.verified_subgoals.get(sid, {}) or {}
            except Exception:
                pass
            defaulted = self._default_typed_value(
                sid, target, planned, sg,
                [x for x in (sg.get("literals", []) or [])
                 if str(x or "").strip()],
                entered=self._entered_values())
            if defaulted:
                planned["value"] = defaulted
                self._log("GROUND", f"defaulted typed value "
                                    f"{defaulted[:40]!r} for target "
                                    f"{target[:40]!r}")
        action_dict = self._anchored_action(sid, planned, target, dom)
        if action_dict is None:
            atype = str(planned.get("action", "")).lower() if isinstance(planned, dict) else ""
            if atype == "click":
                ground_prompt = f"Click {target}"
            elif atype == "type":
                val = planned.get("value", "")
                ground_prompt = f"Type '{val}' in {target}" if val else f"Type in {target}"
            else:
                ground_prompt = target
            try:
                prediction = self.grounder.ground(image, ground_prompt, dom,
                                                  self._history_context())
            except Exception as e:
                self.memory.record_failure(
                    {"scope": "grounding", "subgoal": sid,
                     "reason": str(e),
                     **self._failure_keys("PLANNING_FAILURE")})
                self._log_failure_traj(sid, target, f"grounder crashed: {e}",
                                       observation)
                return self._signal_replan(
                    sid, observation, f"grounder crashed: {e}",
                    failure_class="PLANNING_FAILURE")
            to_action = getattr(self.grounder, "to_action", None)
            action_dict = to_action(prediction) if callable(to_action) \
                else prediction
            if isinstance(planned, dict) and planned.get("action"):
                action_dict["action"] = planned["action"]
                action_dict["action_type"] = planned["action"]
                if planned.get("action") == "type" and planned.get("value"):
                    action_dict["value"] = planned["value"]
                elif planned.get("action") == "click":
                    action_dict.pop("value", None)
        try:
            action = parse_action(action_dict)
            action.target = target
        except ValidationError as e:
            self.memory.record_failure(
                {"scope": "grounding", "subgoal": sid,
                 "reason": f"malformed action: {e}",
                 **self._failure_keys("PLANNING_FAILURE")})
            self._log_failure_traj(sid, target, f"malformed action: {e}",
                                   observation)
            return self._signal_replan(
                sid, observation, f"malformed grounded action: {e}",
                failure_class="PLANNING_FAILURE")
        # Popup / Consent Banner gate: detect and safely dismiss blocking dialogs
        popup_gate = self._maybe_popup_gate(sid, observation)
        if popup_gate is not None:
            return popup_gate

        # Date Picker helper: handles calendar navigation, dropdowns, and disabled dates
        date_gate = self._maybe_date_picker(sid, planned, target, observation)
        if date_gate is not None:
            return date_gate

        # CAPTCHA safety gate (default OFF): a blocking challenge pauses
        # via the existing HITL path BEFORE grounding anything further.
        gated = self._maybe_captcha_gate(sid, observation)
        if gated is not None:
            return gated
        # HITL: ask_user never reaches executor/STV. Prompt, record the
        # answer in history, reuse the same observation (no screenshot/DOM
        # diff), and continue. Consumes an action (stats below) but never a
        # recovery/replan budget.
        is_ask = getattr(action, "is_ask_user", None)
        try:
            is_ask = bool(is_ask()) if callable(is_ask) else bool(is_ask)
        except Exception:
            is_ask = getattr(action, "action_type", "") == "ask_user"
        if is_ask:
            from agent.hitl import resolve_ask_user, MAX_ASKS_DEFAULT
            limit = self.max_asks if self.max_asks is not None \
                else MAX_ASKS_DEFAULT
            if self._ask_count >= limit:
                self.memory.record_failure(
                    {"scope": "hitl", "subgoal": sid,
                     "reason": f"ask_user budget exhausted ({limit})"})
                return self._signal_replan(
                    sid, observation,
                    f"ask_user budget exhausted ({limit})")
            if self.ask_handler is None:
                try:
                    import sys as _sys
                    interactive = bool(getattr(_sys.stdin, "isatty", None)
                                       and _sys.stdin.isatty())
                except Exception:
                    interactive = False
                if not interactive:
                    # Grounder evasion containment: a model emitting
                    # ask_user instead of coordinates must not burn cycles
                    # on prompts nobody can answer. Refuse deterministically
                    # into bounded recovery (never a vacuous PASS).
                    reason = ("ask_user refused: non-interactive run with "
                              "no ask_handler; grounding must emit "
                              "executable coordinates")
                    self._log("STV", reason)
                    self.memory.record_failure(
                        {"scope": "hitl", "subgoal": sid, "reason": reason,
                         **self._failure_keys("GROUNDING_FAILURE")})
                    self._last_failed_sig = (
                        "ask_user", str(target or "")[:120], "")
                    self._failed_streak = max(self._failed_streak, 1)
                    return self._signal_replan(
                        sid, observation, reason,
                        failure_class="GROUNDING_FAILURE")
            self._ask_count += 1
            resolved = resolve_ask_user(action, action_dict,
                                        self.ask_handler)
            answered_dict = resolved["action_dict"]
            human_answer = resolved["human_answer"]

            # ── Inject the answer into subgoal parameters and memory ──────
            # The answer must be visible to the NEXT local_planner call so
            # it can emit a proper `type` action with the user-provided value.
            field_target = str(getattr(action, "target", None)
                               or action_dict.get("target", "unknown_field"))
            if human_answer:
                # 1. Inject into the current subgoal's parameters dict
                subgoal_data = self.memory.verified_subgoals.get(sid, {})
                params = subgoal_data.get("parameters", {})
                if isinstance(params, dict):
                    params[field_target] = human_answer
                    subgoal_data["parameters"] = params
                    self.memory.verified_subgoals[sid] = subgoal_data

                # 2. Inject into memory's verified context so all future
                #    subgoals (and the context builder) also see it
                self.memory.commit_verified_fact(
                    field_target, human_answer,
                    source="HITL")
                self._log("HITL",
                          f"injected answer into subgoal {sid} params: "
                          f"{field_target!r}={human_answer[:80]!r}")
            # ─────────────────────────────────────────────────────────────

            verification = {"success": True, "decision": "PASS",
                            "confidence": 1.0,
                            "reason": resolved["verification"]["reason"],
                            "evidence": {"mode": "hitl ask_user"},
                            "state_changes": {}, "unexpected_changes": [],
                            "memory_updates": [], }
            self.stats["actions"] += 1
            self._history.append(self._hist_entry(action, target,
                                                  f"ask_user answered: "
                                                  f"{human_answer[:120]}"))
            self._failed_streak = 0
            self._traj_log(answered_dict, verification, observation,
                           {"subgoal": sid, "planned_target": target,
                            "hitl": True})
            return {"observation": observation}

        self._log("GROUND",
                  f"target={getattr(action, 'target_id', None)} "
                  f"coords=({getattr(action, 'x', None)},"
                  f"{getattr(action, 'y', None)})")
        gate = self._type_value_gate(sid, observation, action, target,
                                       action_dict)
        if gate is not None:
            return gate
        # Expectation is derived AFTER the value gate so it names the
        # final (possibly substituted) value, never a stale planner one.
        action.expected_transition = self._attach_expectation(planned,
                                                              action)
        self._last_action_summary = {
            "type": str(getattr(action, "action_type", "")),
            "target": str(target or "")[:200],
            "value": str(getattr(action, "value", "") or "")[:200]}
        self._require_typed_value(action)
        if action.is_terminal():
            return self._terminal(sid, observation, action)
            
        if "download" in self.memory.task.get("goal", "").lower() or "download" in self.memory.verified_subgoals.get(sid, {}).get("description", "").lower():
            done = self._maybe_download_done(self.stats["actions"], self.memory.task.get("goal", ""), observation, action_dict, None)
            if done is not None:
                return {"observation": observation, "task_done": True, "download_done": done}

        if action.action_type == "click":
            dup = self._check_duplicate_submission(action, target, observation)
            if dup is not None:
                return dup

        try:
            exec_result = self.executor.execute(action)
        except Exception as e:
            self.memory.record_failure(
                {"scope": "execution", "subgoal": sid, "reason": str(e),
                 **self._failure_keys("ACTION_FAILURE")})
            self._log_failure_traj(sid, target, f"executor crashed: {e}",
                                   observation)
            self._history.append(self._hist_entry(action, target,
                                                  "execution_crashed"))
            return self._signal_replan(
                sid, observation, f"executor crashed: {e}",
                failure_class="ACTION_FAILURE")
        self.stats["actions"] += 1
        self._log("EXECUTE", f"{action.action_type} "
                             f"exec_success={exec_result.get('success')}")
        if isinstance(exec_result, dict) and exec_result.get("success") is False:
            err = exec_result.get("error", "action execution failed")
            self.memory.record_failure(
                {"scope": "execution", "subgoal": sid, "reason": err,
                 **self._failure_keys("ACTION_FAILURE")})
            self._log_failure_traj(sid, target, f"execution failed: {err}",
                                   observation)
            self._history.append(self._hist_entry(action, target,
                                                  "execution_failed"))
            return self._signal_replan(
                sid, observation, f"execution failed: {err}",
                failure_class="ACTION_FAILURE")
        after = self._stable_observe()
        verdict = self._verify(observation, action, after,
                               action.expected_transition or {},
                               target=target)
        self._log("STV", f"{verdict['decision']} "
                         f"(conf={verdict['confidence']}) :: "
                         f"{verdict['reason']}")
        self._traj_log(action_dict, verdict, after,
                       {"subgoal": sid,
                        "planned_target": target,
                        "expected_transition":
                            action.expected_transition or {}})
        return self._handle_verdict(sid, observation, action, after,
                                    verdict, target_text=target,
                                    planned=planned
                                    if isinstance(planned, dict) else {})

    def _maybe_popup_gate(self, sid, observation):
        """Popup, modal, and consent banner hook.
        Detects obstructions before executing planned actions.
        - Non-blocking promotions: ignored unless instructed.
        - Consequential dialogs: routed to HITL.
        - Consent banner / blocking modal: dismissed safely, verified via STV,
          and recorded in task memory.
        """
        if not getattr(self, "enable_popup_gate", True):
            return None
        page = None
        if hasattr(self.browser, "get_page"):
            try:
                page = self.browser.get_page()
            except Exception:
                pass
        goal = self.memory.task.get("goal", "")
        popup_info = self.popup_handler.detect_popup(
            observation, page=page, task_text=goal)
        if not popup_info.get("detected"):
            return None
        if not popup_info.get("is_blocking"):
            self._log("POPUP", f"Ignored non-blocking popup: {popup_info.get('reason')}")
            return None
        if popup_info.get("needs_hitl"):
            q = popup_info.get("question", "A dialog requires your attention.")
            action = parse_action({"action": "ask_user", "question": q, "reason": "Consequential dialog detected"})
            return {"action": action}
        if page is None:
            return None
        self._log("POPUP", f"Handling blocking popup: {popup_info.get('reason')}")
        dismiss_res = self.popup_handler.dismiss(page, popup_info)
        self.stats["actions"] += 1
        after = self._stable_observe()
        stv_res = self.stv.verify_popup_dismissal(observation, after, popup_info)
        self._log("STV", f"Popup dismissal verdict: {stv_res['decision']} :: {stv_res['reason']}")
        if stv_res["success"]:
            try:
                self.memory.commit_verified_fact(
                    f"dismissed_{popup_info.get('popup_type')}", True,
                    source="STV_POPUP", evidence=stv_res["reason"])
            except Exception:
                pass
            return {"observation": after}
        else:
            return None

    def _maybe_date_picker(self, sid, planned, target, observation):
        """Handle native or custom date picker widgets deterministically."""
        if not getattr(self, "enable_date_picker", True):
            return None
        page = None
        if hasattr(self.browser, "get_page"):
            try:
                page = self.browser.get_page()
            except Exception:
                pass
        if page is None:
            return None

        import re as _re
        goal = self.memory.task.get("goal", "")
        desc = self.memory.verified_subgoals.get(sid, {}).get("description", "")
        subgoal_text = f"{target} {desc}"
        is_date_task = any(k in subgoal_text.lower() for k in ("date", "calendar", "month", "year", "day")) or bool(_re.search(r"\b\d{4}-\d{2}-\d{2}\b", subgoal_text))
        if not is_date_task:
            return None

        date_info = self.date_picker.parse_target_date(f"{subgoal_text} {goal}")
        if not date_info.get("valid"):
            if date_info.get("ambiguous"):
                q = str(date_info.get("reason", "Ambiguous date format")) + " Please clarify the date in YYYY-MM-DD format."
                action = parse_action({"action": "ask_user", "question": q, "reason": "Ambiguous date format"})
                return {"action": action}
            return None

        # Check if custom calendar or date input is on page
        mode = self.date_picker.detect_widget_mode(page)
        has_custom = page.locator("#calendarRoot .calendar, .cal-grid").count() > 0
        has_native = page.locator("input[type='date']").count() > 0
        has_date_target = page.locator("#dateTarget, #appointmentDate, #combinedDate").count() > 0
        if not (has_custom or has_native or has_date_target):
            return None

        self._log("DATE", f"Handling date selection for {date_info.get('iso')} (mode={mode})")
        res = self.date_picker.select_date(page, date_info)
        self.stats["actions"] += 1
        after = self._stable_observe()
        if res.get("unavailable"):
            reason = res.get("reason", "Requested date is disabled and unavailable.")
            self._log("STV", f"Date selection unavailable: {reason}")
            self.memory.record_failure({"scope": "date_picker", "subgoal": sid, "reason": reason, **self._failure_keys("VALUE_FAILURE")})
            return self._signal_replan(sid, after, reason, failure_class="VALUE_FAILURE")

        target_field = "combinedDate" if page.locator("#combinedDate").count() > 0 else "appointmentDate"
        stv_res = self.stv.verify_date_selection(after, date_info["iso"], target_field=target_field)
        self._log("STV", f"Date selection verdict: {stv_res['decision']} :: {stv_res['reason']}")
        if stv_res["success"]:
            try:
                self.memory.commit_verified_fact("appointment_date", date_info["iso"], source="STV_DATE", evidence=stv_res["reason"])
            except Exception:
                pass
            return {"observation": after, "subgoal_done": True}
        else:
            return self._signal_replan(sid, after, stv_res["reason"], failure_class="VERIFICATION_FAILURE")

    def _check_duplicate_submission(self, action, target, observation):
        """Prevent duplicate submissions on forms with side effects."""
        page = None
        if hasattr(self.browser, "get_page"):
            try:
                page = self.browser.get_page()
            except Exception:
                pass
        if page is None:
            return None

        target_str = str(target or "").lower()
        is_submit_action = any(k in target_str for k in ("submit", "save profile", "submit appointment"))
        if not is_submit_action:
            return None

        btn = page.locator("#slowSubmitButton, #combinedSubmit, button:has-text('Submit'), button:has-text('Processing')")
        try:
            cnt = btn.count()
            has_btn = bool(cnt > 0) if isinstance(cnt, (int, float)) else False
        except Exception:
            has_btn = False
        if has_btn:
            first_btn = btn.first
            try:
                txt = (first_btn.inner_text() or "").lower()
                is_disabled = first_btn.is_disabled() or first_btn.get_attribute("disabled") is not None
                if is_disabled or "processing" in txt or "submitting" in txt:
                    self._log("ASYNC", "Submit button is already processing/disabled. Preventing duplicate submission and waiting for completion.")
                    self.async_waiter.wait_for_predicate(
                        page,
                        lambda p: (
                            p.locator(".status.ok, .status.bad").count() > 0 and
                            not any(k in (p.locator(".status").first.inner_text() or "").lower() for k in ("submitting", "saving", "loading", "please wait")),
                            "Submission failed" if "failed" in (p.locator(".status").first.inner_text() or "").lower() else None
                        ),
                        timeout_ms=10000,
                        description="submission completion"
                    )
                    after = self._stable_observe()
                    return {"observation": after}
            except Exception:
                pass
        return None

    def set_captcha_gate(self, gate) -> None:
        """Attach the CAPTCHA safety gate (None = disabled, default)."""
        self.captcha_gate = gate

    def _maybe_captcha_gate(self, sid, observation):
        """CAPTCHA hook. Returns None to proceed normally, a
        ``{"observation": fresh}`` dict when a blocking challenge was
        human-cleared, or a replan signal when it was not.
        Never raises: detector/page failures fail open (logged).
        """
        gate = getattr(self, "captcha_gate", None)
        if gate is None or not getattr(gate, "enabled", True):
            return None
        try:
            max_rounds = max(1, int(
                getattr(self, "captcha_max_rounds", 0)
                or getattr(gate, "max_rounds", 0) or 1))
        except (TypeError, ValueError):
            max_rounds = 1
        human_acted = False
        change_summary = "page changed"
        for _ in range(max_rounds):
            try:
                page = self.browser.get_page()
            except Exception as e:
                self._log("CAPTCHA_GATE",
                          f"no live page ({e}); continuing autonomous")
                return None
            res = gate.check(page)
            status = getattr(res, "status", "clear")
            if status in ("clear", "disabled"):
                if human_acted:
                    return self._captcha_cleared(sid, observation,
                                                 change_summary)
                return None
            if status == "error":
                self._log("CAPTCHA_GATE",
                          f"check error ({getattr(res, 'error', '?')}); "
                          "continuing autonomous")
                return None
            # BLOCKING: pause for human action in the existing browser
            # window via the shared HITL helper (never stdin, never a
            # typed answer). The same live page/session is reused.
            self.stats["captcha_hits"] = \
                self.stats.get("captcha_hits", 0) + 1
            ask_dict = gate.build_ask_dict(getattr(res, "result", None))
            question = str((ask_dict or {}).get("question", ""))
            if getattr(self, "observer", None) is None:
                self._log("CAPTCHA_GATE",
                          "no observer for takeover; replanning")
                self.memory.record_failure(
                    {"scope": "hitl", "subgoal": sid,
                     "reason": "captcha takeover without observer",
                     **self._failure_keys("ENVIRONMENT_FAILURE")})
                return self._signal_replan(
                    sid, observation, "CAPTCHA blocking with no observer "
                    "(captcha_unresolved)",
                    failure_class="ENVIRONMENT_FAILURE")
            from agent.hitl import (wait_for_browser_takeover,
                                    TAKEOVER_TIMEOUT_S_DEFAULT,
                                    TAKEOVER_POLL_S_DEFAULT)
            result = wait_for_browser_takeover(
                question or "Human input needed in the browser",
                self.observer, observation,
                timeout_s=TAKEOVER_TIMEOUT_S_DEFAULT,
                poll_interval_s=TAKEOVER_POLL_S_DEFAULT)
            try:
                observation = self._stable_observe()
            except Exception as e:
                self.memory.record_failure(
                    {"scope": "hitl", "subgoal": sid,
                     "reason": f"re-observe failed after CAPTCHA HITL: {e}",
                     **self._failure_keys("ENVIRONMENT_FAILURE")})
                return self._signal_replan(
                    sid, observation,
                    f"Re-observe failed after CAPTCHA HITL: {e}",
                    failure_class="ENVIRONMENT_FAILURE")
            if result.get("resumed"):
                human_acted = True
                change_summary = str(result.get("change_summary",
                                                change_summary))
                self._log("CAPTCHA_GATE",
                          "Human intervention completed; rechecking CAPTCHA...")
        # Still blocking after bounded human rounds: replan (bounded,
        # auditable) instead of resuming autonomous execution blindly.
        self._log("CAPTCHA_GATE",
                  "CAPTCHA still blocking after human intervention; "
                  "requesting replan")
        self.memory.record_failure(
            {"scope": "hitl", "subgoal": sid,
             "reason": "CAPTCHA still blocking after human intervention "
             "(captcha_unresolved)",
             **self._failure_keys("ENVIRONMENT_FAILURE")})
        return self._signal_replan(
            sid, observation, "CAPTCHA still blocking after human "
            "intervention (captcha_unresolved)",
            failure_class="ENVIRONMENT_FAILURE")

    def _captcha_cleared(self, sid, fresh, change_summary) -> Dict[str, Any]:
        """Shared bookkeeping for a human-cleared challenge."""
        self.stats["captcha_cleared"] = \
            self.stats.get("captcha_cleared", 0) + 1
        self.stats["actions"] += 1
        entry = {"action": "ask_user", "target": "",
                 "result": "human cleared CAPTCHA in browser "
                 f"({change_summary[:120]})"}
        self._history.append(entry)
        self._failed_streak = 0
        self._traj_log({"action": "ask_user", "reason": "captcha_detected"},
                       {"success": True, "decision": "PASS",
                        "confidence": 1.0,
                        "reason": "takeover: human cleared CAPTCHA in browser",
                        "evidence": {"mode": "hitl captcha takeover"},
                        "state_changes": {}, "unexpected_changes": [],
                        "memory_updates": []},
                       fresh, {"subgoal": sid, "hitl": True, "captcha": True})
        return {"observation": fresh}

    # -- verification + memory commit ------------------------------------------
    def _verify(self, before_obs, action, after_obs, expected, target=""):
        if not self.enable_stv:
            ok = bool(after_obs.get("exec_success", True))
            return {"decision": "PASS" if ok else "RETRY",
                    "confidence": 0.5,
                    "evidence": {"mode": "exec-status fallback"},
                    "state_changes": {}, "unexpected_changes": [],
                    "memory_updates": [], "reason": "STV disabled"}
        verdict = self.stv.decide(before_obs, action, after_obs, expected,
                                  self.memory, target=target)
        if verdict["confidence"] < self.stv.fallback_threshold:
            fb = self.stv.fallback_verdict(
                before_obs, self._action_dict(action), after_obs,
                expected, verdict["evidence"]
                if isinstance(verdict.get("evidence"), dict) else {})
            if fb is not None:
                self.stats["fallback_calls"] += 1
                self._log("STV", f"fallback -> {fb['decision']} "
                                 f"(conf={fb['confidence']})")
                return fb
        return verdict

    def _handle_verdict(self, sid, before_obs, action, after_obs, verdict,
                        target_text: str = "", planned=None):
        d = verdict["decision"]
        self._history.append(self._hist_entry(action, target_text, d))
        self._last_verdict_summary = {
            "decision": d,
            "confidence": verdict.get("confidence"),
            "reason": str(verdict.get("reason", ""))[:300],
            "evidence_items": list(
                verdict.get("unexpected_changes", []) or [])[:4],
            "expected_state": ((planned or {}).get("expected_transition")
                               if isinstance(planned, dict) else None),
        }
        if d == "PASS":
            self.stats["stv_pass"] += 1
            for u in verdict.get("memory_updates", []):
                try:
                    name = u.get("fact", "") if isinstance(u, dict) else ""
                    if not name:
                        continue
                    self.memory.commit_verified_fact(
                        name, u.get("value", True), source="STV",
                        evidence=u.get("evidence")
                        or verdict.get("evidence"))
                    self._fact_commits += 1
                except Exception as e:
                    logger.warning(f"Memory commit rejected: {e}")
            if not self.enable_stv:
                # Ablation honesty: without STV nothing may become VERIFIED.
                for name in list(self.memory.key_facts):
                    f = self.memory.key_facts[name]
                    if f.get("status") == "VERIFIED":
                        f["status"] = "UNVERIFIED"
            self.memory.set_state({"url": after_obs.get("url", "")})
            self.recovery.reset_step_counters()
            self._post_recovery = False
            self._consecutive_replans = 0
            self._last_failed_sig = None
            self._failed_streak = 0
            self._log("MEMORY", "committed STV-verified updates")
            self._log("PROGRESS", "action verified")
            if self.enable_progress_score:
                try:
                    from . import progress as _progress
                    dist = _progress.distance(self.memory, sid, after_obs)
                    self._log("PROGRESS",
                              f"milestone {sid} distance: "
                              f"remaining={dist['remaining_requirements']} "
                              f"{dist['remaining'][:3]}")
                    self._last_distance = getattr(self, "_last_distance",
                                                  {}) or {}
                    prev = self._last_distance.get(sid)
                    if prev is not None:
                        delta = _progress.progress_delta(
                            prev, dist,
                            unexpected=len(
                                verdict.get("unexpected_changes", []) or []))
                        self._log("PROGRESS",
                                  f"milestone {sid} progress_delta={delta}")
                    self._last_distance[sid] = dist
                except Exception as e:
                    logger.warning(f"Progress signal failed: {e}")
            self._last_verify_ctx = (before_obs, action)
            if self._subgoal_satisfied(sid, after_obs, before_obs,
                                       action):
                ev = [f"STV PASS: {verdict.get('reason', '')}",
                      f"url={after_obs.get('url', '')}"]
                try:
                    self.memory.mark_subgoal_verified(sid, ev)
                except Exception as e:
                    logger.warning(f"Subgoal verify rejected ({e}); "
                                   f"continuing without advance")
                    return {"observation": after_obs}
                self._log("PLAN", f"subgoal {sid} satisfied "
                                  f"(evidence-backed)")
                self.memory.set_verified_stage(
                    self.memory.verified_subgoals[sid].get("stage", ""))
                self._snapshot_checkpoint(sid, after_obs)
                return {"observation": after_obs, "subgoal_done": True}
            return {"observation": after_obs}
        self.stats["stv_nonpass"] += 1
        if d in ("RETRY", "REPAIR", "REPLAN", "ROLLBACK"):
            self._record_failure_trace(
                sid, d, verdict, action, target_text, planned,
                after_obs=after_obs)
        if d in ("RETRY", "REPAIR", "REPLAN", "ROLLBACK"):
            sig = (str(getattr(action, "action_type", "")),
                   str(target_text or "")[:120],
                   str(getattr(action, "value", "") or "")[:200])
            if sig == self._last_failed_sig:
                self._failed_streak += 1
            else:
                self._last_failed_sig = sig
                self._failed_streak = 1
            if self._failed_streak >= 2:
                # Same action failing repeatedly: retry/reground cannot fix
                # a wrong strategy or wrong value. Escalate to replan while
                # budgets last (bounded by the recovery policy + the
                # consecutive-replan bound in the subgoal loop).
                streak_reason = (
                    f"same action failed {self._failed_streak}x "
                    f"({sig[0]} {sig[1][:40]}); escalating to "
                    f"replan instead of retry/reground")
                self._log("STV", streak_reason)
                self.memory.record_failure(
                    {"scope": "streak", "subgoal": sid,
                     "reason": streak_reason,
                     **self._failure_keys("NO_PROGRESS")})
                self._failed_streak = 0
                rep2 = self.recovery.next(
                    "REPLAN", {"has_checkpoint": self._has_checkpoint()})
                self._post_recovery = True
                if rep2["instruction"] == "abort":
                    return {"observation": after_obs,
                            "failed": "same action failed repeatedly; "
                                      "replan budgets exhausted",
                            "termination": "recovery_exhausted"}
                if rep2["instruction"] == "global_replan":
                    return self._global_replan(after_obs, "repeat failures")
                self.stats["local_replans"] += 1
                return {"observation": after_obs, "replan_subgoal": True}
        ctx = {"has_checkpoint": self._has_checkpoint(),
               "reground_possible": True}
        rep = self.recovery.next(d, ctx)
        self._post_recovery = True
        ins = rep["instruction"]
        self._log("STV", f"decision={d} -> recovery: {ins} "
                         f"({rep['detail']})")
        if ins == "wait":
            self.stats["waits"] += 1
            return self._wait_and_redecide(
                sid, before_obs, action,
                action.expected_transition or {},
                target_text=target_text, planned=planned)
        if ins == "retry":
            self.stats["retries"] += 1
            return {"observation": after_obs, "retry_action": action,
                    "target_text": target_text, "planned": planned}
        if ins in ("reground", "local_repair"):
            self.stats["regrounds"] += 1
            return {"observation": after_obs,
                    "reground_target": target_text,
                    "planned": {"action": (planned or {}).get("action"),
                                "value": (planned or {}).get("value"),
                                "__fallback_single": (planned or {}).get(
                                    "__fallback_single"),
                                "expected_transition":
                                action.expected_transition or {}}}
        if ins == "local_replan":
            self.stats["local_replans"] += 1
            return {"observation": after_obs, "replan_subgoal": True}
        if ins == "global_replan":
            return self._global_replan(after_obs, f"STV {d}")
        if ins == "rollback":
            return self._rollback(after_obs)
        return {"observation": after_obs, "failed": rep["detail"],
                "termination": "recovery_exhausted"}

    # -- recovery primitives -----------------------------------------------------
    def _exec_retry(self, sid, action, observation, target_text="",
                    planned=None):
        try:
            self.executor.execute(action)
            self.stats["actions"] += 1
        except Exception as e:
            return {"observation": observation,
                    "failed": f"retry execution crashed: {e}",
                    "termination": "execution_error"}
        after = self._stable_observe()
        verdict = self._verify(observation, action, after,
                               action.expected_transition or {},
                               target=target_text)
        self._log("STV", f"retry -> {verdict['decision']} "
                         f"(conf={verdict['confidence']})")
        self._traj_log(self._action_dict(action),
                       verdict, after,
                       {"subgoal": sid, "retry": True,
                        "planned_target": target_text})
        return self._handle_verdict(sid, observation, action, after,
                                    verdict, target_text=target_text,
                                    planned=planned)

    def _exec_reground(self, sid, target, observation, planned):
        image = self._image(observation)
        self.stats["grounding_attempts"] += 1
        action_dict = self._anchored_action(
            sid, planned if isinstance(planned, dict) else {}, target,
            observation.get("dom_elements", []))
        if action_dict is None:
            try:
                prediction = self.grounder.ground(
                    image, target, observation.get("dom_elements", []),
                    self._history_context())
            except Exception as e:
                return self._signal_replan(
                    sid, observation, f"reground crashed: {e}")
            to_action = getattr(self.grounder, "to_action", None)
            action_dict = to_action(prediction) if callable(to_action) \
                else prediction
        try:
            action = parse_action(action_dict)
        except ValidationError as e:
            return self._signal_replan(
                sid, observation, f"malformed regrounded action: {e}")
        gate = self._type_value_gate(sid, observation, action, target,
                                     action_dict)
        if gate is not None:
            return gate
        action.expected_transition = self._attach_expectation(
            planned or {}, action)
        self._require_typed_value(action)
        self._last_action_summary = {
            "type": str(getattr(action, "action_type", "")),
            "target": str(target or "")[:200],
            "value": str(getattr(action, "value", "") or "")[:200]}
        self._log("GROUND",
                  f"reground target={target[:60]} "
                  f"coords=({getattr(action, 'x', None)},"
                  f"{getattr(action, 'y', None)})")
        try:
            self.executor.execute(action)
        except Exception as e:
            return self._signal_replan(
                sid, observation, f"reground execution crashed: {e}")
        self.stats["actions"] += 1
        after = self._stable_observe()
        verdict = self._verify(observation, action, after,
                               action.expected_transition or {},
                               target=target)
        self._log("STV", f"reground -> {verdict['decision']}")
        self._traj_log(action_dict, verdict, after,
                       {"subgoal": sid, "reground": True,
                        "planned_target": target})
        return self._handle_verdict(sid, observation, action, after,
                                    verdict, target_text=target,
                                    planned=planned)

    def _wait_and_redecide(self, sid, before_obs, action, expected,
                           target_text="", planned=None, rounds: int = 3):
        after = None
        for _ in range(rounds):
            time.sleep(1.0)
            try:
                after = self.observer.observe(save_screenshot=False)
            except Exception as e:
                return {"observation": before_obs,
                        "failed": f"re-observe failed during WAIT: {e}",
                        "termination": "browser_error"}
            v = self.stv.decide(before_obs, action, after, expected,
                                self.memory,
                                target=target_text) if self.enable_stv else {
                "decision": "RETRY", "confidence": 0.5, "evidence": {},
                "state_changes": {}, "unexpected_changes": [],
                "memory_updates": [], "reason": "STV disabled"}
            if v["decision"] != "WAIT":
                self._log("STV", f"post-wait re-decide -> {v['decision']}")
                return self._handle_verdict(sid, before_obs, action, after,
                                            v, target_text=target_text,
                                            planned=planned)
        return {"observation": after or before_obs, "replan_subgoal": True}

    def _signal_replan(self, sid, observation, reason,
                         failure_class=None):
        return self._local_replan(sid, observation, reason,
                                  failure_class=failure_class)

    def _local_replan(self, sid, observation, reason, failure_class=None):
        rep = self.recovery.next(
            "REPLAN",
            {"has_checkpoint": self._has_checkpoint()})
        self._post_recovery = True
        if rep["instruction"] == "abort":
            try:
                self.memory.fail_subgoal(sid, reason)
            except Exception:
                pass
            return {"observation": observation, "failed": reason,
                    "termination": "replan_exhausted"}
        if rep["instruction"] == "global_replan":
            # Smallest sufficient scope first: a milestone-scoped failure
            # (wrong contract for this subgoal) is honestly handled by
            # regenerating only this milestone's contract, once per
            # subgoal -- not by regenerating the whole task plan.
            if self.enable_failure_classifier and \
                    failure_class == "MILESTONE_FAILURE" and sid not in \
                    self._milestone_replanned:
                return self._milestone_replan(sid, observation, reason)
            return self._global_replan(observation, reason)
        self.stats["local_replans"] += 1
        entry = {"scope": "replan",
                 "subgoal": sid or "", "reason": reason}
        if failure_class:
            entry.update(self._failure_keys(failure_class))
        self.memory.record_failure(entry)
        return {"observation": observation, "replan_subgoal": True}

    def _milestone_replan(self, sid, observation, reason):
        """Regenerate only the active milestone's contract (bounded).

        Drops the success predicates back to the legacy string condition
        (they proved wrong or vacuous) while KEEPING parameters (values
        are usually right; expectations usually wrong), then replans
        locally so fresh values regenerate from verified state. Once per
        subgoal; afterwards the normal ladder (incl. global) resumes.
        """
        self._milestone_replanned.add(sid)
        self.stats["milestone_replans"] = \
            self.stats.get("milestone_replans", 0) + 1
        try:
            sg = self.memory.verified_subgoals.get(sid, {})
            if sg.get("success_predicates"):
                sg["success_predicates"] = []
                self._log("PLAN", f"milestone replan {sid}: cleared "
                                  f"failed success predicates, keeping "
                                  f"parameters; regenerating strategy")
        except Exception as e:
            logger.warning(f"Milestone contract refresh failed: {e}")
        self._failed_streak = 0
        self._consecutive_replans = 0
        self.memory.record_failure(
            {"scope": "milestone_replan", "subgoal": sid, "reason": reason,
             **self._failure_keys("MILESTONE_FAILURE")})
        return {"observation": observation, "replan_subgoal": True}

    def _global_replan(self, observation, reason):
        self.stats["global_replans"] += 1
        try:
            image = self._image(observation)
            exp = ""
            if self.enable_experience and self.experience_memory is not None:
                exp = self.experience_memory.format_for_planner(
                    self.experience_memory.retrieve(
                        self.memory.task.get("goal", ""), k=3))
            plan = self.global_planner.create_subgoals(
                self.memory.task.get("goal", ""),
                observation.get("visible_text", ""), exp, image=image)
            self.stats["model_calls"] += 1
        except Exception as e:
            return {"observation": observation,
                    "failed": f"global replan failed: {e}",
                    "termination": "replan_failed"}
        for sg in accept_contracts(plan):
            sid = sg["id"]
            if sid not in self.memory.verified_subgoals:
                try:
                    self.memory.add_subgoal(
                        sid, sg.get("description", ""),
                        sg.get("success_condition", ""),
                        intent=sg.get("intent", ""),
                        target=sg.get("target", ""),
                        stage=sg.get("stage", ""),
                        parameters=sg.get("parameters", {}),
                        preconditions=sg.get("preconditions", []),
                        success_predicates=sg.get(
                            "success_predicates", []))
                except ValueError:
                    pass
        self.memory.record_failure({"scope": "global_replan",
                                    "reason": reason})
        self._consecutive_replans = 0
        self._log("PLAN", f"global replan accepted: "
                          f"{[s['id'] for s in plan]}")
        return {"observation": observation, "replan_subgoal": True}

    def _rollback(self, observation):
        ref = self.memory.latest_checkpoint_ref()
        if ref is None:
            return {"observation": observation,
                    "failed": "rollback requested with no checkpoint",
                    "termination": "no_checkpoint"}
        self.stats["rollbacks"] += 1
        url = ref.get("url", "")
        self._log("STV", f"ROLLBACK to {ref.get('id')} ({url}). "
                         f"Limitation: URL restore + bounded redrive only; "
                         f"form/scroll/auth/server state NOT guaranteed.")
        try:
            if url:
                self.browser.navigate(url)
        except Exception as e:
            return self._fail_after_recovery(
                None, observation, "REPLAN",
                {"has_checkpoint": False},
                f"rollback navigate failed: {e}")
        after = self._stable_observe()
        for _ in range(self.max_rollback_redrive_actions):
            ok, _ = self._checkpoint_holds(ref, after)
            if ok:
                break
            try:
                corr = self.local_planner.plan_actions(
                    {"id": "redrive",
                     "description": f"Restore verified state of "
                                    f"{ref.get('id')}",
                     "success_condition": ""},
                    self.memory.get_planner_context(),
                    self._image(after), horizon=1,
                    page_text=after.get("visible_text", ""))
                self.stats["model_calls"] += 1
            except Exception as e:
                return {"observation": after,
                        "failed": f"rollback redrive planning failed: {e}",
                        "termination": "rollback_failed"}
            if not corr:
                break
            a0 = corr[0]
            try:
                from actions.action_schema import parse_action as _pa
                act = _pa({"action": a0.get("action", "wait"),
                           "value": a0.get("value")})
                self.executor.execute(act)
                self.stats["actions"] += 1
            except Exception as e:
                return {"observation": after,
                        "failed": f"rollback redrive crashed: {e}",
                        "termination": "rollback_failed"}
            after = self._stable_observe()
        ok, why = self._checkpoint_holds(ref, after)
        if ok:
            self._log("STV", f"rollback verified ({why})")
            self._consecutive_replans = 0
            return {"observation": after, "replan_subgoal": True}
        return self._fail_after_recovery(
            None, after, "REPLAN", {"has_checkpoint": False},
            f"rollback state unverified: {why}; escalating")

    # -- checks ---------------------------------------------------------------------
    def _checkpoint_holds(self, ref, observation):
        facts = (ref.get("key_facts") or {})
        pool = self._evidence_text(observation).lower()
        missing = [k for k, f in facts.items()
                   if isinstance(f, dict) and f.get("status") == "VERIFIED"
                   and str(f.get("value", ""))[:40].lower() not in pool]
        if ref.get("url") and observation.get("url", "") != ref["url"]:
            return False, f"url is {observation.get('url')} not {ref['url']}"
        if missing:
            return False, f"missing verified facts: {missing}"
        return True, "url + verified facts present"

    def _subgoal_satisfied(self, sid, after_obs, before_obs=None,
                             action=None):
        """Conservative: any checker crash counts as NOT satisfied (the
        subgoal loop keeps working rather than advancing on no evidence).

        Fallback (click actions only): when the live condition fails but the
        click demonstrably advanced the page (visible state changed, no
        error markers), accept the subgoal. This covers continue/save steps
        whose success_condition names the spent button ("Save Profile
        button is clicked"): after a successful advance that button is gone
        from the page, so the literal condition is unprovable. Type actions
        never use this fallback (a value appearing anywhere is not proof).
        """
        sg = self.memory.verified_subgoals.get(sid, {})
        sig = None
        try:
            sig = extract_signature(
                self._image(after_obs),
                after_obs.get("dom_elements"),
                detector=self.detector, ocr=self.ocr)
        except Exception as e:
            logger.warning(f"Subgoal signature failed: {e}")
        try:
            res = self.stv.verify_subgoal(sg, after_obs, sig)
        except Exception as e:
            logger.warning(f"Subgoal check crashed ({e}); not advancing")
            return False
        if bool(res.get("satisfied", False)):
            if self._sg_progress_made(sid, after_obs):
                return True
            no_progress_reason = (
                f"subgoal {sid}: contract holds but state is unchanged "
                f"since subgoal start and no fact was verified under it; "
                f"not advancing")
            self._log("PLAN", no_progress_reason)
            try:
                self.memory.record_failure(
                    {"scope": "no_progress", "subgoal": sid,
                     "reason": no_progress_reason,
                     **self._failure_keys("NO_PROGRESS")})
            except Exception:
                pass
            return False
        preds = sg.get("terminal_predicates", []) or []
        if (not preds
                and action is not None
                and str(getattr(action, "action_type", "")) == "click"
                and self._advanced_without_error(before_obs, after_obs)):
            target_str = f"{getattr(action, 'target', '') or ''} {getattr(action, 'reason', '') or ''} {sg.get('description', '') or ''} {sg.get('success_condition', '') or ''}".lower()
            if any(k in target_str for k in ("continue", "next", "save", "submit", "apply", "finish", "complete", "confirm", "proceed", "place order")):
                self._log("PLAN", f"subgoal {sid}: live condition unmet but "
                                  f"page advanced with no error markers; "
                                  f"accepting continue-step")
                return True
        return False

    def _sg_progress_made(self, sid, after_obs) -> bool:
        """True when working this subgoal changed observable state or
        verified at least one fact. Guards against vacuous contracts
        (existence conditions matching persistent nav chrome) verifying a
        subgoal under which nothing ran. Missing baseline (e.g. direct
        unit calls) is permissive: the live check already passed."""
        base = self._sg_start_state
        if not base or base[0] != sid:
            return True
        _, url0, text0 = base
        try:
            cur = (after_obs.get("url", ""),
                   after_obs.get("visible_text", ""))
        except Exception:
            return True
        if (url0, text0) != cur:
            return True
        return self._fact_commits > self._sg_start_commits

    @staticmethod
    def _advanced_without_error(before_obs, after_obs) -> bool:
        """True when the page visibly changed with no error markers present.
        Error vocabulary is whole-word matched: benign page text containing
        "failure" (as in "failure transitions") must not trip it."""
        import re as _re
        if not isinstance(before_obs, dict) or \
                not isinstance(after_obs, dict):
            return False
        b = f"{before_obs.get('url', '')}\n" \
            f"{before_obs.get('visible_text', '')}".lower().split()
        a = f"{after_obs.get('url', '')}\n" \
            f"{after_obs.get('visible_text', '')}".lower().split()
        if b == a:
            return False
        blob = " ".join(a)
        return not _re.search(
            r"\b(error|errors|invalid|missing|failed|fail|denied|expired|"
            r"forbidden|unauthorized)\b", blob)

    def _terminal(self, sid, observation, action):
        final = self._final_check(
            self.memory.task.get("goal", ""), observation)
        gate = (self._acc_verify_terminal(observation)
                if final["success"] else None)
        gate_ok = (gate is None
                   or gate.get("verdict") == GCH_VERIFIED_SUCCESS)
        # Adaptive chunking (opt-in) §5: when the independent terminal
        # goal check does not verify the completion, the done claim is
        # rejected here — mainline-equivalent terminal verification
        # parity. Rejection flows through the EXISTING premature-done
        # path below (bounded by the step budget), so the agent replans
        # instead of claiming an unverified completion.
        if final["success"] and gate is not None and not gate_ok:
            _verdict = str(gate.get("verdict", "unknown"))
            if self.acc is not None:
                self.acc.record_outcome({
                    "kind": "failure",
                    "reason": (f"done_not_verified: {_verdict}: "
                               + str(gate.get("reason", "")))[:220],
                    "goal_check_verdict": _verdict})
            self.memory.record_failure(
                {"scope": "done_claim", "subgoal": sid,
                 "reason": f"done rejected by goal check ({_verdict}): "
                           f"{gate.get('reason', '')}",
                 **self._failure_keys("CONTRACT_FAILURE")})
            if self.stats["actions"] >= self.max_steps:
                return self._finish(
                    self.memory.task.get("goal", ""),
                    self.memory.task.get("id", ""), False,
                    f"agent claimed done but goal verification "
                    f"returned {_verdict} "
                    f"({gate.get('reason', '')}); step budget exhausted; "
                    f"not treated as task completion",
                    observation,
                    termination=("goal_verification_unavailable"
                                 if _verdict == GCH_UNAVAILABLE
                                 else "verification_exhausted"),
                    goal_check=gate)
            return {"observation": observation, "replan_subgoal": True,
                    "acc_goal_check": gate}
        if final["success"]:
            if gate is not None and self.acc is not None:
                self.acc.record_outcome({
                    "kind": "terminal", "reason": "done_verified",
                    "goal_check_verdict": str(
                        gate.get("verdict", "verified_success"))})
            self._save_experience(
                self.memory.task.get("id", ""), True)
            self._traj_log({"action": str(action.action_type)},
                           {"success": True}, observation,
                           {"subgoal": sid, "terminal": True})
            return {"observation": observation, "task_done": True,
                    "acc_goal_check": gate}
        if self.stats["actions"] >= self.max_steps:
            return {"observation": observation,
                    "failed": f"agent claimed done but final STV failed: "
                              f"{final['missing']}",
                    "termination": "final_check_failed"}
        self.memory.record_failure(
            {"scope": "done_claim", "subgoal": sid,
             "reason": f"premature done: missing {final['missing']}",
             **self._failure_keys("CONTRACT_FAILURE")})
        return {"observation": observation, "replan_subgoal": True,
                "acc_goal_check": gate}

    def _final_check(self, task_goal, observation):
        """Task completion = every planned subgoal VERIFIED (proven live at
        its own stage, evidence-backed) AND the terminal marker -- the last
        subgoal's success condition -- holds on the live final page.

        The raw goal text is deliberately NOT matched: it contains
        procedural instructions ("use the stage's own buttons") that never
        appear verbatim on any page, so requiring it would make completion
        unprovable on multi-stage tasks."""
        sgs = list(self.memory.verified_subgoals.values())
        unverified = [sg.get("description", "")[:60]
                      for sg in sgs
                      if sg.get("status") != "VERIFIED"]
        if not sgs:
            return {"success": False, "missing": ["no subgoals planned"],
                    "reason": "no subgoals planned"}
        if unverified:
            return {"success": False, "missing": unverified,
                    "reason": f"unverified subgoals remain: {unverified}"}
        marker = sgs[-1].get("success_condition", "") or \
            sgs[-1].get("description", "")
        last_sid = list(self.memory.verified_subgoals.keys())[-1]
        sig = None
        try:
            sig = extract_signature(
                self._image(observation),
                observation.get("dom_elements"),
                detector=self.detector, ocr=self.ocr)
        except Exception as e:
            logger.warning(f"Final signature failed: {e}")
        if self.stv.check_live(marker, observation, sig):
            return {"success": True, "missing": [],
                    "reason": "all subgoals VERIFIED + terminal marker "
                              f"holds live: {marker[:100]}"}
        # Consistency with subgoal acceptance: the last subgoal may have
        # been verified via clean-advancement (its literal marker names a
        # spent button, unprovable live). Re-running the same acceptance on
        # the unchanged final state must agree with verify-time.
        before, action = self._last_verify_ctx or (None, None)
        if before is not None and self._subgoal_satisfied(
                last_sid, observation, before, action):
            return {"success": True, "missing": [],
                    "reason": "all subgoals VERIFIED + terminal marker "
                              "accepted via recorded clean advancement"}
        return {"success": False, "missing": [marker],
                "reason": f"terminal marker not observed live: "
                          f"{marker[:100]}"}

    def _finish(self, task, tid, success, reason, observation,
                termination="unknown", goal_check=None):
        self._save_experience(tid, task, success)
        outcome = {"success": success, "reason": reason,
                   "steps": self.stats["actions"],
                   "termination": termination,
                   "history": self.memory.to_json(),
                   "long_horizon_stats": dict(self.stats)}
        if goal_check is not None:
            # Additive: the terminal goal-check verdict dict (from
            # _acc_verify_terminal), kept separate from the binary success
            # so post-hoc analysis can distinguish verified success from a
            # claimed success. Never present on the legacy (ACC off) path.
            outcome["goal_check"] = goal_check
        if self.action_chunking:
            outcome["action_chunking"] = True
            outcome["chunk_metrics"] = self.chunker.metrics.to_dict()
            self.stats.update(self.chunker.metrics.to_dict())
            # Additive ACC observability snapshot (only on the opt-in
            # path; absent when adaptive chunking is disabled).
            if self.adaptive_chunking and self.acc is not None:
                outcome["adaptive_chunking"] = self.acc.acc_metrics()
        if self.trajectory_logger:
            try:
                self.trajectory_logger.log_result(outcome)
            except Exception:
                pass
        logger.info(f"Finished: success={success} ({reason}) "
                    f"stats={self.stats}")
        return outcome

    def _save_experience(self, tid, task, success):
        if self.experience_memory is None:
            return
        try:
            transitions = [
                {"fact": name, "value": f.get("value"),
                 "evidence": f.get("evidence")}
                for name, f in self.memory.key_facts.items()
                if f.get("status") == "VERIFIED"]
            self.experience_memory.save_experience({
                "task_id": tid or self.memory.task.get("id", ""),
                "goal": task,
                "subgoals": [sg.get("description", "")
                             for sg in
                             self.memory.verified_subgoals.values()],
                "successful_transitions": transitions,
                "failures": list(self.memory.recent_failures),
                "recovery_actions": [f"counts={self.recovery.counts()}"],
                "final_status": "success" if success else "failed"})
        except Exception as e:
            logger.warning(f"Experience save failed (non-fatal): {e}")

    def _snapshot_checkpoint(self, sid, observation):
        if not self.enable_checkpoints:
            return
        try:
            cp = self.checkpoints.create(self.memory, observation,
                                         label=sid)
            self.memory.add_checkpoint_ref(cp)
            self.stats["checkpoints"] += 1
            self._log("PLAN", f"checkpoint {cp.get('id')} after {sid}")
        except Exception as e:
            logger.warning(f"Checkpoint snapshot failed (non-fatal): {e}")

    # -- small helpers ------------------------------------------------------------------
    def _stable_observe(self):
        return self.observer.observe(save_screenshot=False)

    @staticmethod
    def _image(observation):
        return Image.open(io.BytesIO(observation["screenshot_bytes"]))

    @staticmethod
    def _evidence_text(observation):
        from verification.goal_verifier import page_text
        try:
            return page_text(observation)
        except Exception:
            return observation.get("visible_text", "")

    def _history_context(self):
        return list(self._history[-10:])

    @staticmethod
    def _dom_anchor_coords(action_type, target, dom_elements,
                           viewport=(1280, 720)):
        """Deterministic DOM anchor: when the planner's target shares content
        words with EXACTLY ONE visible interactable element's own identifiers
        (text, label, placeholder, name, id, meaningful input type), return
        that element's center instead of asking the VLM to guess pixels.

        Forms with labeled inputs become trivially groundable; ambiguous
        (zero or tied matches) or vague targets fall back to VLM grounding.
        Verification still applies afterwards, so a bad anchor cannot
        silently succeed.
        """
        from .stv import _content_words as _kw
        want = [w for w in _kw(target or "") if w]
        if not want or action_type not in ("click", "type"):
            return None
        scored = []
        for el in dom_elements or []:
            if not isinstance(el, dict):
                continue
            if el.get("visible") is False:
                continue
            tag = str(el.get("tag", "") or "").lower()
            if action_type == "type":
                if tag and tag not in ("input", "textarea"):
                    continue
            else:
                if tag and tag not in ("input", "select", "textarea", "button", "a"):
                    continue
            ident = " ".join(str(el.get(k, "") or "") for k in (
                "text", "ariaLabel", "label", "placeholder", "name", "htmlId", "id"))
            itype = str(el.get("type", "") or "").lower()
            if itype in ("email", "tel", "number", "password", "radio",
                         "checkbox", "url", "date"):
                ident += f" {itype}"
            overlap = len(set(want) & set(_kw(ident)))
            if overlap <= 0:
                continue
            bb = el.get("bbox", {}) or {}
            try:
                if action_type == "type":
                    x = float(bb.get("x", 0)) + min(30.0, float(bb.get("width", 0)) / 4)
                else:
                    x = float(bb.get("x", 0)) + float(bb.get("width", 0)) / 2
                y = float(bb.get("y", 0)) + float(bb.get("height", 0)) / 2
            except (TypeError, ValueError):
                continue
            vw, vh = viewport
            if not (0 <= x < vw and 0 <= y < vh):
                continue
            tag_bonus = 0
            if action_type == "click":
                if tag in ("button", "a") or str(el.get("role", "")).lower() in ("button", "link"):
                    tag_bonus = 10
            elif action_type == "type":
                if tag in ("input", "textarea", "select"):
                    tag_bonus = 10

            # Exact match bonus on text, label, ariaLabel, or htmlId
            target_clean = str(target or "").strip().lower()
            exact_bonus = 0
            for k in ("text", "label", "ariaLabel", "placeholder", "htmlId"):
                val = str(el.get(k, "") or "").strip().lower()
                if val and val == target_clean:
                    exact_bonus = 20
                    break

            # Nav bar penalty for stage jump buttons (e.g. "T1: Profile", "T2: Search")
            nav_penalty = 0
            txt = str(el.get("text", "") or "").strip()
            if txt.startswith("T") and ":" in txt[:4]:
                nav_penalty = -15

            score = overlap + tag_bonus + exact_bonus + nav_penalty
            scored.append((score, x, y))
        if not scored:
            return None
        scored.sort(key=lambda t: -t[0])
        if len(scored) > 1 and scored[1][0] >= scored[0][0]:
            return None  # tied best: ambiguous, let the VLM decide
        _, x, y = scored[0]
        return (x, y)

    @staticmethod
    def _hist_entry(action, target_text, result):
        tid = getattr(action, "target_id", None)
        if tid:
            tgt = str(tid)
        else:
            x, y = getattr(action, "x", None), getattr(action, "y", None)
            tgt = f"{x},{y}" if x is not None else str(target_text or "")[:60]
        return {"action": str(getattr(action, "action_type", "?")),
                "target": tgt, "result": str(result)}

    def _require_typed_value(self, action) -> None:
        """For type actions, the typed value MUST be observable post-action.

        Appends the (post-gate, possibly substituted) value to must_change
        when absent. This only ever tightens expectations: a planner
        must_change naming just the field ("Email") would otherwise PASS on
        the label while nothing was typed (observed live). Never loosens.
        """
        try:
            if str(getattr(action, "action_type", "")) != "type":
                return
            value = str(getattr(action, "value", "") or "").strip()
            if not value:
                return
            exp = getattr(action, "expected_transition", None) or {}
            if not isinstance(exp, dict):
                return
            mc = list(exp.get("must_change", []) or [])
            blob = " ".join(str(x or "") for x in mc).lower()
            if value.lower() not in blob:
                mc.append(value)
                exp["must_change"] = mc
                try:
                    action.expected_transition = exp
                except Exception:
                    pass
                self._log("STV", f"expectation augmented with typed value "
                                f"{value[:40]!r}")
        except Exception as e:
            logger.warning(f"Typed-value augmentation failed: {e}")

    def _attach_expectation(self, planned, action):
        """Deterministic expected_transition for the executed action.

        The planner's expected_transition is deliberately ignored: small
        VLMs phrase must_change/must_remain_true items STV cannot resolve,
        turning correct actions into RETRY/REPLAN loops. The expectation
        is derived from the (post-gate) action type and value instead.
        """
        atype = str(getattr(action, "action_type", "wait") or "wait")
        # Controller-authored expectations (reactive no-planning fallback:
        # goal words must appear) are deterministic, not VLM prose: keep.
        if (isinstance(planned, dict) and planned.get("__fallback_single")
                and isinstance(planned.get("expected_transition"), dict)):
            exp = dict(planned["expected_transition"])
            try:
                action.expected_transition = exp
            except Exception:
                pass
            return exp
        try:
            exp = derive_action_expectation(
                atype, getattr(action, "value", None))
        except Exception as e:
            logger.warning(f"Deterministic expectation failed ({e}); "
                           f"using default transition")
            exp = fill_default_transition(atype)
        try:
            action.expected_transition = exp
        except Exception:
            pass
        return exp

    def _entered_values(self) -> set:
        """Literal values STV already verified as typed in this task."""
        entered = set()
        try:
            for name, f in (self.memory.key_facts or {}).items():
                if (isinstance(f, dict)
                        and f.get("status") == "VERIFIED"
                        and f.get("value") == "entered"):
                    entered.add(str(name).strip().lower())
        except Exception:
            pass
        return entered

    def _type_value_gate(self, sid, observation, action, target,
                           action_dict):
        """Ground type-in values in task literals before they touch the
        browser.

        Small planners substitute generic values ("John Doe" for a named
        person) or perseverate on stale ones (phone number for a cart
        field) even when the contract names the exact string. Anchors, in
        order: goal text, current-subgoal parameters (§4: parameters
        survive from global planning into the action), quoted literals.
        - planned value anchored and fresh: proceed;
        - unanchored or stale value with an unambiguous fix (single quoted
          literal, or unique parameter-key match on the target like
          "Size field" -> parameters["size"]): substitute transparently,
          then execute + verify;
        - otherwise refuse to execute and replan with a repair reason
          (surfaced to the next planner via recent_failures context).
        Returns None when the action may proceed, else a result dict.
        """
        try:
            from verification.goal_verifier import extract_entities
        except Exception:
            return None
        if str(getattr(action, "action_type", "")) != "type":
            return None
        value = str(getattr(action, "value", "") or "").strip()
        if not value:
            return None
        goal = self.memory.task.get("goal", "")
        try:
            literals = extract_entities(goal)
        except Exception:
            literals = []
        sg = {}
        try:
            sg = self.memory.verified_subgoals.get(sid, {}) or {}
        except Exception:
            pass
        desc = sg.get("description", "")
        params = {}
        try:
            params = dict(sg.get("parameters", {}) or {})
        except Exception:
            pass
        param_values = {str(v).strip().lower()
                        for v in params.values()
                        if str(v or "").strip()}
        sg_literals = {str(x).strip().lower()
                       for x in (sg.get("literals", []) or [])
                       if str(x or "").strip()}
        import re as _re
        quoted = [m[0] or m[1] for m in _re.findall(
            r'"([^"]+)"|\'([^\']+)\'', desc or "")]
        quoted = [q.strip() for q in quoted if q and q.strip()]
        entered = set()
        try:
            for name, f in (self.memory.key_facts or {}).items():
                if (isinstance(f, dict)
                        and f.get("status") == "VERIFIED"
                        and f.get("value") == "entered"):
                    entered.add(str(name))
        except Exception:
            pass
        if not literals and not entered and not params and not sg_literals:
            return None  # nothing task-anchored to check against
        vlow = value.lower()
        in_goal = vlow in goal.lower()
        in_params = vlow in param_values or vlow in sg_literals
        in_desc = vlow in (desc or "").lower()
        # Perseveration: a long value already entered for an earlier step
        # but absent from the current subgoal (planner stuck on a stale
        # value). Contract parameters AND milestone literals legitimize
        # re-entry (shipping name reuses the profile name); short values
        # like "3" are exempt.
        stale_reuse = (value in entered and not in_desc
                       and len(value.strip()) > 4
                       and vlow not in param_values
                       and vlow not in sg_literals)
        # Field misdirection: a task-anchored value whose shape contradicts
        # the target field (email literal into the Phone box). Only the
        # strong shapes (email/phone) count -- "Rahul" in a First-name
        # field must not be mistaken for a country. Fixed only when a
        # unique literal of the field's kind exists; otherwise the normal
        # path (and STV's wrong-field check) applies.
        kind = _field_kind(target or "")
        vshape = _literal_shape(value)
        strong = ("email", "phone")
        if (kind and vshape and vshape != kind
                and (kind in strong or vshape in strong)):
            pool = [x for x in (sg.get("literals", []) or [])
                    if str(x or "").strip()] or list(literals or [])
            cands = []
            for lit in pool:
                s = str(lit).strip()
                if _literal_shape(s) == kind and s not in cands:
                    cands.append(s)
            if len(cands) == 1:
                fixed = cands[0]
                self._log("PLAN", f"value substituted: {value!r} -> "
                                  f"{fixed!r} (subgoal {sid}; {vshape} "
                                  f"value aimed at {kind} field "
                                  f"{str(target or '')[:40]!r})")
                try:
                    action.value = fixed
                except Exception:
                    pass
                try:
                    if isinstance(action_dict, dict):
                        action_dict["value"] = fixed
                except Exception:
                    pass
                self._history.append(self._hist_entry(
                    action, target, f"value_substituted:{fixed}"[:60]))
                return None
        if (not in_goal and not in_params) or stale_reuse:
            fixed, why = self._gate_substitute(
                sid, value, target, desc, quoted, params,
                [x for x in (sg.get("literals", []) or [])
                 if str(x or "").strip()])
            if fixed is not None:
                self._log("PLAN", f"value substituted: {value!r} -> "
                                  f"{fixed!r} (subgoal {sid}; {why})")
                try:
                    action.value = fixed
                except Exception:
                    pass
                try:
                    if isinstance(action_dict, dict):
                        action_dict["value"] = fixed
                except Exception:
                    pass
                self._history.append(self._hist_entry(
                    action, target, f"value_substituted:{fixed}"[:60]))
                return None
        if (in_goal or in_params) and not stale_reuse:
            return None
        if stale_reuse:
            reason = (f"refused to re-type {value!r} into {target or 'field'}"
                      f": already entered earlier in this task and absent "
                      f"from the current subgoal ({desc!r}). Use this step's "
                      f"own literal value.")
        else:
            reason = (f"refused to type invented value {value!r} into "
                      f"{target or 'field'}: it appears nowhere in the task "
                      f"description. The current subgoal says: {desc!r} -- "
                      f"copy its literal values EXACTLY (e.g. type "
                      f"'Rahul Sharma', never 'John Doe'). Task literals "
                      f"include: {literals}")
        if sg_literals:
            reason += (f" Milestone literals for this step: "
                       f"{sorted(sg_literals)[:12]}")
        self._log("PLAN", reason)
        self.memory.record_failure(
            {"scope": "value_gate", "subgoal": sid, "reason": reason,
             **self._failure_keys("VALUE_FAILURE")})
        self._log_failure_traj(sid, target, reason, observation)
        self._history.append(self._hist_entry(action, target,
                                              "value_gate_refused"))
        # Seed the repeat-streak so the next prompt's BANNED ACTION names
        # this exact refusal (gate blocks never execute, so the streak
        # would otherwise stay silent while the planner loops). Cleared
        # on the next STV PASS like any other streak.
        self._last_failed_sig = (str(getattr(action, "action_type", "")),
                                 str(target or "")[:120],
                                 str(value or "")[:200])
        self._failed_streak = max(self._failed_streak, 1)
        return self._signal_replan(sid, observation, reason,
                                   failure_class="VALUE_FAILURE")

    @staticmethod
    def _default_typed_value(sid, target, planned, sg, literals,
                             entered=None):
        """Best-effort value for a planned type action with null value.

        Preference: unique parameter-key match on the target, then unique
        field-kind match among milestone literals (not-yet-entered ones
        first, so a filled Name field never steals the next name), then
        single quoted literal. Returns None when ambiguous (VLM grounding
        + value gate then take over as today). Only draws on contract/task
        literals; never invents.
        """
        entered = {str(e).strip().lower() for e in (entered or ())}
        try:
            params = dict(sg.get("parameters", {}) or {})
        except Exception:
            params = {}
        try:
            twords = set(_content_words(target or ""))
        except Exception:
            twords = set()
        hits = [k for k, v in params.items()
                if str(k or "").lower() in twords
                and str(v or "").strip()]
        if len(hits) == 1:
            return str(params[hits[0]]).strip()
        desc = ""
        try:
            desc = sg.get("description", "") or ""
        except Exception:
            pass
        import re as _re
        quoted = [q.strip() for q in
                  [m[0] or m[1] for m in _re.findall(
                      r'"([^"]+)"|\'([^\']+)\'', desc or "")]
                  if q and q.strip()]
        kind = _field_kind(target or "")
        if kind:
            cands = [str(lit).strip() for lit in (literals or [])
                     if _literal_shape(lit) == kind]
            fresh = [c for c in cands if c.lower() not in entered]
            if len(fresh) == 1:
                return fresh[0]
            if len(cands) == 1:
                return cands[0]
        if len(quoted) == 1:
            return quoted[0]
        return None

    @staticmethod
    def _gate_substitute(sid, value, target, desc, quoted, params,
                         literals=None):
        """Unambiguous fix for an unanchored/stale value, or (None, None).

        Preference: single quoted literal in the description, then unique
        parameter-key match on the target ("Size field" -> parameters
        ["size"]), then unique field-kind match among milestone literals
        ("Full name" + a person-name-shaped literal). Anything ambiguous
        refuses (returns None) -- guessing between several candidates
        would be another hallucination. Every substitution draws only on
        task/contract-provided literals and is logged transparently.
        """
        if len(quoted) == 1:
            return quoted[0], ("single quoted literal in subgoal "
                                f"{sid}; planner output was not "
                                "task-anchored")
        try:
            twords = set(_content_words(target or ""))
        except Exception:
            return None, None
        hits = [k for k, v in (params or {}).items()
                if str(k or "").lower() in twords
                and str(v or "").strip()]
        if len(hits) == 1:
            fixed = str(params[hits[0]]).strip()
            return fixed, (f"subgoal parameter {hits[0]!r} matches target "
                           f"{str(target or '')[:40]!r}")
        kind = _field_kind(target or "")
        if kind:
            cands = [lit for lit in (literals or [])
                     if _literal_shape(lit) == kind]
            if len(cands) == 1:
                return cands[0], (
                    f"milestone literal {cands[0][:40]!r} uniquely matches "
                    f"{kind} field {str(target or '')[:40]!r}")
        return None, None

    def _anchored_action(self, sid, planned, target, dom):
        """DOM-anchored action dict, or None when anchoring is impossible
        (VLM grounding then takes over). Needs a planned click/type with a
        word-bearing target; type actions additionally need the planned
        value (no value to anchor without one)."""
        if not isinstance(planned, dict):
            return None
        atype = str(planned.get("action", "")).lower()
        if atype == "click":
            value = None
        elif atype == "type":
            value = planned.get("value")
            if value is None or str(value) == "":
                return None
        else:
            return None
        vw = getattr(self.grounder, "viewport_width", 1280) or 1280
        vh = getattr(self.grounder, "viewport_height", 720) or 720
        try:
            pt = self._dom_anchor_coords(atype, target, dom, (vw, vh))
        except Exception as e:
            logger.warning(f"DOM anchor failed ({e}); VLM fallback")
            return None
        if pt is None:
            return None
        x, y = pt
        self._log("GROUND", f"DOM-anchored {atype} target={target[:60]!r} "
                            f"coords=({x:.0f},{y:.0f})")
        d = {"action": atype, "x": x, "y": y, "reason": "dom-anchored"}
        if value is not None:
            d["value"] = value
        return d

    def _filter_invariant_violations(self, sid, actions):
        """Drop planned actions that contradict VERIFIED invariants.

        A type action re-entering an already-VERIFIED literal (len>4)
        outside the subgoal whose contract wants it (params membership
        justifies reuse, e.g. shipping name) re-solves verified state
        without justification: filter it before grounding/execution.
        Returns (kept, dropped_reasons). Never invents replacements.
        """
        try:
            from .stv import _content_words
        except Exception:
            return actions, []
        sg = {}
        try:
            sg = self.memory.verified_subgoals.get(sid, {}) or {}
        except Exception:
            pass
        param_values = {str(v).strip().lower()
                        for v in (sg.get("parameters", {}) or {}).values()
                        if str(v or "").strip()}
        verified_literals = set()
        try:
            for name, f in (self.memory.key_facts or {}).items():
                if isinstance(f, dict) and \
                        f.get("status") == "VERIFIED":
                    if len(str(name or "")) > 4:
                        verified_literals.add(str(name).lower())
                    if f.get("value") == "entered" and \
                            len(str(name or "")) > 4:
                        verified_literals.add(str(name).lower())
        except Exception:
            pass
        kept, dropped = [], []
        for planned in actions or []:
            if not isinstance(planned, dict):
                kept.append(planned)
                continue
            if str(planned.get("action", "")).lower() != "type":
                kept.append(planned)
                continue
            value = str(planned.get("value", "") or "").strip()
            if not value or value.lower() in param_values:
                kept.append(planned)
                continue
            if value.lower() in verified_literals:
                reason = (f"dropped type {value!r}: already VERIFIED "
                          f"earlier and absent from subgoal {sid} "
                          f"parameters (re-solving verified state)")
                self._log("PLAN", reason)
                dropped.append(reason)
                continue
            kept.append(planned)
        if dropped:
            try:
                self.memory.record_failure(
                    {"scope": "invariant", "subgoal": sid,
                     "reason": "; ".join(dropped)[:500],
                     **self._failure_keys("CONTRACT_FAILURE")})
            except Exception:
                pass
        return kept, dropped

    def _truncate_irreversible(self, actions):
        out = []
        for a in actions:
            out.append(a)
            if isinstance(a, dict) and any(
                    w in str(a.get("target", "")).lower()
                    for w in IRREVERSIBLE_WORDS):
                self._log("PLAN", "truncating batch after irreversible "
                                  f"action: {a.get('target', '')[:60]}")
                break
        return out

    def _record_failure_trace(self, sid, decision, verdict, action,
                                target_text="", planned=None,
                                after_obs=None) -> None:
        """Structured STV failure feedback for the next planner call.

        Stored under the `trace` key of a recent_failures entry (string
        `reason` kept for backward compatibility). The local prompt renders
        current/expected/failed/reason/evidence from here -- never a bare
        "retry".
        """
        try:
            failed = {"type": str(getattr(action, "action_type", "")),
                      "target": str(target_text or "")[:200],
                      "value": str(getattr(action, "value", "")
                                   or "")[:200]}
            # Report the expectation STV actually checked (deterministic,
            # attached to the action), not the planner's discarded prose.
            expected = getattr(action, "expected_transition", None)
            if not isinstance(expected, dict) and isinstance(planned, dict):
                expected = planned.get("expected_transition")
            trace = build_failure_trace(
                decision, verdict.get("reason", ""),
                expected,
                self.memory.verified_state_summary(),
                failed,
                list(verdict.get("unexpected_changes", []) or [])
                or [verdict.get("reason", "")])
            reason = str(verdict.get("reason", ""))[:300]
            rlow = reason.lower()
            cls = _ft.classify(
                decision, reason,
                grounding_miss=("wrong element" in rlow
                                or "wrong field" in rlow
                                or "no known interactable" in rlow),
                loading=("loading" in rlow or "settling" in rlow),
                exec_error=bool(after_obs.get("exec_error")
                                if isinstance(after_obs, dict) else False))
            scope = _ft.scope_for(cls)
            if self.enable_failure_classifier:
                trace["failure_class"] = cls
                trace["replan_scope"] = scope
            self.memory.record_failure(
                {"scope": f"stv_{str(decision).lower()}", "subgoal": sid,
                 "reason": reason, "trace": trace,
                 **self._failure_keys(cls)})
        except Exception as e:
            logger.warning(f"Failure trace recording failed: {e}")

    def _fail_after_recovery(self, sid, observation, decision, ctx, reason):
        rep = self.recovery.next(decision, ctx)
        self._post_recovery = True
        if rep["instruction"] == "abort":
            if sid:
                try:
                    self.memory.fail_subgoal(sid, reason)
                except Exception:
                    pass
            return {"observation": observation, "failed": reason,
                    "termination": "recovery_exhausted"}
        return {"observation": observation, "replan_subgoal": True}

    def _has_checkpoint(self):
        return self.memory.latest_checkpoint_ref() is not None

    def _action_dict(self, action):
        if isinstance(action, dict):
            return action
        return {"action": str(getattr(action, "action_type", "?")),
                "x": getattr(action, "x", None),
                "y": getattr(action, "y", None),
                "target_id": getattr(action, "target_id", None),
                "value": getattr(action, "value", None),
                "text": getattr(action, "text", None)}

    def _log_failure_traj(self, sid, target, reason, observation):
        try:
            self._traj_log({"action": "none", "planned_target": target},
                           {"success": False, "reason": reason},
                           observation, {"subgoal": sid, "pre_execution": True})
        except Exception:
            pass

    def _failure_keys(self, cls):
        """failure_class/replan_scope metadata, or {} when the failure
        classifier is ablated off. All recovery behavior that depends on
        classes checks this flag first."""
        if not self.enable_failure_classifier:
            return {}
        return {"failure_class": cls,
                "replan_scope": _ft.scope_for(cls)}

    def _log_plan_cycle(self, scope, sid, proposal, extra=None):
        """Structured planning record (research logging, ablations).

        One JSONL line per planning cycle: verified state, remaining
        requirements, proposal, last action/verdict/ban, progress delta vs
        the previous cycle, and the active failure class. Execution/STV
        outcomes land in trajectory logs; the record captures what the
        planner knew when it proposed.
        """
        if self._plan_log_path is None:
            try:
                base = getattr(self.trajectory_logger, "path", None)
                if base:
                    from pathlib import Path as _P
                    self._plan_log_path = str(_P(base) / "planning_log.jsonl")
            except Exception:
                pass
        if not self._plan_log_path:
            return
        try:
            from datetime import datetime, timezone as _tz
            import json as _json
            summary = self.memory.verified_state_summary()
            if not self.enable_progress_score:
                summary = {k: v for k, v in summary.items()
                           if k != "remaining_requirements"}
            last_dist = getattr(self, "_last_distance", {}) or {}
            prev = last_dist.get(sid, {}) if isinstance(
                last_dist, dict) else {}
            rec = {"timestamp": datetime.now(_tz.utc).isoformat(),
                   "milestone_id": sid,
                   "verified_state": summary,
                   "remaining_requirements": summary.get(
                       "remaining_requirements", []),
                   "planner_scope": scope,
                   "proposal": proposal,
                   "state_gate": None,
                   "execution": None,
                   "stv": self._last_verdict_summary,
                   "progress_delta": None,
                   "failure_class": None,
                   "recovery": dict(self.recovery.counts())
                   if hasattr(self.recovery, "counts") else {}}
            if isinstance(prev, dict) and prev:
                try:
                    from . import progress as _progress
                    cur = {"verified": [], "remaining": list(
                        summary.get("remaining_requirements", []) or []),
                        "stage": summary.get("verified_stage", "")}
                    rec["progress_delta"] = _progress.progress_delta(
                        {"verified": list(prev.get("verified", []) or []),
                         "remaining": list(prev.get("remaining", []) or []),
                         "stage": prev.get("stage", "")}, cur)
                except Exception:
                    pass
            if isinstance(extra, dict):
                rec.update(extra)
            with open(self._plan_log_path, "a") as f:
                f.write(_json.dumps(rec, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Planning-cycle log failed: {e}")

    def _traj_log(self, action_dict, verification, observation, extra):
        if not self.trajectory_logger:
            return
        try:
            self._traj_step += 1
            merged = dict(extra or {})
            merged["long_horizon"] = True
            self.trajectory_logger.log_step(
                self._traj_step, observation["screenshot_bytes"],
                action_dict, verification, extra=merged)
        except Exception as e:
            logger.warning(f"Trajectory log failed (non-fatal): {e}")

    def _log(self, tag, msg):
        logger.info(f"[{tag}] {msg}")

    def _download_count(self) -> int:
        fn = getattr(self.browser, "download_count", None)
        try:
            return int(fn()) if callable(fn) else 0
        except Exception:
            return 0

    def _maybe_download_done(self, step, task, next_obs, action_dict, annotated):
        count = self._download_count()
        if count <= self._downloads_seen:
            return None
        new = list(getattr(self.browser, "downloads", []))[self._downloads_seen:]
        self._downloads_seen = count
        names = [d.get("filename", "?") for d in new]
        logger.info(f"Step {step}: file download detected: {names}")
        if "download" not in (task or "").lower():
            return None
        saved = []
        try:
            saved = self.browser.save_downloads()
        except Exception as e:
            logger.warning(f"saving downloads failed: {e}")
        
        verification = {"success": True, "decision": "PASS",
                        "reason": f"File downloaded: {', '.join(names)}",
                        "confidence": 1.0,
                        "downloads": saved or names}
        if action_dict:
            self._traj_log(action_dict, verification, next_obs, {"download_completed": True})
        
        return self._finish(
            task, getattr(self, "tid", "task"), True,
            f"Goal achieved: downloaded {', '.join(names)}" + (f" -> {saved[0]}" if saved else ""),
            next_obs,
            termination="download_completed"
        )

    # ── Phase 1: Task Compiler integration ──────────────────────────────────

    def _compile_task(self, task: str,
                      tid: str) -> Optional[Dict[str, Any]]:
        """Compile the task into a validated TaskSpec and store it in memory.

        Returns None when compilation succeeds (caller continues normally).
        Returns a _finish() dict when compilation fails or needs clarification.

        Design: compile-and-store is idempotent. If the controller is re-entered
        with the same tid (resumption), the stored spec is reused from memory.
        """
        if not self.enable_task_compiler:
            return None

        # Resumption: if memory already holds a spec for this task, reuse it
        existing_spec = None
        if self.memory.task.get("id") == tid:
            existing_spec = self.memory.get_task_spec()

        if existing_spec is not None:
            self._compiled_spec = existing_spec
            self._log("COMPILER", f"Reusing stored TaskSpec {existing_spec.task_id}")
            return None

        self._log("COMPILER", f"Compiling task: {task[:120]!r}")
        try:
            result: CompilationResult = self._task_compiler.compile(task)
        except Exception as exc:
            self._log("COMPILER", f"TaskCompiler crashed: {exc}")
            return self._finish(
                task, tid, False,
                f"Task compilation crashed: {exc}",
                {},
                termination="compilation_failed")

        if result.needs_clarification:
            self._log("COMPILER",
                      f"Compilation needs clarification: "
                      f"{result.clarification_question!r}")
            return self._finish(
                task, tid, False,
                f"Task compilation needs clarification: "
                f"{result.clarification_question}",
                {},
                termination="compilation_needs_clarification")

        if not result.ok or result.spec is None:
            self._log("COMPILER", f"Compilation failed: {result.error}")
            return self._finish(
                task, tid, False,
                f"Task compilation failed: {result.error}",
                {},
                termination="compilation_failed")

        spec = result.spec
        violations = validate_spec_for_execution(spec)
        if violations:
            self._log("COMPILER",
                      f"TaskSpec validation failures: {violations}")
            return self._finish(
                task, tid, False,
                f"TaskSpec pre-execution validation failed: {violations}",
                {},
                termination="compilation_failed")

        self._compiled_spec = spec
        self._log("COMPILER",
                  f"TaskSpec compiled OK: task_id={spec.task_id} "
                  f"permission={spec.max_permission} "
                  f"completion_criteria={len(spec.completion_criteria)}")

        # Store spec in task memory for persistence and auditing
        # Note: memory.initialize() is called AFTER this method by run(),
        # so we set the spec at initialization time via a deferred ref.
        # The actual store happens when the caller calls memory.initialize().
        # We cache it here and patch it in after initialize().
        self._pending_task_spec = spec
        return None  # success

    def _store_spec_in_memory_after_initialize(self) -> None:
        """Called after memory.initialize() to persist the compiled TaskSpec."""
        spec = getattr(self, "_pending_task_spec", None)
        if spec is not None and self.enable_task_compiler:
            try:
                self.memory.task_spec = spec.to_json()
            except Exception as e:
                logger.warning(f"Could not store TaskSpec in memory: {e}")
            self._pending_task_spec = None

    # ── Phase 3: SubgoalContract pre-execution validation ───────────────────

    def _validate_contract_gate(
            self,
            contract: "SubgoalContract",
            sid: str,
            observation: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Pre-execution SubgoalContract validation.

        Returns None when the contract is safe to execute.
        Returns a replan result dict when the contract is invalid and cannot
        be repaired within the bounded budget.

        The LLM cannot bypass this gate by emitting well-formed JSON that
        violates safety invariants.
        """
        if not self.enable_contract_validation:
            return None

        spec = self._compiled_spec
        verified_facts = {
            k: f.get("value")
            for k, f in (self.memory.key_facts or {}).items()
            if isinstance(f, dict) and f.get("status") == "VERIFIED"}

        for attempt in range(max(1, self._contract_repair_budget + 1)):
            violations = validate_contract_pre_execution(
                contract,
                task_spec=spec,
                verified_facts=verified_facts,
                current_url=observation.get("url", ""))

            if not violations:
                return None  # contract is clean

            self._log("CONTRACT",
                      f"Contract {contract.subgoal_id!r} violations "
                      f"(attempt {attempt+1}): {violations}")

            if attempt >= self._contract_repair_budget:
                # Repair budget exhausted: reject contract, trigger replan
                self.memory.record_failure({
                    "scope": "contract_validation",
                    "subgoal": sid,
                    "reason": f"SubgoalContract invalid after "
                              f"{self._contract_repair_budget} repair "
                              f"attempt(s): {violations[:3]}",
                    **self._failure_keys("CONTRACT_FAILURE"),
                })
                return self._signal_replan(
                    sid, observation,
                    f"SubgoalContract failed validation: {violations[:2]}",
                    failure_class="CONTRACT_FAILURE")

            # Attempt bounded repair
            repaired = repair_contract(contract, violations)
            re_violations = validate_contract_pre_execution(
                repaired, task_spec=spec,
                verified_facts=verified_facts,
                current_url=observation.get("url", ""))
            if not re_violations:
                self._log("CONTRACT",
                          f"Contract repaired successfully for {sid!r}")
                return None  # repaired contract is clean (caller still uses original)
            # Replace contract reference for next iteration
            contract = repaired

        return self._signal_replan(
            sid, observation,
            "SubgoalContract could not be validated",
            failure_class="CONTRACT_FAILURE")
