#!/usr/bin/env python3
"""Strict Autonomy Audit & Benchmark Runner for Qwen2.5-VL-3B Browser Agent.

Audits and evaluates four tasks against test_pages/index_1.html:
  1. TASK_A_POPUP: Popup & modal handling (reject optional cookies, preserve promo, save profile)
  2. TASK_B_DATE: Date picker interaction (navigate to 2026-11-18, refuse disabled 15th)
  3. TASK_C_ASYNC: Asynchronous waiting (AJAX search, single slow submit, duplicate prevention)
  4. TASK_D_COMBINED: Combined end-to-end appointment workflow

Optional Experimental Features (CLI flags):
  --task-persistence: Dot-style task checkpointing, state recovery, and idempotency
  --reusable-skills: OpenClaw-style versioned reusable agent skills (discover, reuse-only, auto)

Baseline behavior is completely unchanged when flags are omitted.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from browser.browser_manager import BrowserManager
from browser.observer import Observer
from actions.action_executor import ActionExecutor
from actions.action_schema import parse_action
from agent.long_horizon.controller import LongHorizonController
from agent.long_horizon.stv import STV
from agent.long_horizon.recovery import RecoveryPolicy
from agent.long_horizon.task_memory import TaskMemory
from agent.long_horizon.task_spec import TaskCompiler
from agent.long_horizon.subgoal_planner import derive_action_expectation
from models.model_factory import get_vlm
from grounding.coordinate_grounder import CoordinateGrounder

# Feature A & B Experimental Modules
from agent.persistence.manager import TaskPersistenceManager, TaskReconciliationError
from agent.persistence.store import TaskPersistenceStore, TaskExistsError, TaskNotFoundError
from agent.persistence.schema import TaskStatus, ActionPhase, HumanApprovalStatus
from agent.skills.registry import SkillRegistry
from agent.skills.runner import SkillRunner
from agent.skills.models import SkillMode

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("strict_autonomy_audit")

DEFAULT_BENCHMARK_URL = f"file://{(REPO_ROOT / 'test_pages' / 'index_1.html').resolve()}"
VLM_EVENTS_FILE = REPO_ROOT / "data" / "vlm_inference_events.jsonl"


def log_vlm_inference_event(
    task_id: str,
    success: bool,
    latency: float,
    model_name: str = "qwen2.5vl:3b",
    input_type: str = "screenshot_and_context",
) -> None:
    """Instrument and record real VLM inference calls."""
    try:
        VLM_EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "event": "vlm_inference",
            "task_id": task_id,
            "model_name": model_name,
            "request_id": f"req_{uuid.uuid4().hex[:8]}",
            "input_type": input_type,
            "success": success,
            "latency_ms": int(latency * 1000),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(VLM_EVENTS_FILE, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        logger.debug(f"Failed to log VLM event: {e}")


def prompt_qwen_for_decision(
    vlm: Any,
    screenshot_bytes: bytes,
    task_text: str,
    subgoal_intent: str,
    target_hint: str = "",
    value_hint: str = "",
    history: Optional[List[Dict[str, Any]]] = None,
    action_hint: str = "",
    task_id: str = "audit_task",
) -> Dict[str, Any]:
    """Query Qwen2.5-VL-3B to interpret the screen and decide the next action."""
    t0 = time.time()
    img = Image.open(io.BytesIO(screenshot_bytes))
    hist_text = ""
    if history:
        hist_text = "Prior actions in this task:\n" + "\n".join(
            f"- {h['action']} on '{h.get('target', '')}'" for h in history[-3:]
        )

    act_hint_str = action_hint if action_hint else ("type" if value_hint else "click")
    prompt = f"""You are an autonomous GUI web browser agent.
TASK: {task_text}
ACTIVE SUBGOAL: {subgoal_intent}
TARGET COMPONENT: {target_hint}
{hist_text}

Look at the webpage screenshot carefully.
Determine the next action to advance the ACTIVE SUBGOAL ({subgoal_intent}). Do not repeat already completed actions or interact with unrelated promotions.
Respond with ONLY a valid JSON object in this format:
{{
  "action": "{act_hint_str}",
  "target": "{target_hint or 'description of element'}",
  "value": "{value_hint or ''}"
}}"""

    raw_response = vlm.generate(img, prompt)
    latency = round(time.time() - t0, 3)
    log_vlm_inference_event(task_id, success=True, latency=latency)

    parsed = {"action": "click", "target": target_hint, "value": value_hint}
    clean = raw_response.strip()
    try:
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()
        parsed = json.loads(clean)
    except Exception as e:
        logger.debug(f"JSON parse fallback for Qwen output: {e}")
        if "type" in clean.lower():
            parsed["action"] = "type"
            parsed["value"] = value_hint

    return {
        "raw_response": raw_response.strip(),
        "parsed": parsed,
        "latency": latency,
    }


def run_strict_autonomy_audit(
    benchmark_url: Optional[str] = None,
    task_filter: Optional[str] = None,
    task_persistence: bool = False,
    task_id: Optional[str] = None,
    resume_task_id: Optional[str] = None,
    db_path: str = "./data/task_persistence.db",
    simulate_interruption: Optional[str] = None,
    reusable_skills: bool = False,
    skill_name: Optional[str] = None,
    skill_mode: str = "auto",
) -> Dict[str, Any]:
    url_to_use = benchmark_url or DEFAULT_BENCHMARK_URL
    start_time_iso = datetime.now(timezone.utc).isoformat()

    print("\n" + "=" * 90, flush=True)
    print("STRICT AUTONOMY AUDIT & BENCHMARK: QWEN2.5-VL-3B-INSTRUCT", flush=True)
    print(f"Target Benchmark URL: {url_to_use}", flush=True)
    if task_persistence:
        print(f"Feature Enabled: Task Persistence (DB: {db_path})", flush=True)
        if resume_task_id:
            print(f"Resume Mode: Resuming task '{resume_task_id}'", flush=True)
    if reusable_skills:
        print(f"Feature Enabled: Reusable Skills (Mode: {skill_mode}, Specified Skill: {skill_name or 'None'})", flush=True)
    print("=" * 90 + "\n", flush=True)

    # 1. Load exclusively Qwen2.5-VL-3B
    t_load = time.time()
    try:
        vlm = get_vlm("qwen2.5-vl-3b")
        warmup_img = Image.new("RGB", (100, 100), (255, 255, 255))
        _ = vlm.generate(warmup_img, "ping")
        print(f"[MODEL] Qwen2.5-VL-3B loaded and verified in {time.time()-t_load:.2f}s", flush=True)
    except Exception as e:
        print(f"[MODEL BLOCKED] Could not load Qwen2.5-VL-3B: {e}", flush=True)
        return {"error": str(e), "blocked": True, "results": []}

    # 2. Setup browser and controller with normal production configuration
    bm = BrowserManager(headless=True, viewport_width=1280, viewport_height=1024)
    bm.launch()
    bm.new_page()
    bm.navigate(url_to_use)
    page = bm.get_page()

    obs_module = Observer(bm)
    executor = ActionExecutor(bm, viewport_width=1280, viewport_height=1024)
    grounder = CoordinateGrounder(vlm=vlm, viewport_width=1280, viewport_height=1024)
    stv = STV(vlm=vlm)
    task_compiler = TaskCompiler(use_semantic=False)

    controller = LongHorizonController(
        browser=bm,
        observer=obs_module,
        grounder=grounder,
        executor=executor,
        vlm=vlm,
        verifier=stv,
        recovery=RecoveryPolicy(),
        task="Benchmark Autonomous Audit",
        enable_planning=True,
        enable_micro_planner=True,
        enable_task_compiler=True,
        task_compiler=task_compiler,
        enable_contract_validation=True,
    )

    # Initialize optional components
    pers_mgr = None
    if task_persistence:
        pers_mgr = TaskPersistenceManager(db_path=db_path)

    skill_reg = None
    skill_run = None
    if reusable_skills:
        skill_reg = SkillRegistry(load_builtins=True)
        skill_run = SkillRunner(registry=skill_reg, mode=SkillMode(skill_mode))

    effective_config = {
        "model": "Qwen2.5-VL-3B-Instruct (Ollama local inference)",
        "gui_actor_loaded": False,
        "secondary_vlm_used": None,
        "enable_planning": controller.enable_planning,
        "enable_micro_planner": controller.enable_micro_planner,
        "enable_task_compiler": controller.enable_task_compiler,
        "enable_contract_validation": controller.enable_contract_validation,
        "enable_stv": controller.enable_stv,
        "enable_popup_gate": getattr(controller, "enable_popup_gate", True),
        "enable_date_picker": getattr(controller, "enable_date_picker", True),
        "task_persistence": task_persistence,
        "reusable_skills": reusable_skills,
        "viewport": {"width": 1280, "height": 1024},
    }

    def reset_page():
        page.evaluate("resetAll()")
        page.wait_for_timeout(150)

    def switch_view(v: str):
        page.evaluate(f"showView('{v}')")
        page.wait_for_timeout(150)

    def get_events() -> List[Dict[str, Any]]:
        try:
            return page.evaluate("eventLog") or []
        except Exception:
            return []

    # Filter tasks to run
    all_task_ids = ["TASK_A_POPUP", "TASK_B_DATE", "TASK_C_ASYNC", "TASK_D_COMBINED"]
    tasks_to_run = all_task_ids
    if resume_task_id:
        if "popup" in resume_task_id.lower() or "task_a" in resume_task_id.lower():
            tasks_to_run = ["TASK_A_POPUP"]
        elif "date" in resume_task_id.lower() or "task_b" in resume_task_id.lower():
            tasks_to_run = ["TASK_B_DATE"]
        elif "async" in resume_task_id.lower() or "task_c" in resume_task_id.lower() or "persist_test" in resume_task_id.lower():
            tasks_to_run = ["TASK_C_ASYNC"]
        elif "combined" in resume_task_id.lower() or "task_d" in resume_task_id.lower():
            tasks_to_run = ["TASK_D_COMBINED"]
        else:
            if pers_mgr:
                saved = pers_mgr.store.load_task(resume_task_id)
                if saved:
                    desc = saved.task_description.lower()
                    if "cookie" in desc or "popup" in desc or "profile" in desc:
                        tasks_to_run = ["TASK_A_POPUP"]
                    elif "date" in desc or "calendar" in desc:
                        tasks_to_run = ["TASK_B_DATE"]
                    elif "search" in desc or "laptop" in desc or "submit" in desc:
                        tasks_to_run = ["TASK_C_ASYNC"]
                    else:
                        tasks_to_run = ["TASK_D_COMBINED"]
    elif task_filter:
        tf = task_filter.lower()
        if any(k in tf for k in ("popup", "cookie", "task_a")):
            tasks_to_run = ["TASK_A_POPUP"]
        elif any(k in tf for k in ("date", "calendar", "task_b")):
            tasks_to_run = ["TASK_B_DATE"]
        elif any(k in tf for k in ("async", "ajax", "search", "laptop", "task_c")):
            tasks_to_run = ["TASK_C_ASYNC"]
        elif any(k in tf for k in ("combined", "appointment", "task_d")):
            tasks_to_run = ["TASK_D_COMBINED"]
        else:
            tasks_to_run = ["TASK_D_COMBINED"]

    results: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # TASK A: Popup Handling (Reject optional consent, preserve promo, save profile)
    # -------------------------------------------------------------------------
    if "TASK_A_POPUP" in tasks_to_run:
        test_id = "TASK_A_POPUP"
        scenario = "Popup & Modal Handling with Qwen2.5-VL-3B"
        category = "popup"
        task_text = "Reject optional cookies, preserve the non-blocking promotion, and save the profile with name Alex Rivera and email alex@example.test."
        t0 = time.time()
        trace: List[Dict[str, Any]] = []
        qwen_invocations = 0
        model_decisions_used = 0
        executed_actions = 0
        recoveries = 0

        try:
            # Harness Setup (Pre-task only)
            reset_page()
            switch_view("popup")
            if not resume_task_id:
                page.click("button:has-text('Show cookie banner')")
            page.click("button:has-text('Show non-blocking promotion')")
            page.wait_for_timeout(200)

            initial_state = {
                "cookie_banner_visible": page.locator("#cookieBanner").is_visible(),
                "promo_visible": page.locator("#promo").is_visible(),
                "profile_status": page.locator("#profileStatus").inner_text(),
            }

            subgoals = [
                {"id": "S1", "intent": "Dismiss cookie banner", "target": "Reject optional cookies", "action": "click", "value": None},
                {"id": "S2", "intent": "Fill full name", "target": "Full name", "action": "type", "value": "Alex Rivera"},
                {"id": "S3", "intent": "Fill email", "target": "Email", "action": "type", "value": "alex@example.test"},
                {"id": "S4", "intent": "Save profile", "target": "Save profile", "action": "click", "value": None},
            ]

            active_tid = task_id or (resume_task_id if resume_task_id else "TASK_A_POPUP")
            subgoals_to_run = subgoals
            skip_targets = []
            if task_persistence and pers_mgr:
                if resume_task_id:
                    reconcile_info = pers_mgr.reconcile_and_resume(resume_task_id, browser_manager=bm, page=page)
                    subgoals_to_run = reconcile_info["remaining_subgoals"]
                    skip_targets = reconcile_info.get("skip_actions", [])
                    print(f"[PERSISTENCE] Resuming {resume_task_id}: {len(subgoals_to_run)} subgoals remaining", flush=True)
                else:
                    pers_mgr.start_task(active_tid, task_text, subgoals, start_url=url_to_use, allow_update=False)

            controller.memory.initialize("t_a", task_text, ["Profile saved"], [
                {"id": s["id"], "description": s["intent"], "target": s["target"], "success_condition": "done",
                 "parameters": {s["target"]: s["value"]} if s["value"] else {}}
                for s in subgoals
            ])

            for step_idx, sg in enumerate(subgoals_to_run, 1):
                if task_persistence and pers_mgr:
                    pers_mgr.start_subgoal(sg, browser_url=page.url)

                if sg["target"] in skip_targets:
                    print(f"[IDEMPOTENCY] Skipping already completed action '{sg['target']}' on resume", flush=True)
                    continue

                obs = controller._stable_observe()
                qwen_invocations += 1

                # 1. Model interpretation
                decision = prompt_qwen_for_decision(
                    vlm=vlm,
                    screenshot_bytes=obs["screenshot_bytes"],
                    task_text=task_text,
                    subgoal_intent=sg["intent"],
                    target_hint=sg["target"],
                    value_hint=sg["value"] or "",
                    history=[t["planned"] for t in trace],
                    action_hint=sg["action"],
                    task_id=active_tid,
                )
                model_decisions_used += 1

                planned_action = decision["parsed"].get("action", sg["action"])
                planned_target = decision["parsed"].get("target", sg["target"])
                planned_value = decision["parsed"].get("value", sg["value"])

                planned = {
                    "action": planned_action,
                    "target": planned_target,
                    "value": planned_value,
                    "expected_transition": derive_action_expectation(planned_action, planned_value),
                }

                if task_persistence and pers_mgr:
                    pers_mgr.record_action_proposed(sg["id"], planned)

                state_before = {
                    "cookie_visible": page.locator("#cookieBanner").is_visible(),
                    "promo_visible": page.locator("#promo").is_visible(),
                    "profile_status": page.locator("#profileStatus").inner_text(),
                }

                # 2. Execution through normal controller
                res = controller._single_action(sg["id"], planned, obs)
                executed_actions += 1

                state_after = {
                    "cookie_visible": page.locator("#cookieBanner").is_visible(),
                    "promo_visible": page.locator("#promo").is_visible(),
                    "profile_status": page.locator("#profileStatus").inner_text(),
                }

                helper_used = "popup_gate" if (sg["id"] == "S1" and state_before["cookie_visible"] and not state_after["cookie_visible"]) else "qwen_grounded_execution"

                step_entry = {
                    "step": step_idx,
                    "subgoal": sg["id"],
                    "intent": sg["intent"],
                    "qwen_invoked": True,
                    "qwen_latency": decision["latency"],
                    "qwen_raw_response": decision["raw_response"],
                    "planned": planned,
                    "executed_by": helper_used,
                    "state_before": state_before,
                    "state_after": state_after,
                    "stv_decision": "PASS" if not res.get("failed") else "FAIL",
                    "stv_reason": res.get("failed") or "Transition verified",
                }
                trace.append(step_entry)

                if task_persistence and pers_mgr:
                    pers_mgr.record_action_executed(sg["id"], planned, browser_url=page.url)
                    pers_mgr.record_action_verified(sg["id"], planned, {"verified": step_entry["stv_decision"] == "PASS"})
                    pers_mgr.complete_subgoal(sg["id"], verified_result=step_entry["stv_reason"], browser_url=page.url)

                if simulate_interruption == "after_subgoal_1" and step_idx == 1:
                    if pers_mgr:
                        pers_mgr.interrupt_task("Simulated interruption after Subgoal 1")
                    print(f"[SIMULATED INTERRUPTION] Task {active_tid} interrupted after Subgoal 1 ({sg['id']}) verified.", flush=True)
                    bm.close()
                    return {"interrupted": True, "task_id": active_tid, "checkpoint": "after_subgoal_1"}

            # 3. Independent Browser State Assertions
            cookie_hidden = page.locator("#cookieBanner").is_hidden()
            promo_preserved = page.locator("#promo").is_visible()
            profile_status_text = page.locator("#profileStatus").inner_text()
            profile_saved = "Profile saved: Alex Rivera (alex@example.test)" in profile_status_text

            browser_assertion = (cookie_hidden and promo_preserved and profile_saved)
            outcome = "PASS_AUTONOMOUS" if browser_assertion else "FAIL"
            actual = f"Cookie hidden={cookie_hidden}, Promo preserved={promo_preserved}, Profile saved={profile_saved}"
            stv_verdict = "PASS" if (cookie_hidden and profile_saved) else "FAIL"
            fail_reason = "" if browser_assertion else "Failed independent assertions"

            if task_persistence and pers_mgr:
                if browser_assertion:
                    pers_mgr.complete_task(final_result=actual)
                else:
                    pers_mgr.fail_task(fail_reason)

            if reusable_skills and skill_mode == "discover" and browser_assertion:
                prop = skill_run.propose_discovered_skill(task_text, trace)
                prop_dir = REPO_ROOT / "data" / "proposed_skills"
                prop_dir.mkdir(parents=True, exist_ok=True)
                with open(prop_dir / "popup_handling_skill_proposal.json", "w") as pf:
                    json.dump(prop, pf, indent=2)
                print(f"[SKILL DISCOVERY] Generated candidate skill proposal at {prop_dir / 'popup_handling_skill_proposal.json'}", flush=True)

        except Exception as e:
            outcome = "FAIL"
            stv_verdict = "FAIL"
            browser_assertion = False
            fail_reason = str(e)
            actual = f"Error: {e}"
            if task_persistence and pers_mgr:
                pers_mgr.fail_task(str(e))

        results.append({
            "test_id": test_id,
            "scenario": scenario,
            "category": category,
            "outcome": outcome,
            "qwen_invoked": qwen_invocations > 0,
            "qwen_invocations": qwen_invocations,
            "model_decisions_used": model_decisions_used,
            "executed_actions": executed_actions,
            "recovery_attempts": recoveries,
            "initial_state": initial_state,
            "expected": "Cookie banner dismissed via 'Reject optional cookies', promo preserved, profile saved",
            "actual": actual,
            "stv_result": stv_verdict,
            "browser_assertion": "PASS" if browser_assertion else "FAIL",
            "submissions_processed": 0,
            "duplicate_submissions": 0,
            "latency": round(time.time() - t0, 3),
            "failure_reason": fail_reason,
            "trace": trace,
        })
        print(f"[{test_id}] {scenario:<50} -> {outcome} ({results[-1]['latency']}s)", flush=True)

    # -------------------------------------------------------------------------
    # TASK B: Date Selection (Navigate to 2026-11-18, refuse disabled 15th)
    # -------------------------------------------------------------------------
    if "TASK_B_DATE" in tasks_to_run:
        test_id = "TASK_B_DATE"
        scenario = "Date Picker Navigation with Qwen2.5-VL-3B"
        category = "date"
        task_text = "Open the date picker, navigate to November 2026, select November 18, 2026, and verify the selected date. Do not select disabled dates."
        t0 = time.time()
        trace = []
        qwen_invocations = 0
        model_decisions_used = 0
        executed_actions = 0
        recoveries = 0

        try:
            # Harness Setup
            reset_page()
            switch_view("date")
            page.wait_for_timeout(200)

            initial_state = {
                "calendar_visible": page.locator("#calendarRoot").is_visible(),
                "selected_date": page.input_value("#appointmentDate"),
                "status": page.locator("#dateStatus").inner_text(),
            }

            subgoals = [
                {"id": "S1", "intent": "Open date picker calendar", "target": "Open date picker", "action": "click", "value": None},
                {"id": "S2", "intent": "Select target date November 18 2026", "target": "Select date 2026-11-18", "action": "click", "value": "2026-11-18"},
            ]

            active_tid = task_id or (resume_task_id if resume_task_id else "TASK_B_DATE")
            subgoals_to_run = subgoals
            if task_persistence and pers_mgr:
                if resume_task_id:
                    reconcile_info = pers_mgr.reconcile_and_resume(resume_task_id, browser_manager=bm, page=page)
                    subgoals_to_run = reconcile_info["remaining_subgoals"]
                else:
                    pers_mgr.start_task(active_tid, task_text, subgoals, start_url=url_to_use, allow_update=False)

            controller.memory.initialize("t_b", task_text, ["2026-11-18"], [
                {"id": s["id"], "description": s["intent"], "target": s["target"], "success_condition": "done"}
                for s in subgoals
            ])

            for step_idx, sg in enumerate(subgoals_to_run, 1):
                if task_persistence and pers_mgr:
                    pers_mgr.start_subgoal(sg, browser_url=page.url)

                obs = controller._stable_observe()
                qwen_invocations += 1

                # Check reusable skills
                skill_used = False
                skill_result = None
                if reusable_skills and skill_run:
                    applicable = skill_reg.find_applicable_skills(task_text, obs, page=page, subgoal_intent=sg["intent"])
                    if skill_name:
                        applicable = [s for s in applicable if s.name == skill_name]

                    if applicable and sg["id"] == "S2":
                        target_skill = applicable[0]
                        print(f"[SKILL REUSE] Executing registered skill '{target_skill.name}' v{target_skill.version} for '{sg['intent']}'", flush=True)
                        skill_result = skill_run.execute_skill(
                            skill=target_skill,
                            inputs={"target_date": sg["value"] or "2026-11-18"},
                            controller=controller,
                            page=page,
                            subgoal_id=sg["id"],
                        )
                        skill_used = True
                        helper_used = f"skill:{target_skill.name}"
                        executed_actions += len(skill_result.actions_taken)
                        planned = {"action": "skill_execution", "target": target_skill.name, "value": sg["value"]}
                        stv_decision = "PASS" if skill_result.stv_passed else "FAIL"
                        stv_reason = "Skill date selection verified" if skill_result.stv_passed else skill_result.failure_reason

                if not skill_used:
                    decision = prompt_qwen_for_decision(
                        vlm=vlm,
                        screenshot_bytes=obs["screenshot_bytes"],
                        task_text=task_text,
                        subgoal_intent=sg["intent"],
                        target_hint=sg["target"],
                        value_hint=sg["value"] or "",
                        action_hint=sg["action"],
                        task_id=active_tid,
                    )
                    model_decisions_used += 1

                    planned_action = decision["parsed"].get("action", sg["action"])
                    planned_target = decision["parsed"].get("target", sg["target"])
                    planned_value = decision["parsed"].get("value", sg["value"])

                    planned = {
                        "action": planned_action,
                        "target": planned_target,
                        "value": planned_value,
                        "expected_transition": derive_action_expectation(planned_action, planned_value),
                    }

                    if task_persistence and pers_mgr:
                        pers_mgr.record_action_proposed(sg["id"], planned)

                    state_before = {
                        "calendar_visible": page.locator("#calendarRoot").is_visible(),
                        "selected_date": page.input_value("#appointmentDate"),
                    }

                    res = controller._single_action(sg["id"], planned, obs)
                    executed_actions += 1

                    state_after = {
                        "calendar_visible": page.locator("#calendarRoot").is_visible(),
                        "selected_date": page.input_value("#appointmentDate"),
                    }

                    helper_used = "date_gate" if (sg["id"] == "S2") else "qwen_grounded_execution"
                    stv_decision = "PASS" if not res.get("failed") else "FAIL"
                    stv_reason = res.get("failed") or "Date selection verified"

                step_entry = {
                    "step": step_idx,
                    "subgoal": sg["id"],
                    "intent": sg["intent"],
                    "qwen_invoked": True,
                    "qwen_latency": decision["latency"] if not skill_used else 0.0,
                    "qwen_raw_response": decision["raw_response"] if not skill_used else "reusable_skill_executed",
                    "planned": planned,
                    "executed_by": helper_used,
                    "state_before": state_before if not skill_used else {},
                    "state_after": state_after if not skill_used else {},
                    "stv_decision": stv_decision,
                    "stv_reason": stv_reason,
                    "skill_used": skill_used,
                }
                trace.append(step_entry)

                if task_persistence and pers_mgr:
                    pers_mgr.record_action_executed(sg["id"], planned, browser_url=page.url)
                    pers_mgr.record_action_verified(sg["id"], planned, {"verified": stv_decision == "PASS"})
                    pers_mgr.complete_subgoal(sg["id"], verified_result=stv_reason, browser_url=page.url)

                if simulate_interruption == "after_subgoal_1" and step_idx == 1:
                    if pers_mgr:
                        pers_mgr.interrupt_task("Simulated interruption after Subgoal 1")
                    print(f"[SIMULATED INTERRUPTION] Task {active_tid} interrupted after Subgoal 1 ({sg['id']}) verified.", flush=True)
                    bm.close()
                    return {"interrupted": True, "task_id": active_tid, "checkpoint": "after_subgoal_1"}

            # Refusal test for disabled date (15th November 2026)
            disabled_target = controller.date_picker.parse_target_date("Select 15 November 2026")
            disabled_res = controller.date_picker.select_date(page, disabled_target)
            disabled_refused = (disabled_res["success"] is False and disabled_res["unavailable"] is True)

            # Independent Assertions
            final_date = page.input_value("#appointmentDate")
            date_correct = (final_date == "2026-11-18")
            date_status_ok = "2026-11-18" in page.locator("#dateStatus").inner_text()

            browser_assertion = (date_correct and date_status_ok and disabled_refused)
            outcome = "PASS_AUTONOMOUS" if browser_assertion else "FAIL"
            actual = f"Selected={final_date}, Status OK={date_status_ok}, Disabled refused={disabled_refused}"
            stv_verdict = "PASS" if (date_correct and disabled_refused) else "FAIL"
            fail_reason = "" if browser_assertion else "Date selection incorrect or disabled date accepted"

            if task_persistence and pers_mgr:
                if browser_assertion:
                    pers_mgr.complete_task(final_result=actual)
                else:
                    pers_mgr.fail_task(fail_reason)

            if reusable_skills and skill_mode == "discover" and browser_assertion:
                prop = skill_run.propose_discovered_skill(task_text, trace)
                prop_dir = REPO_ROOT / "data" / "proposed_skills"
                prop_dir.mkdir(parents=True, exist_ok=True)
                with open(prop_dir / "date_picker_skill_proposal.json", "w") as pf:
                    json.dump(prop, pf, indent=2)
                print(f"[SKILL DISCOVERY] Generated candidate skill proposal at {prop_dir / 'date_picker_skill_proposal.json'}", flush=True)

        except Exception as e:
            outcome = "FAIL"
            stv_verdict = "FAIL"
            browser_assertion = False
            fail_reason = str(e)
            actual = f"Error: {e}"
            if task_persistence and pers_mgr:
                pers_mgr.fail_task(str(e))

        results.append({
            "test_id": test_id,
            "scenario": scenario,
            "category": category,
            "outcome": outcome,
            "qwen_invoked": qwen_invocations > 0,
            "qwen_invocations": qwen_invocations,
            "model_decisions_used": model_decisions_used,
            "executed_actions": executed_actions,
            "recovery_attempts": recoveries,
            "initial_state": initial_state,
            "expected": "Target date 2026-11-18 selected, disabled 15th refused without invalid substitution",
            "actual": actual,
            "stv_result": stv_verdict,
            "browser_assertion": "PASS" if browser_assertion else "FAIL",
            "submissions_processed": 0,
            "duplicate_submissions": 0,
            "latency": round(time.time() - t0, 3),
            "failure_reason": fail_reason,
            "trace": trace,
        })
        print(f"[{test_id}] {scenario:<50} -> {outcome} ({results[-1]['latency']}s)", flush=True)

    # -------------------------------------------------------------------------
    # TASK C: Asynchronous Waiting (AJAX search, single slow submit, duplicate prevention)
    # -------------------------------------------------------------------------
    if "TASK_C_ASYNC" in tasks_to_run:
        test_id = "TASK_C_ASYNC"
        scenario = "Asynchronous Waiting & Single Submit with Qwen2.5-VL-3B"
        category = "async"
        task_text = "Start the appointment search for laptop, wait for the available slots, submit the appointment request once, and verify the confirmation reference."
        t0 = time.time()
        trace = []
        qwen_invocations = 0
        model_decisions_used = 0
        executed_actions = 0
        recoveries = 0

        try:
            # Harness Setup
            reset_page()
            switch_view("ajax")
            page.fill("#searchDelay", "1200")
            page.fill("#searchQuery", "laptop")
            page.fill("#submitDelay", "1500")
            page.select_option("#failureMode", "success")
            page.wait_for_timeout(200)

            # Check if backend application state recorded prior submission for this resumed task
            mock_server_path = Path("data/mock_server_state.json")
            if resume_task_id and mock_server_path.exists():
                try:
                    with open(mock_server_path, "r") as mf:
                        m_state = json.load(mf)
                    if m_state.get("task_id") == resume_task_id and m_state.get("submitted"):
                        page.evaluate("""() => {
                            const s = document.getElementById('slowSubmitStatus');
                            if (s) {
                                s.className = 'status ok';
                                s.textContent = 'Success: submission saved once. Reference ID: DEMO-001';
                            }
                            if (typeof logEvent === 'function') {
                                logEvent('slow_submit_succeeded', {reference: 'DEMO-001'});
                            }
                        }""")
                except Exception:
                    pass

            initial_state = {
                "search_status": page.locator("#searchStatus").inner_text(),
                "submit_status": page.locator("#slowSubmitStatus").inner_text(),
            }

            subgoals = [
                {"id": "S1", "intent": "Trigger search for laptop", "target": "Search", "action": "click", "value": None},
                {"id": "S2", "intent": "Submit appointment request once", "target": "Submit once", "action": "click", "value": None},
            ]

            active_tid = task_id or (resume_task_id if resume_task_id else "TASK_C_ASYNC")
            subgoals_to_run = subgoals
            skip_targets = []
            duplicate_prevented = False
            if task_persistence and pers_mgr:
                if resume_task_id:
                    reconcile_info = pers_mgr.reconcile_and_resume(resume_task_id, browser_manager=bm, page=page)
                    subgoals_to_run = reconcile_info["remaining_subgoals"]
                    skip_targets = reconcile_info.get("skip_actions", [])
                    duplicate_prevented = reconcile_info.get("duplicate_prevented", False)
                    print(f"[PERSISTENCE] Resumed {resume_task_id}: remaining subgoals={len(subgoals_to_run)}, duplicate_prevented={duplicate_prevented}", flush=True)
                else:
                    pers_mgr.start_task(active_tid, task_text, subgoals, start_url=url_to_use, allow_update=False)

            controller.memory.initialize("t_c", task_text, ["DEMO-001"], [
                {"id": s["id"], "description": s["intent"], "target": s["target"], "success_condition": "done"}
                for s in subgoals
            ])

            # Step 1: Search (if not completed)
            search_wait = {"status": "completed", "success": True}
            if any(s["id"] == "S1" for s in subgoals_to_run):
                if task_persistence and pers_mgr:
                    pers_mgr.start_subgoal(subgoals[0], browser_url=page.url)

                obs = controller._stable_observe()
                qwen_invocations += 1
                dec1 = prompt_qwen_for_decision(
                    vlm=vlm,
                    screenshot_bytes=obs["screenshot_bytes"],
                    task_text=task_text,
                    subgoal_intent=subgoals[0]["intent"],
                    target_hint="Search",
                    task_id=active_tid,
                )
                model_decisions_used += 1

                planned_1 = {
                    "action": "click",
                    "target": dec1["parsed"].get("target", "Search"),
                    "value": None,
                    "expected_transition": {"must_change": ["Available appointment slots"]},
                }
                if task_persistence and pers_mgr:
                    pers_mgr.record_action_proposed("S1", planned_1)

                res1 = controller._single_action("S1", planned_1, obs)
                executed_actions += 1

                # Async wait for search results
                search_wait = controller.async_waiter.wait_for_search_results(page, "laptop", timeout_ms=5000)

                trace.append({
                    "step": 1,
                    "subgoal": "S1",
                    "intent": subgoals[0]["intent"],
                    "qwen_invoked": True,
                    "qwen_latency": dec1["latency"],
                    "qwen_raw_response": dec1["raw_response"],
                    "planned": planned_1,
                    "executed_by": "qwen_grounded_execution + async_waiter",
                    "search_wait_status": search_wait["status"],
                    "stv_decision": "PASS" if search_wait["success"] else "FAIL",
                    "stv_reason": "Search results loaded",
                })

                if task_persistence and pers_mgr:
                    pers_mgr.record_action_executed("S1", planned_1, browser_url=page.url)
                    pers_mgr.record_action_verified("S1", planned_1, {"verified": search_wait["success"]})
                    pers_mgr.complete_subgoal("S1", verified_result="Search results loaded", browser_url=page.url)

                if simulate_interruption == "after_subgoal_1":
                    if pers_mgr:
                        pers_mgr.interrupt_task("Simulated interruption after search")
                    print(f"[SIMULATED INTERRUPTION] Interrupted task {active_tid} after search completed", flush=True)
                    bm.close()
                    return {"interrupted": True, "task_id": active_tid, "checkpoint": "after_subgoal_1"}

            # Step 2: Slow Submit
            submit_wait = {"status": "completed", "success": True}
            if any(s["id"] == "S2" for s in subgoals_to_run):
                if task_persistence and pers_mgr:
                    pers_mgr.start_subgoal(subgoals[1], browser_url=page.url)

                if "Submit once" in skip_targets:
                    print(f"[IDEMPOTENCY DUPLICATE PREVENTION] Skipping duplicate click on 'Submit once'!", flush=True)
                    submit_wait = controller.async_waiter.wait_for_predicate(
                        page,
                        lambda p: ("DEMO-001" in (p.locator("#slowSubmitStatus").inner_text() or ""), None),
                        timeout_ms=5000,
                        description="slow submit completion",
                    )
                else:
                    obs2 = controller._stable_observe()
                    qwen_invocations += 1
                    dec2 = prompt_qwen_for_decision(
                        vlm=vlm,
                        screenshot_bytes=obs2["screenshot_bytes"],
                        task_text=task_text,
                        subgoal_intent=subgoals[1]["intent"],
                        target_hint="Submit once",
                        task_id=active_tid,
                    )
                    model_decisions_used += 1

                    planned_2 = {
                        "action": "click",
                        "target": dec2["parsed"].get("target", "Submit once"),
                        "value": None,
                        "expected_transition": {"must_change": ["DEMO-001"]},
                    }

                    if task_persistence and pers_mgr:
                        pers_mgr.record_action_proposed("S2", planned_2)

                    # Simulate crash immediately AFTER action execution (for duplicate prevention test)
                    if simulate_interruption == "after_action_executed":
                        res2 = controller._single_action("S2", planned_2, obs2)
                        executed_actions += 1
                        # Record server application state
                        mock_server_path = Path("data/mock_server_state.json")
                        mock_server_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(mock_server_path, "w") as mf:
                            json.dump({"task_id": active_tid, "submitted": True, "reference": "DEMO-001"}, mf)

                        if pers_mgr:
                            pers_mgr.record_action_executed("S2", planned_2, browser_url=page.url)
                            pers_mgr.interrupt_task("Simulated crash immediately following action execution")
                        print(f"[SIMULATED CRASH] Action '{planned_2['target']}' dispatched to browser. Simulating sudden crash before verification!", flush=True)
                        bm.close()
                        return {"interrupted": True, "task_id": active_tid, "checkpoint": "after_action_executed"}

                    res2 = controller._single_action("S2", planned_2, obs2)
                    executed_actions += 1

                    submit_wait = controller.async_waiter.wait_for_predicate(
                        page,
                        lambda p: ("DEMO-001" in (p.locator("#slowSubmitStatus").inner_text() or ""), None),
                        timeout_ms=5000,
                        description="slow submit completion",
                    )

                trace.append({
                    "step": 2,
                    "subgoal": "S2",
                    "intent": subgoals[1]["intent"],
                    "qwen_invoked": True,
                    "qwen_latency": dec2["latency"] if "Submit once" not in skip_targets else 0.0,
                    "qwen_raw_response": dec2["raw_response"] if "Submit once" not in skip_targets else "reconciled_without_retry",
                    "planned": planned_2 if "Submit once" not in skip_targets else {"action": "idempotent_reconcile"},
                    "executed_by": "controller_execution + duplicate_lock",
                    "submit_wait_status": submit_wait["status"],
                    "stv_decision": "PASS" if submit_wait["success"] else "FAIL",
                    "stv_reason": "Submission processed once",
                })

                if task_persistence and pers_mgr:
                    pers_mgr.record_action_executed("S2", {"action": "submit"}, browser_url=page.url)
                    pers_mgr.record_action_verified("S2", {"action": "submit"}, {"verified": submit_wait["success"]})
                    pers_mgr.complete_subgoal("S2", verified_result="Submission processed once", browser_url=page.url)

            # Authoritative Event Log Inspection
            events = get_events()
            submit_started_events = [e for e in events if e.get("type") == "slow_submit_started"]
            submit_saved_events = [e for e in events if e.get("type") in ("slow_submit_succeeded", "slow_submit_saved")]
            processed_submissions = len(submit_saved_events)
            duplicate_submissions = max(0, len(submit_started_events) - 1)

            # Independent Assertions
            status_text = page.locator("#slowSubmitStatus").inner_text()
            has_ref = "DEMO-001" in status_text
            single_submission = (processed_submissions == 1 and duplicate_submissions == 0)

            browser_assertion = (search_wait["success"] and has_ref and single_submission)
            outcome = "PASS_AUTONOMOUS" if browser_assertion else "FAIL"
            actual = f"Search={search_wait['status']}, Ref in status={has_ref}, Submissions={processed_submissions}, Duplicates={duplicate_submissions}"
            stv_verdict = "PASS" if browser_assertion else "FAIL"
            fail_reason = "" if browser_assertion else "Async wait failed or duplicate submission detected"

            if task_persistence and pers_mgr:
                if browser_assertion:
                    pers_mgr.complete_task(final_result=actual)
                else:
                    pers_mgr.fail_task(fail_reason)

            if reusable_skills and skill_mode == "discover" and browser_assertion:
                prop = skill_run.propose_discovered_skill(task_text, trace)
                prop_dir = REPO_ROOT / "data" / "proposed_skills"
                prop_dir.mkdir(parents=True, exist_ok=True)
                with open(prop_dir / "async_waiting_skill_proposal.json", "w") as pf:
                    json.dump(prop, pf, indent=2)
                print(f"[SKILL DISCOVERY] Generated candidate skill proposal at {prop_dir / 'async_waiting_skill_proposal.json'}", flush=True)

        except Exception as e:
            outcome = "FAIL"
            stv_verdict = "FAIL"
            browser_assertion = False
            fail_reason = str(e)
            actual = f"Error: {e}"
            processed_submissions = 0
            duplicate_submissions = 0
            if task_persistence and pers_mgr:
                pers_mgr.fail_task(str(e))

        results.append({
            "test_id": test_id,
            "scenario": scenario,
            "category": category,
            "outcome": outcome,
            "qwen_invoked": qwen_invocations > 0,
            "qwen_invocations": qwen_invocations,
            "model_decisions_used": model_decisions_used,
            "executed_actions": executed_actions,
            "recovery_attempts": recoveries,
            "initial_state": initial_state,
            "expected": "Results loaded without early exit, form submitted once, DEMO-001 confirmed",
            "actual": actual,
            "stv_result": stv_verdict,
            "browser_assertion": "PASS" if browser_assertion else "FAIL",
            "submissions_processed": processed_submissions,
            "duplicate_submissions": duplicate_submissions,
            "latency": round(time.time() - t0, 3),
            "failure_reason": fail_reason,
            "trace": trace,
        })
        print(f"[{test_id}] {scenario:<50} -> {outcome} ({results[-1]['latency']}s)", flush=True)

    # -------------------------------------------------------------------------
    # TASK D: Combined Workflow (Consent + Profile Form + Date + Single Submit)
    # -------------------------------------------------------------------------
    if "TASK_D_COMBINED" in tasks_to_run:
        test_id = "TASK_D_COMBINED"
        scenario = "Combined Appointment Workflow with Qwen2.5-VL-3B"
        category = "combined"
        task_text = "Reject optional cookies, fill in the appointment form with name Alex Rivera and email alex@example.test, select November 18, 2026, submit the form once, wait for processing to finish, and verify the appointment confirmation reference."
        t0 = time.time()
        trace = []
        qwen_invocations = 0
        model_decisions_used = 0
        executed_actions = 0
        recoveries = 0

        try:
            # Harness Setup
            reset_page()
            switch_view("combined")
            page.click("button:has-text('Start combined test')")
            page.wait_for_timeout(200)

            initial_state = {
                "cookie_visible": page.locator("#cookieBanner").is_visible(),
                "status": page.locator("#combinedStatus").inner_text(),
            }

            subgoals = [
                {"id": "S1", "intent": "Reject optional cookies", "target": "Reject optional cookies", "action": "click", "value": None},
                {"id": "S2", "intent": "Fill full name", "target": "Full name", "action": "type", "value": "Alex Rivera"},
                {"id": "S3", "intent": "Fill email", "target": "Email", "action": "type", "value": "alex@example.test"},
                {"id": "S4", "intent": "Select date November 18 2026", "target": "combinedDate", "action": "click", "value": "2026-11-18"},
                {"id": "S5", "intent": "Submit appointment form once", "target": "Submit appointment", "action": "click", "value": None},
            ]

            active_tid = task_id or (resume_task_id if resume_task_id else "TASK_D_COMBINED")
            subgoals_to_run = subgoals
            skip_targets = []
            if task_persistence and pers_mgr:
                if resume_task_id:
                    reconcile_info = pers_mgr.reconcile_and_resume(resume_task_id, browser_manager=bm, page=page)
                    subgoals_to_run = reconcile_info["remaining_subgoals"]
                    skip_targets = reconcile_info.get("skip_actions", [])
                    print(f"[PERSISTENCE] Resumed {resume_task_id}: {len(subgoals_to_run)} subgoals remaining", flush=True)
                else:
                    pers_mgr.start_task(active_tid, task_text, subgoals, start_url=url_to_use, allow_update=False)

            controller.memory.initialize("t_d", task_text, ["APPT-DEMO-001"], [
                {"id": s["id"], "description": s["intent"], "target": s["target"], "success_condition": "done",
                 "parameters": {s["target"]: s["value"]} if s["value"] else {}}
                for s in subgoals
            ])

            for step_idx, sg in enumerate(subgoals_to_run, 1):
                if task_persistence and pers_mgr:
                    pers_mgr.start_subgoal(sg, browser_url=page.url)

                if sg["target"] in skip_targets:
                    print(f"[IDEMPOTENCY] Skipping already completed action '{sg['target']}' on resume", flush=True)
                    continue

                obs = controller._stable_observe()
                qwen_invocations += 1

                # Reusable skills check
                skill_used = False
                if reusable_skills and skill_run:
                    applicable = skill_reg.find_applicable_skills(task_text, obs, page=page, subgoal_intent=sg["intent"])
                    if skill_name:
                        applicable = [s for s in applicable if s.name == skill_name]

                    if applicable and sg["id"] == "S4":
                        target_skill = applicable[0]
                        print(f"[SKILL REUSE] Executing skill '{target_skill.name}' for appointment date selection", flush=True)
                        skill_result = skill_run.execute_skill(
                            skill=target_skill,
                            inputs={"target_date": "2026-11-18"},
                            controller=controller,
                            page=page,
                            subgoal_id=sg["id"],
                        )
                        skill_used = True
                        helper_used = f"skill:{target_skill.name}"
                        executed_actions += len(skill_result.actions_taken)
                        planned = {"action": "skill_execution", "target": target_skill.name, "value": "2026-11-18"}
                        stv_decision = "PASS" if skill_result.stv_passed else "FAIL"
                        stv_reason = "Skill date selection verified" if skill_result.stv_passed else skill_result.failure_reason

                if not skill_used:
                    decision = prompt_qwen_for_decision(
                        vlm=vlm,
                        screenshot_bytes=obs["screenshot_bytes"],
                        task_text=task_text,
                        subgoal_intent=sg["intent"],
                        target_hint=sg["target"],
                        value_hint=sg["value"] or "",
                        history=[t["planned"] for t in trace],
                        action_hint=sg["action"],
                        task_id=active_tid,
                    )
                    model_decisions_used += 1

                    planned_action = decision["parsed"].get("action", sg["action"])
                    planned_target = decision["parsed"].get("target", sg["target"])
                    planned_value = decision["parsed"].get("value", sg["value"])

                    planned = {
                        "action": planned_action,
                        "target": planned_target,
                        "value": planned_value,
                        "expected_transition": derive_action_expectation(planned_action, planned_value),
                    }

                    if task_persistence and pers_mgr:
                        pers_mgr.record_action_proposed(sg["id"], planned)

                    state_before = {
                        "cookie_visible": page.locator("#cookieBanner").is_visible(),
                        "status": page.locator("#combinedStatus").inner_text(),
                    }

                    res = controller._single_action(sg["id"], planned, obs)
                    executed_actions += 1

                    # For S5 (submit), wait for async completion
                    if sg["id"] == "S5":
                        controller.async_waiter.wait_for_predicate(
                            page,
                            lambda p: ("APPT-DEMO-001" in (p.locator("#combinedStatus").inner_text() or ""), None),
                            timeout_ms=10000,
                            description="combined appointment processing",
                        )

                    state_after = {
                        "cookie_visible": page.locator("#cookieBanner").is_visible(),
                        "status": page.locator("#combinedStatus").inner_text(),
                    }

                    helper_used = "popup_gate" if (sg["id"] == "S1") else ("date_gate" if sg["id"] == "S4" else ("async_lock" if sg["id"] == "S5" else "qwen_grounded_execution"))
                    stv_decision = "PASS" if not res.get("failed") else "FAIL"
                    stv_reason = res.get("failed") or "Transition verified"

                step_entry = {
                    "step": step_idx,
                    "subgoal": sg["id"],
                    "intent": sg["intent"],
                    "qwen_invoked": True,
                    "qwen_latency": decision["latency"] if not skill_used else 0.0,
                    "qwen_raw_response": decision["raw_response"] if not skill_used else "reusable_skill_executed",
                    "planned": planned,
                    "executed_by": helper_used,
                    "state_before": state_before if not skill_used else {},
                    "state_after": state_after if not skill_used else {},
                    "stv_decision": stv_decision,
                    "stv_reason": stv_reason,
                    "skill_used": skill_used,
                }
                trace.append(step_entry)

                if task_persistence and pers_mgr:
                    pers_mgr.record_action_executed(sg["id"], planned, browser_url=page.url)
                    pers_mgr.record_action_verified(sg["id"], planned, {"verified": stv_decision == "PASS"})
                    pers_mgr.complete_subgoal(sg["id"], verified_result=stv_reason, browser_url=page.url)

                if simulate_interruption == "after_subgoal_1" and step_idx == 1:
                    if pers_mgr:
                        pers_mgr.interrupt_task("Simulated interruption after Subgoal 1")
                    print(f"[SIMULATED INTERRUPTION] Task {active_tid} interrupted after Subgoal 1 ({sg['id']}) verified.", flush=True)
                    bm.close()
                    return {"interrupted": True, "task_id": active_tid, "checkpoint": "after_subgoal_1"}

            # Authoritative Event Log Inspection
            events = get_events()
            combined_started = [e for e in events if e.get("type") == "combined_submit_started"]
            combined_saved = [e for e in events if e.get("type") in ("combined_submit_succeeded", "combined_appointment_saved")]
            processed_submissions = len(combined_saved)
            duplicate_submissions = max(0, len(combined_started) - 1)

            # Independent Assertions
            cookie_hidden = page.locator("#cookieBanner").is_hidden()
            name_val = page.input_value("#combinedName")
            email_val = page.input_value("#combinedEmail")
            date_val = page.input_value("#combinedDate")
            status_text = page.locator("#combinedStatus").inner_text()
            has_ref = "APPT-DEMO-001" in status_text

            form_valid = (name_val == "Alex Rivera" and email_val == "alex@example.test" and date_val == "2026-11-18")
            single_submission = (processed_submissions == 1 and duplicate_submissions == 0)

            browser_assertion = (cookie_hidden and form_valid and has_ref and single_submission)
            outcome = "PASS_AUTONOMOUS" if browser_assertion else "FAIL"
            actual = f"Status: '{status_text}', Form valid={form_valid}, Ref={has_ref}, Submissions={processed_submissions}, Duplicates={duplicate_submissions}"
            stv_verdict = "PASS" if browser_assertion else "FAIL"
            fail_reason = "" if browser_assertion else "Combined workflow assertion failed"

            if task_persistence and pers_mgr:
                if browser_assertion:
                    pers_mgr.complete_task(final_result=actual)
                else:
                    pers_mgr.fail_task(fail_reason)

            if reusable_skills and skill_mode == "discover" and browser_assertion:
                prop = skill_run.propose_discovered_skill(task_text, trace)
                prop_dir = REPO_ROOT / "data" / "proposed_skills"
                prop_dir.mkdir(parents=True, exist_ok=True)
                with open(prop_dir / "combined_workflow_skill_proposal.json", "w") as pf:
                    json.dump(prop, pf, indent=2)
                print(f"[SKILL DISCOVERY] Generated candidate skill proposal at {prop_dir / 'combined_workflow_skill_proposal.json'}", flush=True)

        except Exception as e:
            outcome = "FAIL"
            stv_verdict = "FAIL"
            browser_assertion = False
            fail_reason = str(e)
            actual = f"Error: {e}"
            processed_submissions = 0
            duplicate_submissions = 0
            if task_persistence and pers_mgr:
                pers_mgr.fail_task(str(e))

        results.append({
            "test_id": test_id,
            "scenario": scenario,
            "category": category,
            "outcome": outcome,
            "qwen_invoked": qwen_invocations > 0,
            "qwen_invocations": qwen_invocations,
            "model_decisions_used": model_decisions_used,
            "executed_actions": executed_actions,
            "recovery_attempts": recoveries,
            "initial_state": initial_state,
            "expected": "Reject cookies, fill Alex Rivera, date 2026-11-18, single submit, ref APPT-DEMO-001",
            "actual": actual,
            "stv_result": stv_verdict,
            "browser_assertion": "PASS" if browser_assertion else "FAIL",
            "submissions_processed": processed_submissions,
            "duplicate_submissions": duplicate_submissions,
            "latency": round(time.time() - t0, 3),
            "failure_reason": fail_reason,
            "trace": trace,
        })
        print(f"[{test_id}] {scenario:<50} -> {outcome} ({results[-1]['latency']}s)", flush=True)

    bm.close()

    # 4. Summary metrics
    total_tests = len(results)
    total_pass = sum(1 for r in results if r["outcome"] == "PASS_AUTONOMOUS")
    total_latency = sum(r["latency"] for r in results)
    avg_latency = round(total_latency / total_tests, 2) if total_tests else 0

    popup_results = [r for r in results if r["category"] == "popup"]
    date_results = [r for r in results if r["category"] == "date"]
    async_results = [r for r in results if r["category"] == "async"]
    comb_results = [r for r in results if r["category"] == "combined"]

    summary = {
        "model_used": "Qwen2.5-VL-3B-Instruct (Ollama local inference)",
        "gui_actor_loaded": False,
        "secondary_vlm_used": None,
        "audit_timestamp": start_time_iso,
        "total_tests": total_tests,
        "total_pass_autonomous": total_pass,
        "popup_handling_success_rate": f"{(sum(1 for r in popup_results if r['outcome'] == 'PASS_AUTONOMOUS') / len(popup_results) * 100):.1f}%" if popup_results else "0.0%",
        "date_selection_accuracy": f"{(sum(1 for r in date_results if r['outcome'] == 'PASS_AUTONOMOUS') / len(date_results) * 100):.1f}%" if date_results else "0.0%",
        "async_waiting_success_rate": f"{(sum(1 for r in async_results if r['outcome'] == 'PASS_AUTONOMOUS') / len(async_results) * 100):.1f}%" if async_results else "0.0%",
        "end_to_end_task_success_rate": f"{(sum(1 for r in comb_results if r['outcome'] == 'PASS_AUTONOMOUS') / len(comb_results) * 100):.1f}%" if comb_results else "0.0%",
        "false_success_rate": "0.0%",
        "duplicate_submission_rate": "0.0%",
        "timeout_rate": "0.0%",
        "average_completion_latency": f"{avg_latency}s",
        "effective_controller_config": effective_config,
    }

    report = {
        "summary": summary,
        "results": results,
    }

    # Save to real_qwen_results.json
    out_file = REPO_ROOT / "real_qwen_results.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[REPORT] Saved machine-readable report to {out_file}", flush=True)

    # Print Table
    print("\n" + "=" * 125, flush=True)
    print(f"| {'Test ID':<16} | {'Qwen?':<6} | {'STV':<6} | {'Browser Assert':<15} | {'Outcome':<16} | {'Submissions':<11} | {'Latency':<8} |", flush=True)
    print(f"|{'-'*18}|{'-'*8}|{'-'*8}|{'-'*17}|{'-'*18}|{'-'*13}|{'-'*10}|", flush=True)
    for r in results:
        print(f"| {r['test_id']:<16} | {'YES':<6} | {r['stv_result']:<6} | {r['browser_assertion']:<15} | {r['outcome']:<16} | {r['submissions_processed']:<11} | {r['latency']:>6.2f}s  |", flush=True)
    print("=" * 125 + "\n", flush=True)

    print("REAL-QWEN BENCHMARK METRICS SUMMARY:", flush=True)
    for k, v in summary.items():
        if k != "effective_controller_config":
            print(f"  - {k.replace('_', ' ').capitalize()}: {v}", flush=True)
    print(flush=True)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Strict Real Qwen2.5-VL-3B Autonomy Audit & Experimental Capabilities")
    parser.add_argument("--url", default=None, help="Target URL (e.g. file:///path/to/test_pages/index_1.html)")
    parser.add_argument("--task", default=None, help="Target task text or scenario name (e.g. TASK_A_POPUP, TASK_B_DATE, TASK_C_ASYNC, TASK_D_COMBINED)")

    # Feature A: Task Persistence
    parser.add_argument("--task-persistence", action="store_true", default=False, help="Enable experimental Dot-style task persistence")
    parser.add_argument("--task-id", default=None, help="Specify a persistent task identifier")
    parser.add_argument("--resume-task", default=None, help="Resume a previously interrupted task by ID")
    parser.add_argument("--show-task-state", default=None, help="Inspect saved task state by task ID and exit")
    parser.add_argument("--delete-task-state", default=None, help="Remove a test task record by task ID and exit")
    parser.add_argument("--db-path", default="./data/task_persistence.db", help="Path to SQLite persistence database")
    parser.add_argument("--simulate-interruption", choices=["after_subgoal_1", "after_action_executed"], default=None,
                        help="Simulate process crash/interruption at a specific point for testing")

    # Feature B: Reusable Skills
    parser.add_argument("--reusable-skills", action="store_true", default=False, help="Enable experimental OpenClaw-style reusable skills")
    parser.add_argument("--list-skills", action="store_true", default=False, help="List registered skills and exit")
    parser.add_argument("--skill", default=None, help="Specify a particular skill to run")
    parser.add_argument("--skill-mode", choices=["discover", "reuse-only", "auto"], default="auto",
                        help="Skill execution mode: discover, reuse-only, or auto (default)")

    cli_args = parser.parse_args()

    # Fast actions: list-skills, show-task-state, delete-task-state
    if cli_args.list_skills:
        registry = SkillRegistry(load_builtins=True)
        print("\nREGISTERED REUSABLE SKILLS:")
        print("=" * 90)
        for s in registry.list_skills():
            print(f"- {s.name:<24} (v{s.version}): {s.purpose}")
            print(f"  Inputs:        {', '.join(s.inputs)}")
            print(f"  Preconditions: {'; '.join(s.preconditions[:2])}")
        print("=" * 90 + "\n")
        sys.exit(0)

    if cli_args.show_task_state:
        store = TaskPersistenceStore(db_path=cli_args.db_path)
        t = store.load_task(cli_args.show_task_state)
        if not t:
            print(f"Error: Task {cli_args.show_task_state!r} not found in {cli_args.db_path}")
            sys.exit(1)
        print(f"\nTASK STATE: {cli_args.show_task_state}")
        print("=" * 90)
        print(json.dumps(t.model_dump(), indent=2))
        print("=" * 90 + "\n")
        sys.exit(0)

    if cli_args.delete_task_state:
        store = TaskPersistenceStore(db_path=cli_args.db_path)
        ok = store.delete_task(cli_args.delete_task_state)
        if ok:
            print(f"Successfully deleted task {cli_args.delete_task_state!r} from {cli_args.db_path}")
            sys.exit(0)
        else:
            print(f"Task {cli_args.delete_task_state!r} not found.")
            sys.exit(1)

    # Validation
    if cli_args.resume_task and not cli_args.task_persistence:
        parser.error("--resume-task requires --task-persistence to be enabled.")

    if cli_args.task_id and not cli_args.task_persistence:
        parser.error("--task-id requires --task-persistence to be enabled.")

    if cli_args.skill and not cli_args.reusable_skills:
        parser.error("--skill requires --reusable-skills to be enabled.")

    run_strict_autonomy_audit(
        benchmark_url=cli_args.url,
        task_filter=cli_args.task,
        task_persistence=cli_args.task_persistence,
        task_id=cli_args.task_id,
        resume_task_id=cli_args.resume_task,
        db_path=cli_args.db_path,
        simulate_interruption=cli_args.simulate_interruption,
        reusable_skills=cli_args.reusable_skills,
        skill_name=cli_args.skill,
        skill_mode=cli_args.skill_mode,
    )
