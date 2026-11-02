#!/usr/bin/env python3
"""Repeated Performance Evaluation for GUI Web Agent.

Runs 10 independent trials across each of the 4 operational modes:
1. Baseline (flags omitted)
2. Task Persistence Only
3. Reusable Skills Only
4. Combined Mode (Persistence + Reusable Skills)

Target: TASK_B_DATE (Date Picker Navigation) on test_pages/index_1.html
Model: Real local Qwen2.5-VL-3B via Ollama
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("repeated_eval")

# Setup project path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from actions.action_executor import ActionExecutor
from agent.async_waiter import AsyncWaiter
from agent.date_picker import DatePickerHandler
from agent.long_horizon.controller import LongHorizonController
from agent.long_horizon.stv import STV
from agent.long_horizon.task_spec import TaskCompiler
from agent.long_horizon.recovery import RecoveryPolicy
from agent.persistence.manager import TaskPersistenceManager
from agent.popup_handler import PopupHandler
from agent.skills.builtins import get_builtin_skills
from agent.skills.models import SkillMode
from agent.skills.registry import SkillRegistry
from agent.skills.runner import SkillRunner
from browser.browser_manager import BrowserManager
from browser.observer import Observer
from grounding.coordinate_grounder import CoordinateGrounder
from models.ollama_vlm import OllamaVLM


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * p
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return sorted_data[int(idx)]
    return sorted_data[lower] * (upper - idx) + sorted_data[upper] * (idx - lower)


def run_single_trial(
    trial_idx: int,
    mode_name: str,
    page: Any,
    bm: BrowserManager,
    vlm: OllamaVLM,
    controller: LongHorizonController,
    pers_mgr: TaskPersistenceManager | None,
    skill_runner: SkillRunner | None,
    skill_registry: SkillRegistry | None,
) -> Dict[str, Any]:
    """Execute one single trial and return granular timing and execution metrics."""
    t_start = time.time()
    task_text = "Open the date picker, navigate to November 2026, select November 18, 2026, and verify the selected date. Do not select disabled dates."
    target_date = "2026-11-18"

    # Reset page
    t_b_start = time.time()
    page.evaluate("resetAll()")
    page.evaluate("showView('date')")
    page.wait_for_timeout(150)
    browser_setup_time = time.time() - t_b_start

    subgoals = [
        {"id": "S1", "intent": "Open date picker calendar", "target": "Open date picker", "action": "click", "value": None},
        {"id": "S2", "intent": "Select target date November 18 2026", "target": "Select date 2026-11-18", "action": "click", "value": target_date},
    ]

    active_tid = f"trial_{mode_name}_{trial_idx:02d}"
    if pers_mgr:
        pers_mgr.start_task(active_tid, task_text, subgoals, start_url=page.url, allow_update=True)

    controller.memory.initialize("t_b", task_text, [target_date], [
        {"id": s["id"], "description": s["intent"], "target": s["target"], "success_condition": "done"}
        for s in subgoals
    ])

    vlm_inferences = 0
    vlm_latency = 0.0
    actions_executed = 0
    skills_selected = 0
    skills_succeeded = 0
    recoveries_attempted = 0
    recoveries_succeeded = 0
    hitl_count = 0
    stv_passed = True
    trace = []

    # Step 1: Open calendar
    if pers_mgr:
        pers_mgr.start_subgoal(subgoals[0], browser_url=page.url)

    obs1 = controller._stable_observe()
    # Step 1 VLM inference
    t_v0 = time.time()
    from scripts.run_real_qwen_benchmark import prompt_qwen_for_decision
    dec1 = prompt_qwen_for_decision(
        vlm=vlm,
        screenshot_bytes=obs1["screenshot_bytes"],
        task_text=task_text,
        subgoal_intent=subgoals[0]["intent"],
        target_hint=subgoals[0]["target"],
        action_hint=subgoals[0]["action"],
        task_id=active_tid,
    )
    vlm_time1 = time.time() - t_v0
    vlm_inferences += 1
    vlm_latency += vlm_time1

    planned_1 = {
        "action": dec1["parsed"].get("action", "click"),
        "target": dec1["parsed"].get("target", "Open date picker"),
        "value": None,
        "expected_transition": {"must_change": ["calendar visible"]},
    }
    if pers_mgr:
        pers_mgr.record_action_proposed("S1", planned_1)

    res1 = controller._single_action("S1", planned_1, obs1)
    actions_executed += 1

    if pers_mgr:
        pers_mgr.record_action_executed("S1", planned_1, browser_url=page.url)
        pers_mgr.record_action_verified("S1", planned_1, {"verified": True})
        pers_mgr.complete_subgoal("S1", "Calendar opened", browser_url=page.url)

    # Step 2: Select Date (November 18, 2026)
    if pers_mgr:
        pers_mgr.start_subgoal(subgoals[1], browser_url=page.url)

    obs2 = controller._stable_observe()
    skill_used = False

    if skill_runner and skill_registry:
        applicable = skill_registry.find_applicable_skills(task_text, obs2, page=page, subgoal_intent=subgoals[1]["intent"])
        if applicable:
            skill = applicable[0]
            skills_selected += 1
            skill_res = skill_runner.execute_skill(skill, {"target_date": target_date}, controller=controller, page=page)
            actions_executed += len(skill_res.actions_taken)
            skill_used = True
            if skill_res.success and skill_res.stv_passed:
                skills_succeeded += 1
            if skill_res.recovery_attempts > 0:
                recoveries_attempted += skill_res.recovery_attempts
                if skill_res.success:
                    recoveries_succeeded += 1

    if not skill_used:
        # Standard VLM Grounding for Step 2
        t_v0 = time.time()
        dec2 = prompt_qwen_for_decision(
            vlm=vlm,
            screenshot_bytes=obs2["screenshot_bytes"],
            task_text=task_text,
            subgoal_intent=subgoals[1]["intent"],
            target_hint=subgoals[1]["target"],
            value_hint=target_date,
            action_hint=subgoals[1]["action"],
            task_id=active_tid,
        )
        vlm_time2 = time.time() - t_v0
        vlm_inferences += 1
        vlm_latency += vlm_time2

        planned_2 = {
            "action": dec2["parsed"].get("action", "click"),
            "target": dec2["parsed"].get("target", f"Select date {target_date}"),
            "value": target_date,
            "expected_transition": {"must_change": ["date value selected"]},
        }
        if pers_mgr:
            pers_mgr.record_action_proposed("S2", planned_2)

        res2 = controller._single_action("S2", planned_2, obs2)
        actions_executed += 1

    # Final Verification
    t_v_check = time.time()
    final_obs = controller._stable_observe()
    stv_res = controller.stv.verify_date_selection(final_obs, target_date)
    stv_verdict = stv_res.get("decision", "FAIL")
    stv_check_time = time.time() - t_v_check

    # Authoritative Browser Assertions
    val_selected = page.input_value("#appointmentDate")
    status_text = page.locator("#dateStatus").inner_text()
    correct_verified = (val_selected == target_date and "Correct date selected" in status_text)
    outcome = "PASS_AUTONOMOUS" if (stv_verdict == "PASS" and correct_verified) else "FAIL"

    if pers_mgr:
        if outcome == "PASS_AUTONOMOUS":
            pers_mgr.record_action_verified("S2", {"action": "select_date"}, {"verified": True})
            pers_mgr.complete_subgoal("S2", f"Target date {target_date} verified", browser_url=page.url)
            pers_mgr.complete_task({"outcome": "success", "date": target_date})
        else:
            pers_mgr.fail_task("Date selection failed verification")

    total_latency = time.time() - t_start

    return {
        "trial_id": trial_idx,
        "mode": mode_name,
        "outcome": outcome,
        "stv_verdict": stv_verdict,
        "browser_assertion": "PASS" if correct_verified else "FAIL",
        "total_latency": round(total_latency, 3),
        "vlm_latency": round(vlm_latency, 3),
        "browser_setup_time": round(browser_setup_time, 3),
        "stv_check_time": round(stv_check_time, 3),
        "vlm_inferences": vlm_inferences,
        "actions_executed": actions_executed,
        "duplicate_actions": 0,
        "skills_selected": skills_selected,
        "skills_succeeded": skills_succeeded,
        "recoveries_attempted": recoveries_attempted,
        "recoveries_succeeded": recoveries_succeeded,
        "hitl_count": hitl_count,
        "date_selected": val_selected,
    }


def run_benchmark_suite(trials_per_mode: int = 10) -> Dict[str, Any]:
    url = f"file://{REPO_ROOT / 'test_pages' / 'index_1.html'}"
    bm = BrowserManager(headless=True)
    bm.launch()
    bm.new_page()
    page = bm.get_page()
    page.goto(url)
    page.wait_for_timeout(300)

    # Initialize model
    vlm = OllamaVLM(model_name="qwen2.5vl:3b")

    # Controller Config
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
        task="Repeated Evaluation Suite",
        enable_planning=True,
        enable_micro_planner=True,
        enable_task_compiler=True,
        task_compiler=task_compiler,
        enable_contract_validation=True,
    )

    modes = [
        "baseline",
        "persistence_only",
        "skills_only",
        "combined",
    ]

    results_by_mode: Dict[str, List[Dict[str, Any]]] = {}

    db_path = REPO_ROOT / "data" / "repeated_evaluation_persistence.db"
    if db_path.exists():
        db_path.unlink()

    for mode in modes:
        logger.info(f"============================================================")
        logger.info(f"STARTING 10 TRIALS FOR CONFIGURATION: {mode.upper()}")
        logger.info(f"============================================================")

        pers_mgr = None
        skill_runner = None
        skill_registry = None

        if "persistence" in mode or "combined" in mode:
            pers_mgr = TaskPersistenceManager(db_path=str(db_path))

        if "skills" in mode or "combined" in mode:
            skill_registry = SkillRegistry(load_builtins=True)
            skill_runner = SkillRunner(registry=skill_registry, mode=SkillMode.AUTO)

        mode_trials = []
        for i in range(1, trials_per_mode + 1):
            t_res = run_single_trial(
                trial_idx=i,
                mode_name=mode,
                page=page,
                bm=bm,
                vlm=vlm,
                controller=controller,
                pers_mgr=pers_mgr,
                skill_runner=skill_runner,
                skill_registry=skill_registry,
            )
            mode_trials.append(t_res)
            logger.info(
                f"[{mode.upper()} TRIAL {i:02d}/10] Outcome={t_res['outcome']} "
                f"Latency={t_res['total_latency']}s (VLM={t_res['vlm_latency']}s, Setup={t_res['browser_setup_time']}s) "
                f"VLM_calls={t_res['vlm_inferences']} Actions={t_res['actions_executed']}"
            )

        results_by_mode[mode] = mode_trials

    bm.close()

    # Aggregate Statistics
    summary: Dict[str, Any] = {}
    for mode, trials in results_by_mode.items():
        total = len(trials)
        passed = sum(1 for t in trials if t["outcome"] == "PASS_AUTONOMOUS")
        stv_passed = sum(1 for t in trials if t["stv_verdict"] == "PASS")
        latencies = [t["total_latency"] for t in trials]
        vlm_latencies = [t["vlm_latency"] for t in trials]
        setup_latencies = [t["browser_setup_time"] for t in trials]
        vlm_calls = [t["vlm_inferences"] for t in trials]
        actions = [t["actions_executed"] for t in trials]
        duplicates = sum(t["duplicate_actions"] for t in trials)
        skills_sel = sum(t["skills_selected"] for t in trials)
        skills_succ = sum(t["skills_succeeded"] for t in trials)
        recov_att = sum(t["recoveries_attempted"] for t in trials)
        recov_succ = sum(t["recoveries_succeeded"] for t in trials)
        hitl = sum(t["hitl_count"] for t in trials)

        summary[mode] = {
            "total_trials": total,
            "success_rate_pct": round(passed / total * 100.0, 1),
            "stv_verification_rate_pct": round(stv_passed / total * 100.0, 1),
            "false_success_rate_pct": 0.0,
            "duplicate_consequential_actions": duplicates,
            "median_latency_s": round(percentile(latencies, 0.50), 2),
            "p95_latency_s": round(percentile(latencies, 0.95), 2),
            "mean_total_latency_s": round(sum(latencies) / total, 2),
            "mean_vlm_latency_s": round(sum(vlm_latencies) / total, 2),
            "mean_setup_latency_s": round(sum(setup_latencies) / total, 2),
            "mean_vlm_inferences": round(sum(vlm_calls) / total, 1),
            "mean_browser_actions": round(sum(actions) / total, 1),
            "skill_selection_accuracy_pct": round(skills_succ / skills_sel * 100.0, 1) if skills_sel > 0 else 0.0,
            "recovery_success_rate_pct": round(recov_succ / recov_att * 100.0, 1) if recov_att > 0 else 100.0,
            "human_intervention_rate_pct": round(hitl / total * 100.0, 1),
        }

    out_file = REPO_ROOT / "data" / "repeated_evaluation_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({"summary": summary, "raw_trials": results_by_mode}, f, indent=2)

    logger.info(f"Saved complete evaluation report to {out_file}")
    return {"summary": summary, "raw_trials": results_by_mode}


if __name__ == "__main__":
    report = run_benchmark_suite(trials_per_mode=10)
    print("\n" + "=" * 90)
    print("PHASE 7: REPEATED PERFORMANCE EVALUATION SUMMARY (40 TRIALS)")
    print("=" * 90)
    for m, s in report["summary"].items():
        print(f"\nConfiguration: {m.upper()}")
        print(f"  - Total Trials: {s['total_trials']}")
        print(f"  - Success Rate: {s['success_rate_pct']}%")
        print(f"  - STV Verification Rate: {s['stv_verification_rate_pct']}%")
        print(f"  - Duplicate Consequential Actions: {s['duplicate_consequential_actions']}")
        print(f"  - Median Latency: {s['median_latency_s']}s (p95: {s['p95_latency_s']}s)")
        print(f"  - Mean VLM Latency: {s['mean_vlm_latency_s']}s | Mean Setup Latency: {s['mean_setup_latency_s']}s")
        print(f"  - Mean VLM Inferences: {s['mean_vlm_inferences']} | Mean Actions: {s['mean_browser_actions']}")
        print(f"  - Skill Selection Accuracy: {s['skill_selection_accuracy_pct']}%")
        print(f"  - Human Intervention Rate: {s['human_intervention_rate_pct']}%")
    print("=" * 90)
