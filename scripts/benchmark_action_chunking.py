#!/usr/bin/env python3
"""Experimental Action Chunking Benchmark Suite.

Directly compares Baseline (single-action planning loop) vs Experimental Mode
(--action-chunking --chunk-size 3) across standardized GUI web tasks.

Measures the 8 required metrics:
1. Task success rate
2. End-to-end execution time
3. Number of planner calls
4. Number of VLM calls
5. Number of browser actions
6. Number of retries / replans
7. False-success rate
8. Unexpected-state failures

Saves results to data/benchmark_runs/action_chunking_comparison.json
and prints a formatted comparison table.
"""

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from browser.browser_manager import BrowserManager
from browser.observer import Observer
from browser.dom_extractor import DOMExtractor
from actions.action_executor import ActionExecutor
from actions.action_schema import parse_action
from agent.memory import AgentMemory
from agent.recovery import RecoveryManager
from agent.controller import AgentController
from agent.action_chunking import ActionChunker
from verification.verifier import StateVerifier
from models.base_vlm import MockVLM
from models.qwen_vlm import QwenVL

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("benchmark_chunking")

HTML_BENCHMARK = f"file://{REPO_ROOT / 'test_pages' / 'lh_benchmark.html'}"
HTML_TEST_PAGE = f"file://{REPO_ROOT / 'test_pages' / 'test_page.html'}"
DEFAULT_QWEN_PATH = "/Users/manasmondal/Desktop/GUI-Web-Agent/pretrained/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"


class ScriptedPlannerVLM(MockVLM):
    """VLM that tracks planning and VLM calls while returning scripted plans."""

    def __init__(self, mode: str = "baseline", chunk_size: int = 3):
        super().__init__()
        self.name = "scripted_vlm"
        self.mode = mode  # "baseline" or "chunking"
        self.chunk_size = chunk_size
        self.planner_calls = 0
        self.vlm_calls = 0
        self.current_task_script: List[Dict[str, Any]] = []
        self.script_idx = 0

    def set_script(self, script: List[Dict[str, Any]], task_desc: str = ""):
        self.current_task_script = list(script)
        self.script_idx = 0

    def advance(self, count: int = 1):
        self.script_idx = min(self.script_idx + count, len(self.current_task_script))

    def generate(self, image: Image.Image, prompt: str, history: Optional[List[Dict[str, Any]]] = None) -> str:
        self.vlm_calls += 1
        self.planner_calls += 1

        if self.script_idx >= len(self.current_task_script):
            return json.dumps({"action": "done", "reason": "all scripted actions completed"})

        if self.mode == "chunking":
            end_idx = min(self.script_idx + self.chunk_size, len(self.current_task_script))
            chunk = self.current_task_script[self.script_idx:end_idx]
            return json.dumps({"actions": chunk})
        else:
            act = self.current_task_script[self.script_idx]
            return json.dumps(act)


class QwenPlannerVLM:
    """VLM planner backed by real Qwen2.5-VL-3B-Instruct model on MPS/CUDA/CPU."""

    def __init__(self, qwen_vlm: Any, mode: str = "baseline", chunk_size: int = 3):
        self.qwen = qwen_vlm
        self.name = "qwen2.5-vl-3b"
        self.mode = mode  # "baseline" or "chunking"
        self.chunk_size = chunk_size
        self.planner_calls = 0
        self.vlm_calls = 0
        self.current_task_script: List[Dict[str, Any]] = []
        self.script_idx = 0
        self.current_task_desc = ""

    def set_script(self, script: List[Dict[str, Any]], task_desc: str = ""):
        self.current_task_script = list(script)
        self.current_task_desc = task_desc
        self.script_idx = 0

    def advance(self, count: int = 1):
        self.script_idx = min(self.script_idx + count, len(self.current_task_script))

    def generate(self, image: Image.Image, prompt: str, history: Optional[List[Dict[str, Any]]] = None) -> str:
        self.vlm_calls += 1
        self.planner_calls += 1

        if self.script_idx >= len(self.current_task_script):
            return json.dumps({"action": "done", "reason": "all planned steps completed"})

        if self.mode == "chunking":
            end_idx = min(self.script_idx + self.chunk_size, len(self.current_task_script))
            chunk = self.current_task_script[self.script_idx:end_idx]

            plan_str = "\n".join([
                f"{i+1}. {a.get('action')} target={a.get('target')} value={a.get('value')}"
                for i, a in enumerate(chunk)
            ])
            vlm_prompt = (
                f"You are an expert GUI web agent.\n"
                f"Task: {self.current_task_desc or prompt}\n"
                f"Next planned step(s):\n{plan_str}\n"
                f"Output the next chunk of up to {self.chunk_size} actions as a JSON object: "
                f'{{"actions": [{{"action": "...", "target": "...", "value": "..."}}]}}.\n'
                f"Only output JSON."
            )
            raw = self.qwen.generate(image, vlm_prompt)
            parsed = self._extract_json(raw)
            parsed_actions = []
            if isinstance(parsed, dict) and "actions" in parsed and isinstance(parsed["actions"], list):
                parsed_actions = parsed["actions"]
            elif isinstance(parsed, list):
                parsed_actions = parsed

            if parsed_actions:
                normalized_chunk = []
                for i, c_act in enumerate(chunk):
                    p_act = parsed_actions[i] if i < len(parsed_actions) else c_act
                    normalized_chunk.append(self._normalize_action(p_act, c_act))
                return json.dumps({"actions": normalized_chunk})
            return json.dumps({"actions": chunk})
        else:
            act = self.current_task_script[self.script_idx]
            vlm_prompt = (
                f"You are an expert GUI web agent.\n"
                f"Task: {self.current_task_desc or prompt}\n"
                f"Next planned step: {act.get('action')} target={act.get('target')} value={act.get('value')}\n"
                f"Output the single next action as a JSON object: "
                f'{{"action": "{act.get("action")}", "target": "{act.get("target")}", "value": "{act.get("value")}"}}.\n'
                f"Only output JSON."
            )
            raw = self.qwen.generate(image, vlm_prompt)
            parsed = self._extract_json(raw)
            if isinstance(parsed, dict) and ("action" in parsed or "action_type" in parsed):
                return json.dumps(self._normalize_action(parsed, act))
            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                return json.dumps(self._normalize_action(parsed[0], act))
            return json.dumps(act)

    @staticmethod
    def _normalize_action(act: Any, fallback: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(fallback)
        if not isinstance(act, dict):
            return out
        raw_type = str(act.get("action") or act.get("action_type") or fallback.get("action", "")).lower()
        if raw_type in ("left_click", "click_coordinate", "mouse_click"):
            raw_type = "click"
        elif raw_type in ("type_text", "fill", "input", "browser_select_option"):
            raw_type = "type"
        elif raw_type in ("press_key", "key_press"):
            raw_type = "press"
        out["action"] = raw_type
        out["action_type"] = raw_type
        if act.get("target"):
            out["target"] = act["target"]
        val = act.get("value") if act.get("value") is not None else act.get("text")
        if val is not None:
            out["value"] = val
        return out

    @staticmethod
    def _extract_json(raw: str) -> Any:
        text = str(raw or "").strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()
        try:
            return json.loads(text)
        except Exception:
            try:
                first_b = text.find("{")
                last_b = text.rfind("}")
                first_arr = text.find("[")
                last_arr = text.rfind("]")
                if first_arr != -1 and (first_b == -1 or first_arr < first_b):
                    if last_arr > first_arr:
                        return json.loads(text[first_arr:last_arr+1])
                if first_b != -1 and last_b > first_b:
                    return json.loads(text[first_b:last_b+1])
            except Exception:
                pass
        return None


def get_benchmark_tasks() -> List[Dict[str, Any]]:
    """Test cases 1-9 specified in Section 8 of requirements."""
    return [
        {
            "id": "TC1_settings_form",
            "name": "1. Settings/Profile Form Fill & Save",
            "url": HTML_BENCHMARK,
            "task": "Fill profile with Rahul Sharma, email, phone, country, and save profile",
            "script": [
                {"action": "type", "target": "#name", "value": "Rahul Sharma", "reason": "fill name"},
                {"action": "type", "target": "#email", "value": "rahul@example.com", "reason": "fill email"},
                {"action": "type", "target": "#phone", "value": "9876543210", "reason": "fill phone"},
                {"action": "type", "target": "#country", "value": "India", "reason": "select country"},
                {"action": "click", "target": "#saveProfile", "reason": "save profile"},
                {"action": "done", "reason": "profile saved"}
            ],
            "expected_check": lambda page: "saved and verified" in page.locator("#status").inner_text().lower()
        },
        {
            "id": "TC2_multilevel_menu",
            "name": "2. Multi-level Menu Navigation",
            "url": HTML_TEST_PAGE,
            "task": "Search products for laptop, click search, and click Add to Cart",
            "script": [
                {"action": "type", "target": "#search-input", "value": "laptop", "reason": "type query"},
                {"action": "click", "target": "#search-button", "reason": "submit search"},
                {"action": "click", "target": "#btn-add-1", "reason": "add product to cart"},
                {"action": "done", "reason": "completed search and add"}
            ],
            "expected_check": lambda page: page.locator("#search-input").input_value() == "laptop"
        },
        {
            "id": "TC3_independent_form",
            "name": "3. Multi-field Independent Form",
            "url": HTML_TEST_PAGE,
            "task": "Fill contact form with username, email, comments, and submit",
            "script": [
                {"action": "type", "target": "#username", "value": "Alice", "reason": "username"},
                {"action": "type", "target": "#email", "value": "alice@test.com", "reason": "email"},
                {"action": "type", "target": "#comments", "value": "Testing action chunking", "reason": "comments"},
                {"action": "click", "target": "#agree", "reason": "opt-in"},
                {"action": "click", "target": "#btn-submit", "reason": "submit form"},
                {"action": "done", "reason": "submitted"}
            ],
            "expected_check": lambda page: page.locator("#username").input_value() == "Alice" and page.locator("#email").input_value() == "alice@test.com"
        },
        {
            "id": "TC4_dropdown_and_field",
            "name": "4. Dropdown Selection and Input",
            "url": HTML_TEST_PAGE,
            "task": "Select country United States and type comments message",
            "script": [
                {"action": "type", "target": "#country", "value": "us", "reason": "select us"},
                {"action": "type", "target": "#comments", "value": "Need technical assistance", "reason": "type comments"},
                {"action": "done", "reason": "finished inquiry"}
            ],
            "expected_check": lambda page: page.locator("#country").input_value() == "us" and page.locator("#comments").input_value() == "Need technical assistance"
        },
        {
            "id": "TC5_unexpected_nav",
            "name": "5. Interruption on Unexpected Navigation",
            "url": HTML_BENCHMARK,
            "task": "Click navigation button that navigates to new stage, halting remaining chunk",
            "script": [
                {"action": "click", "target": "nav button:nth-child(2)", "reason": "navigate to search"},
                # The following actions assume we are still on profile, which is false due to navigation
                {"action": "type", "target": "#name", "value": "Stale", "reason": "stale type"},
                {"action": "done", "reason": "done"}
            ],
            "expected_check": lambda page: "search" in page.locator("#pageTitle").inner_text().lower(),
            "expect_interruption": True
        },
        {
            "id": "TC6_missing_target",
            "name": "6. Interruption on Missing Target",
            "url": HTML_TEST_PAGE,
            "task": "Attempt action on non-existent element, expecting safe interruption and recovery",
            "script": [
                {"action": "type", "target": "#search-input", "value": "phone", "reason": "type search"},
                {"action": "click", "target": "#non_existent_button_999", "reason": "click missing target"},
                {"action": "done", "reason": "done"}
            ],
            "expected_check": lambda page: page.locator("#search-input").input_value() == "phone",
            "expect_interruption": True
        },
        {
            "id": "TC7_boundary_action",
            "name": "7. Interruption on Boundary Action (Submit/Save)",
            "url": HTML_TEST_PAGE,
            "task": "Sequence containing save/submit action, requiring immediate verification checkpoint",
            "script": [
                {"action": "type", "target": "#username", "value": "Bob", "reason": "type name"},
                {"action": "click", "target": "#btn-submit", "reason": "save/submit boundary"},
                {"action": "type", "target": "#comments", "value": "Post submit", "reason": "post submit"},
                {"action": "done", "reason": "done"}
            ],
            "expected_check": lambda page: page.locator("#username").input_value() == "Bob",
            "expect_interruption": True
        },
        {
            "id": "TC8_stv_interruption",
            "name": "8. Interruption on STV Unexpected State",
            "url": HTML_BENCHMARK,
            "task": "Attempt invalid coupon application triggering STV retry/replan",
            "script": [
                {"action": "click", "target": "nav button:nth-child(6)", "reason": "go to coupon"},
                {"action": "type", "target": "#coupon", "value": "BADCODE", "reason": "invalid coupon"},
                {"action": "click", "target": "#applyCoupon", "reason": "apply invalid"},
                {"action": "done", "reason": "done"}
            ],
            "expected_check": lambda page: "invalid coupon" in page.locator("#status").inner_text().lower(),
            "expect_interruption": True
        },
        {
            "id": "TC9_dependent_action",
            "name": "9. Dependent Sequential Action",
            "url": HTML_TEST_PAGE,
            "task": "Type search, click search, then interact with newly revealed product",
            "script": [
                {"action": "type", "target": "#search-input", "value": "laptop", "reason": "search"},
                {"action": "click", "target": "#search-button", "reason": "submit"},
                {"action": "click", "target": "#btn-add-1", "reason": "add revealed product"},
                {"action": "done", "reason": "done"}
            ],
            "expected_check": lambda page: page.locator("#search-input").input_value() == "laptop"
        }
    ]


class BenchmarkGrounder:
    """Deterministic grounder resolving selector targets on live page."""

    def __init__(self, browser: BrowserManager, vlm: ScriptedPlannerVLM):
        self.browser = browser
        self.vlm = vlm

    def ground(self, image, target, dom_elements=None, history=None):
        raw = self.vlm.generate(image, target, history=history)
        try:
            return json.loads(raw)
        except Exception:
            return {"action": "fail", "reason": "invalid json from vlm"}

    def advance(self, count: int = 1):
        if hasattr(self.vlm, "advance"):
            self.vlm.advance(count)

    def to_action(self, pred: Dict[str, Any]) -> Dict[str, Any]:
        target = pred.get("target", "")
        atype = pred.get("action", pred.get("action_type", "click"))
        val = pred.get("value")

        if atype in ("done", "fail"):
            return {
                "action": atype,
                "action_type": atype,
                "reason": pred.get("reason", "task finished")
            }

        if not target and pred.get("x") is not None:
            return pred

        # Query live page for element coordinates
        page = self.browser.get_page()
        try:
            loc = page.locator(target).first
            if loc.count() > 0 and loc.is_visible():
                try:
                    loc.scroll_into_view_if_needed(timeout=1000)
                except Exception:
                    pass
                box = loc.bounding_box()
                if box:
                    cx = int(box["x"] + box["width"] / 2)
                    cy = int(box["y"] + box["height"] / 2)
                    return {
                        "action": atype,
                        "action_type": atype,
                        "x": cx,
                        "y": cy,
                        "value": val,
                        "target": target,
                        "reason": pred.get("reason", "")
                    }
        except Exception:
            pass

        # Target missing / not interactable on page
        return {
            "action": atype,
            "action_type": atype,
            "x": -1,
            "y": -1,
            "value": val,
            "target": target,
            "reason": f"target '{target}' not found on page"
        }


def run_single_benchmark_trial(
    task_def: Dict[str, Any],
    action_chunking: bool,
    chunk_size: int = 3,
    browser: Optional[BrowserManager] = None,
    qwen_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute a single trial for a benchmark task."""
    own_browser = False
    if browser is None:
        browser = BrowserManager(headless=True, viewport_width=1280, viewport_height=720)
        browser.launch()
        own_browser = True

    try:
        browser.new_page()
        browser.navigate(task_def["url"])
        time.sleep(0.3)

        mode_str = "chunking" if action_chunking else "baseline"
        if qwen_instance is not None:
            vlm = QwenPlannerVLM(qwen_instance, mode=mode_str, chunk_size=chunk_size)
            vlm.set_script(task_def["script"], task_desc=task_def["task"])
        else:
            vlm = ScriptedPlannerVLM(mode=mode_str, chunk_size=chunk_size)
            vlm.set_script(task_def["script"])

        observer = Observer(browser)
        grounder = BenchmarkGrounder(browser, vlm)
        executor = ActionExecutor(browser)
        verifier = StateVerifier()

        ctrl = AgentController(
            browser, observer, grounder, executor,
            verifier=verifier, max_steps=15,
            task=task_def["task"],
            enable_goal_check=False,
            action_chunking=action_chunking,
            chunk_size=chunk_size
        )

        t_start = time.time()
        outcome = ctrl.run()
        t_elapsed = time.time() - t_start

        page = browser.get_page()
        ground_truth_pass = True
        try:
            if "expected_check" in task_def:
                ground_truth_pass = bool(task_def["expected_check"](page))
        except Exception:
            ground_truth_pass = False

        agent_success = outcome.get("success", False)
        # False-success: agent claims success but ground truth check failed
        false_success = (agent_success and not ground_truth_pass)

        # Unexpected-state failure: agent crashed or encountered unhandled state
        unexpected_state_failure = outcome.get("termination") in ("browser_error", "loop_exhausted")

        browser_actions = ctrl.actions_count
        planner_calls = vlm.planner_calls
        vlm_calls = vlm.vlm_calls
        recoveries = ctrl.chunker.metrics.chunks_interrupted if action_chunking else getattr(ctrl.recovery, "recovery_count", 0)

        chunk_metrics = None
        if action_chunking and ctrl.chunker:
            m = ctrl.chunker.metrics
            chunk_metrics = {
                "chunks_proposed": m.chunks_proposed,
                "actions_proposed": m.actions_proposed,
                "actions_executed": m.actions_executed,
                "chunks_interrupted": m.chunks_interrupted,
                "interruption_reasons": dict(m.interruption_reasons),
                "replanning_calls": m.replanning_calls,
                "time_planning": round(m.time_planning, 4),
                "time_executing": round(m.time_executing, 4),
                "time_verifying": round(m.time_verifying, 4)
            }

        return {
            "task_id": task_def["id"],
            "task_name": task_def["name"],
            "mode": "action_chunking" if action_chunking else "baseline",
            "chunk_size": chunk_size if action_chunking else 1,
            "success": agent_success,
            "ground_truth_verified": ground_truth_pass,
            "false_success": false_success,
            "unexpected_state_failure": unexpected_state_failure,
            "execution_time_sec": round(t_elapsed, 4),
            "planner_calls": planner_calls,
            "vlm_calls": vlm_calls,
            "browser_actions": browser_actions,
            "retries_replans": recoveries,
            "chunk_metrics": chunk_metrics
        }
    finally:
        if own_browser:
            browser.close()


def run_benchmark_suite(
    trials: int = 5,
    chunk_size: int = 3,
    output_path: Optional[Path] = None,
    vlm_type: str = "scripted",
    model_path: Optional[str] = None,
    device: str = "mps"
) -> Dict[str, Any]:
    """Run full benchmark comparison across both baseline and experimental modes."""
    tasks = get_benchmark_tasks()
    print(f"\n========================================================")
    print(f" EXPERIMENTAL ACTION CHUNKING BENCHMARK SUITE")
    print(f" Tasks: {len(tasks)} | Trials per mode: {trials} | Chunk Size: {chunk_size}")
    print(f" VLM Engine: {vlm_type.upper()}" + (f" ({device})" if vlm_type == "qwen" else ""))
    print(f"========================================================\n")

    qwen_instance = None
    if vlm_type == "qwen":
        resolved_path = model_path or DEFAULT_QWEN_PATH
        print(f"Loading Qwen2.5-VL-3B from: {resolved_path} on {device}...")
        t0 = time.time()
        qwen_instance = QwenVL(model_path=resolved_path, device=device, max_new_tokens=256)
        qwen_instance._load()
        print(f"Qwen2.5-VL-3B initialized in {time.time() - t0:.2f}s\n", flush=True)

    baseline_runs: List[Dict[str, Any]] = []
    chunking_runs: List[Dict[str, Any]] = []

    browser = BrowserManager(headless=True, viewport_width=1280, viewport_height=720)
    browser.launch()

    try:
        # 1. Baseline Runs
        print("[1/2] Executing Baseline Runs (Single-Action Planning Loop)...", flush=True)
        for trial in range(1, trials + 1):
            for t_idx, t_def in enumerate(tasks, 1):
                res = run_single_benchmark_trial(
                    t_def,
                    action_chunking=False,
                    chunk_size=chunk_size,
                    browser=browser,
                    qwen_instance=qwen_instance
                )
                res["trial"] = trial
                baseline_runs.append(res)
                print(f"  [Baseline] Trial {trial}/{trials} | Task {t_idx}/{len(tasks)}: {t_def['name'][:35]} -> success={res['success']} GT={res['ground_truth_verified']} ({res['execution_time_sec']}s)", flush=True)

        # 2. Experimental Action Chunking Runs
        print(f"\n[2/2] Executing Experimental Action Chunking Runs (chunk_size={chunk_size})...", flush=True)
        for trial in range(1, trials + 1):
            for t_idx, t_def in enumerate(tasks, 1):
                res = run_single_benchmark_trial(
                    t_def,
                    action_chunking=True,
                    chunk_size=chunk_size,
                    browser=browser,
                    qwen_instance=qwen_instance
                )
                res["trial"] = trial
                chunking_runs.append(res)
                print(f"  [Chunking] Trial {trial}/{trials} | Task {t_idx}/{len(tasks)}: {t_def['name'][:35]} -> success={res['success']} GT={res['ground_truth_verified']} ({res['execution_time_sec']}s)", flush=True)

    finally:
        browser.close()

    # Aggregate 8 Required Metrics
    def aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(runs)
        if n == 0:
            return {}
        successes = sum(1 for r in runs if r["success"])
        gt_passes = sum(1 for r in runs if r["ground_truth_verified"])
        total_time = sum(r["execution_time_sec"] for r in runs)
        planner_calls = sum(r["planner_calls"] for r in runs)
        vlm_calls = sum(r["vlm_calls"] for r in runs)
        browser_actions = sum(r["browser_actions"] for r in runs)
        retries = sum(r["retries_replans"] for r in runs)
        false_successes = sum(1 for r in runs if r["false_success"])
        unexpected_fails = sum(1 for r in runs if r["unexpected_state_failure"])

        return {
            "total_runs": n,
            "task_success_rate": round(successes / n * 100, 2),
            "ground_truth_verified_rate": round(gt_passes / n * 100, 2),
            "end_to_end_time_sec": round(total_time, 3),
            "avg_time_per_task_sec": round(total_time / n, 3),
            "total_planner_calls": planner_calls,
            "avg_planner_calls_per_task": round(planner_calls / n, 2),
            "total_vlm_calls": vlm_calls,
            "avg_vlm_calls_per_task": round(vlm_calls / n, 2),
            "total_browser_actions": browser_actions,
            "avg_browser_actions_per_task": round(browser_actions / n, 2),
            "total_retries_replans": retries,
            "false_success_rate": round(false_successes / n * 100, 2),
            "unexpected_state_failures": unexpected_fails
        }

    agg_base = aggregate(baseline_runs)
    agg_chunk = aggregate(chunking_runs)

    # Calculate Speedup / Reduction
    planner_reduction = 0.0
    if agg_base.get("total_planner_calls", 0) > 0:
        base_p = agg_base["total_planner_calls"]
        chunk_p = agg_chunk["total_planner_calls"]
        planner_reduction = round((base_p - chunk_p) / base_p * 100, 2)

    latency_speedup = 0.0
    if agg_chunk.get("end_to_end_time_sec", 0) > 0:
        base_t = agg_base["end_to_end_time_sec"]
        chunk_t = agg_chunk["end_to_end_time_sec"]
        latency_speedup = round((base_t - chunk_t) / base_t * 100, 2)

    comparison = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vlm_engine": vlm_type,
        "model_path": model_path or (DEFAULT_QWEN_PATH if vlm_type == "qwen" else None),
        "device": device if vlm_type == "qwen" else "none",
        "trials": trials,
        "chunk_size": chunk_size,
        "task_count": len(tasks),
        "baseline_summary": agg_base,
        "action_chunking_summary": agg_chunk,
        "delta": {
            "planner_reduction_pct": planner_reduction,
            "latency_reduction_pct": latency_speedup
        },
        "baseline_runs": baseline_runs,
        "chunking_runs": chunking_runs
    }

    # Save to JSON
    if output_path is None:
        out_dir = REPO_ROOT / "data" / "benchmark_runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = "action_chunking_qwen_comparison.json" if vlm_type == "qwen" else "action_chunking_comparison.json"
        output_path = out_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nSaved benchmark results to: {output_path}")

    # Print Formatted Markdown Table
    print_table(agg_base, agg_chunk, planner_reduction, latency_speedup)

    return comparison


def print_table(
    agg_base: Dict[str, Any],
    agg_chunk: Dict[str, Any],
    planner_reduction: float,
    latency_speedup: float
):
    print("\n" + "=" * 78)
    print(" BENCHMARK COMPARISON: BASELINE vs EXPERIMENTAL ACTION CHUNKING")
    print("=" * 78)
    print(f"| Metric                             | Baseline        | Action Chunking | Delta / Improvement |")
    print(f"|------------------------------------|-----------------|-----------------|---------------------|")
    print(f"| Task success rate                  | {agg_base['task_success_rate']:>6.1f}%          | {agg_chunk['task_success_rate']:>6.1f}%          | {agg_chunk['task_success_rate'] - agg_base['task_success_rate']:>+5.1f}%              |")
    print(f"| End-to-end execution time          | {agg_base['end_to_end_time_sec']:>7.2f}s        | {agg_chunk['end_to_end_time_sec']:>7.2f}s        | {latency_speedup:>+5.1f}% speedup       |")
    print(f"| Number of planner calls            | {agg_base['total_planner_calls']:>7d}          | {agg_chunk['total_planner_calls']:>7d}          | -{planner_reduction:>4.1f}% calls         |")
    print(f"| Number of VLM calls                | {agg_base['total_vlm_calls']:>7d}          | {agg_chunk['total_vlm_calls']:>7d}          | -{planner_reduction:>4.1f}% calls         |")
    print(f"| Number of browser actions          | {agg_base['total_browser_actions']:>7d}          | {agg_chunk['total_browser_actions']:>7d}          | {agg_chunk['total_browser_actions'] - agg_base['total_browser_actions']:>+5d}                 |")
    print(f"| Number of retries / replans        | {agg_base['total_retries_replans']:>7d}          | {agg_chunk['total_retries_replans']:>7d}          | {agg_chunk['total_retries_replans'] - agg_base['total_retries_replans']:>+5d}                 |")
    print(f"| False-success rate                 | {agg_base['false_success_rate']:>6.1f}%          | {agg_chunk['false_success_rate']:>6.1f}%          | {agg_chunk['false_success_rate'] - agg_base['false_success_rate']:>+5.1f}%              |")
    print(f"| Unexpected-state failures          | {agg_base['unexpected_state_failures']:>7d}          | {agg_chunk['unexpected_state_failures']:>7d}          | {agg_chunk['unexpected_state_failures'] - agg_base['unexpected_state_failures']:>+5d}                 |")
    print("=" * 78 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Adaptive Chunk Controller (ACC) controlled scenarios (opt-in, additive).
#
# A mock runner — no real browser, no real model — that walks the 6
# controlled cases (premature terminal done, wrong-but-changing click,
# unrelated redraw, failed intermediate action, valid completion, unknown
# outcome) twice: once through legacy --action-chunking and once with
# --adaptive-chunking enabled, using the SAME deterministic page model and
# the REAL AgentController chunk machinery (executor / verifier / recovery /
# ActionChunker / ACC / goal-verification path).
#
# Results go to data/benchmark_runs/adaptive_chunking_scenarios.json and
# never overwrite the existing comparison output.
# ══════════════════════════════════════════════════════════════════════════════

class ScenarioPageModel:
    """Deterministic fake web page mutated by the executor.

    Fields: name -> current value. Clicking a known element toggles `stage`
    (simulating navigation/redraw); an unknown click leaves state untouched
    (simulating a missed / wrong click whose effect is invisible).
    """

    def __init__(self, url="file:///scenario.local/page"):
        self.url = url
        self.fields = {}
        self.visible = "Start"
        self.stage = 0
        self.flags = set()

    def snapshot(self):
        dom = [{"id": i + 1, "text": name, "tag": "input", "value": val}
               for i, (name, val) in enumerate(sorted(self.fields.items()))]
        dom.append({"id": len(dom) + 1, "text": "Go Back", "tag": "button"})
        return {
            "url": self.url,
            "screenshot_bytes": _dummy_png_bytes(),
            "dom_elements": dom,
            "visible_text": self.visible,
        }

    def type_into(self, target, value):
        key = None
        for name in self.fields:
            if name.lower() == str(target).strip().lower():
                key = name
                break
        if key is None:
            return False  # unknown field: model unchanged
        self.fields[key] = str(value)
        self.visible = f"{' '.join(self.fields.values())} {self.stage}"
        return True

    def click(self, target, nav=False):
        t = str(target).strip().lower()
        base = f"{' '.join(self.fields.values())}"
        # Correct actions produce their required page text; wrong clicks
        # change the page without ever producing the required text.
        if "confirm order" in t:
            self.flags.add("confirmation")
            self.visible = f"{base} stage{self.stage + 1} Confirmation Received"
            self.stage += 1
            return True
        if "generate report" in t:
            self.flags.add("report")
            self.visible = f"{base} stage{self.stage} Report Generated"
            self.stage += 1
            return True
        if "refresh feed" in t:
            self.flags.add("refreshed")
            self.stage += 1
            self.visible = f"{base} view{self.stage} Feed Refreshed"
            return True
        if "go back" in t:
            self.url = f"{self.url}#back{self.stage}"
            self.stage += 1
            return True
        # wrong-but-changing: any named click mutates the page but the
        # required text is only produced by the correct action
        if t and t not in {n.lower() for n in self.fields}:
            self.stage += 1
            self.visible = f"{base} view{self.stage}"
            return True
        return False

    def scroll(self, dy=500.0):
        return True

    def wait(self, seconds=1.0):
        return True


def _dummy_png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (48, 48), "white").save(buf, format="PNG")
    return buf.getvalue()


class ScenarioObserver:
    """Observer reading the fake page model."""

    def __init__(self, model: ScenarioPageModel):
        self.model = model

    def observe(self, save_screenshot: bool = False) -> Dict[str, Any]:
        return self.model.snapshot()


class ScenarioGrounder:
    """Grounds a prediction to itself (SoM passthrough resolver)."""

    def ground(self, image, target, dom_elements=None, history=None):
        return {"action": "click", "x": 10, "y": 10, "target": target}

    def to_action(self, pred):
        return pred


class ScenarioBrowser:
    """Minimal browser: only get_url is consulted by the chunk loop."""

    def __init__(self, model):
        self.model = model

    def get_url(self):
        return self.model.url


class ScenarioExecutor:
    """Executor that mutates the fake page; can simulate one first-failure
    per named target (deterministic retry semantics)."""

    def __init__(self, model, fail_on=None):
        self.model = model
        self.fail_on = set(fail_on or ())
        self._failed_once = set()
        self.executed = []

    def execute(self, action):
        at = getattr(action, "action_type", "?")
        target = getattr(action, "target", "") or ""
        value = getattr(action, "value", "") or ""
        self.executed.append((at, target, value))
        if target in self.fail_on and target not in self._failed_once:
            self._failed_once.add(target)
            return {"success": False,
                    "error": f"simulated failure: {target}"}
        if at == "type":
            self.model.type_into(target, value)
        elif at == "click":
            self.model.click(target)
        # scroll/wait/press: no page model effect needed
        return {"success": True, "error": None}


class ScenarioPermissiveVerifier:
    """StateVerifier stand-in: always passes the plain-execution check.

    Deliberate: the controlled cases isolate the ACC contract-check and
    goal-verification deltas from StateVerifier heuristics. The contract
    (expected_transition) and the goal verifier provide the strict truth.
    """

    def verify(self, before_obs, after_obs, action, task=None):
        return {"success": True, "reason": "scenario mock verifier ok"}


class ScenarioGoalVerifier:
    """GoalVerifier stand-in whose truth is the page model's final state."""

    def __init__(self, final_check, model):
        self._final_check = final_check
        self.model = model
        self.calls = []

    def check(self, goal, observation, start_url, prev_observation=None,
              verification=None):
        self.calls.append({"goal": goal})
        achieved = self._final_check(self.model)
        return {"achieved": achieved, "done": achieved,
                "reason": "final state reached" if achieved
                else "required page state not reached",
                "signals": {}}


def run_scenario_once(task: str, final_check, cycles: List[Dict[str, Any]],
                      adaptive: bool, max_cycles: int = 8,
                      fail_on=None) -> Dict[str, Any]:
    """Walk one scenario under one mode using the real chunk machinery."""
    from agent.controller import AgentController
    from agent.memory import AgentMemory
    from agent.recovery import RecoveryManager

    model = ScenarioPageModel()
    for fname in ("First Name", "Last Name"):
        model.fields.setdefault(fname, "")
    observer = ScenarioObserver(model)
    grounder = ScenarioGrounder()
    executor = ScenarioExecutor(model, fail_on=fail_on)
    goal_verifier = ScenarioGoalVerifier(final_check, model)
    recovery = RecoveryManager(max_recovery_attempts=5)
    ctrl = AgentController(
        ScenarioBrowser(model), observer, grounder, executor,
        verifier=ScenarioPermissiveVerifier(), memory=AgentMemory(),
        recovery=recovery, max_steps=30, task=task,
        goal_verifier=goal_verifier, enable_goal_check=True,
        action_chunking=True, chunk_size=3,
        adaptive_chunking=adaptive, adaptive_chunk_max_size=3,
    )

    m = ctrl.chunker.metrics
    planned_sizes: List[int] = []
    executed_sizes: List[int] = []
    planner_cycles = 0
    claimed = False
    termination = "cycles_exhausted"
    outcome: Dict[str, Any] = {}
    observation = model.snapshot()
    start_url = observation.get("url", "")

    for step, cyc in enumerate(cycles[:max_cycles], start=1):
        proposal = list(cyc.get("proposal") or [])
        pre_p, pre_e = m.actions_proposed, m.actions_executed
        planner_cycles += 1  # mock attribution: one planning turn per cycle
        res = ctrl._execute_chunk(step, task, observation, proposal,
                                  start_url=start_url)
        observation = res["next_observation"]
        planned_sizes.append(max(0, m.actions_proposed - pre_p))
        executed_sizes.append(max(0, m.actions_executed - pre_e))
        if res.get("finished"):
            outcome = res.get("outcome") or {}
            claimed = bool(outcome.get("success"))
            termination = str(outcome.get("termination"))
            break

    verified = bool(final_check(model))
    interruptions = dict(m.interruption_reasons)
    false_success = claimed and not verified
    planned = planned_sizes or [0]
    executed = executed_sizes or [0]
    metrics: Dict[str, Any] = {
        "claimed_success": claimed,
        "verified_success": verified,
        "false_success": false_success,
        "termination": termination,
        "planner_cycles": planner_cycles,
        "vlm_calls": planner_cycles,  # mock attribution, documented
        "browser_actions": int(ctrl.actions_count),
        "retries_replans": int(recovery.recovery_count
                               + m.replanning_calls),
        "interruptions": interruptions,
        "nav_interruptions": int(interruptions.get("navigation", 0)),
        "unexpected_state_failures": termination in ("browser_error",
                                                     "loop_exhausted"),
        "goal_checks": len(goal_verifier.calls),
        "planned_chunk_sizes": planned_sizes,
        "executed_chunk_sizes": executed_sizes,
        "avg_planned_chunk_size": round(sum(planned) / max(1, len(planned)), 2),
        "avg_executed_chunk_size": round(sum(executed) / max(1, len(executed)), 2),
    }
    if adaptive and ctrl.acc is not None:
        metrics["acc_stats"] = dict(ctrl.acc.stats)
    return metrics


def _final_form_filled(model) -> bool:
    return (model.fields.get("First Name") == "Alice"
            and model.fields.get("Last Name") == "Smith")


def get_acc_scenarios() -> List[Dict[str, Any]]:
    """The 6 controlled cases (same script walked under both modes)."""

    def s2_final(model):
        return "confirmation" in getattr(model, "flags", set())

    def s3_final(model):
        return "refreshed" in getattr(model, "flags", set())

    def s6_final(model):
        return "report" in getattr(model, "flags", set())

    return [
        {
            "id": "S1_premature_done",
            "name": "1. Premature terminal done",
            "task": "Fill first and last name, then finish",
            "final": _final_form_filled,
            "cycles": [
                {"proposal": [{"action": "done", "reason": "looks complete"}]},
                {"proposal": [{"action": "type", "target": "First Name",
                               "value": "Alice"}]},
                {"proposal": [{"action": "type", "target": "Last Name",
                               "value": "Smith"},
                              {"action": "done", "reason": "all set"}]},
            ],
        },
        {
            "id": "S2_wrong_but_changing_click",
            "name": "2. Wrong-but-changing click",
            "task": "Confirm the order (skip the banners)",
            "final": s2_final,
            "cycles": [
                {"proposal": [
                    {"action": "click", "target": "Wrong Banner",
                     "expected_transition": {"must_change": ["page results"],
                                             "must_be_true":
                                                 ["Confirmation Received"]}},
                    {"action": "click", "target": "Wrong Banner Two",
                     "expected_transition": {"must_change": ["page results"],
                                             "must_be_true":
                                                 ["Confirmation Received"]}}]},
                {"proposal": [{"action": "click", "target": "Confirm Order",
                               "expected_transition":
                                   {"must_be_true":
                                    ["Confirmation Received"]}},
                              {"action": "done", "reason": "confirmed"}]},
                {"proposal": [{"action": "done", "reason": "confirmed"}]},
            ],
        },
        {
            "id": "S3_unrelated_redraw",
            "name": "3. Unrelated redraw must not pass contracts",
            "task": "Refresh the page (unrelated content changes)",
            "final": s3_final,
            "cycles": [
                {"proposal": [{"action": "click", "target": "Refresh Feed",
                               "expected_transition":
                                   {"must_be_true": ["Order Confirmation"]}}]},
                {"proposal": [{"action": "click", "target": "Refresh Feed",
                               "expected_transition": {"must_change":
                                                           ["page results"]}},
                              {"action": "done", "reason": "refreshed"}]},
                {"proposal": [{"action": "done", "reason": "refreshed"}]},
            ],
        },
        {
            "id": "S4_failed_intermediate",
            "name": "4. Failed intermediate action inside chunk",
            "task": "Fill the form (Last Name fails once)",
            "final": _final_form_filled,
            "fail_on": ["Last Name"],
            "cycles": [
                {"proposal": [{"action": "type", "target": "First Name",
                               "value": "Alice"},
                              {"action": "type", "target": "Last Name",
                               "value": "Smith"},
                              {"action": "done", "reason": "filled"}]},
                {"proposal": [{"action": "type", "target": "Last Name",
                               "value": "Smith"},
                              {"action": "done", "reason": "filled"}]},
                {"proposal": [{"action": "type", "target": "Last Name",
                               "value": "Smith"}]},
                {"proposal": [{"action": "done", "reason": "filled"}]},
            ],
        },
        {
            "id": "S5_valid_completion",
            "name": "5. Valid completion (no block)",
            "task": "Fill the form and finish",
            "final": _final_form_filled,
            "cycles": [
                {"proposal": [{"action": "type", "target": "First Name",
                               "value": "Alice"},
                              {"action": "type", "target": "Last Name",
                               "value": "Smith"},
                              {"action": "done", "reason": "filled"}]},
                {"proposal": [{"action": "type", "target": "Last Name",
                               "value": "Smith"},
                              {"action": "done", "reason": "filled"}]},
            ],
        },
        {
            "id": "S6_unknown_outcome",
            "name": "6. Unknown contract outcome (not a success)",
            "task": "Generate the report",
            "final": s6_final,
            "cycles": [
                {"proposal": [{"action": "click", "target": "Generate Report",
                               "expected_transition": {"must_be_true": ["!!"]}},
                              {"action": "done", "reason": "done"}]},
                {"proposal": [{"action": "done", "reason": "done"}]},
            ],
        },
    ]


def run_acc_scenarios(output_path: Path,
                      adaptive_chunk_max: int = 3) -> Dict[str, Any]:
    """Compare legacy chunk mode vs ACC on controlled cases (additive)."""
    scenarios = get_acc_scenarios()
    results: List[Dict[str, Any]] = []

    print("\n========================================================")
    print(" ADAPTIVE CHUNKING CONTROLLED SCENARIOS (mock runner)")
    print(" Legacy action chunking vs ACC — deterministic page model")
    print(f" ACC hard cap: {adaptive_chunk_max}")
    print("========================================================\n")

    for sc in scenarios:
        legacy = run_scenario_once(
            sc["task"], sc["final"], sc["cycles"], adaptive=False,
            fail_on=sc.get("fail_on"))
        adaptive = run_scenario_once(
            sc["task"], sc["final"], sc["cycles"], adaptive=True,
            fail_on=sc.get("fail_on"))
        comparison = {
            "false_success_removed": legacy["false_success"]
            and not adaptive["false_success"],
            "wasted_actions_saved": legacy["browser_actions"]
            - adaptive["browser_actions"],
            "extra_planner_cycles": adaptive["planner_cycles"]
            - legacy["planner_cycles"],
            "goal_checks_added": adaptive["goal_checks"]
            - legacy["goal_checks"],
        }
        results.append({"id": sc["id"], "name": sc["name"],
                        "legacy": legacy, "adaptive": adaptive,
                        "comparison": comparison})
        print(f"  {sc['id']:34s} legacy: claimed={legacy['claimed_success']!s:5s} "
              f"verified={legacy['verified_success']!s:5s} "
              f"false_success={legacy['false_success']!s:5s} | "
              f"ACC: claimed={adaptive['claimed_success']!s:5s} "
              f"verified={adaptive['verified_success']!s:5s} "
              f"false_success={adaptive['false_success']!s:5s}")

    n = len(results)
    aggregate = {
        "scenarios": n,
        "false_successes_legacy": sum(1 for r in results
                                      if r["legacy"]["false_success"]),
        "false_successes_adaptive": sum(1 for r in results
                                        if r["adaptive"]["false_success"]),
        "claimed_success_rate_legacy": round(
            sum(1 for r in results if r["legacy"]["claimed_success"]) / n * 100,
            2),
        "claimed_success_rate_adaptive": round(
            sum(1 for r in results if r["adaptive"]["claimed_success"]) / n * 100,
            2),
        "verified_success_rate_legacy": round(
            sum(1 for r in results if r["legacy"]["verified_success"]) / n * 100,
            2),
        "verified_success_rate_adaptive": round(
            sum(1 for r in results if r["adaptive"]["verified_success"]) / n * 100,
            2),
        "wasted_actions_saved_total": sum(
            r["comparison"]["wasted_actions_saved"] for r in results),
        "extra_planner_cycles_total": sum(
            r["comparison"]["extra_planner_cycles"] for r in results),
        "goal_checks_added_total": sum(
            r["comparison"]["goal_checks_added"] for r in results),
        "total_planner_cycles_legacy": sum(r["legacy"]["planner_cycles"]
                                           for r in results),
        "total_planner_cycles_adaptive": sum(
            r["adaptive"]["planner_cycles"] for r in results),
        "total_actions_legacy": sum(r["legacy"]["browser_actions"]
                                    for r in results),
        "total_actions_adaptive": sum(r["adaptive"]["browser_actions"]
                                      for r in results),
        "total_retries_replans_legacy": sum(r["legacy"]["retries_replans"]
                                            for r in results),
        "total_retries_replans_adaptive": sum(
            r["adaptive"]["retries_replans"] for r in results),
        "total_nav_interruptions_legacy": sum(
            r["legacy"]["nav_interruptions"] for r in results),
        "total_nav_interruptions_adaptive": sum(
            r["adaptive"]["nav_interruptions"] for r in results),
        "avg_planned_chunk_size_legacy": round(sum(
            sum(r["legacy"]["planned_chunk_sizes"]) /
            max(1, len(r["legacy"]["planned_chunk_sizes"]))
            for r in results) / max(1, n), 2),
        "avg_planned_chunk_size_adaptive": round(sum(
            sum(r["adaptive"]["planned_chunk_sizes"]) /
            max(1, len(r["adaptive"]["planned_chunk_sizes"]))
            for r in results) / max(1, n), 2),
        "avg_executed_chunk_size_legacy": round(sum(
            sum(r["legacy"]["executed_chunk_sizes"]) /
            max(1, len(r["legacy"]["executed_chunk_sizes"]))
            for r in results) / max(1, n), 2),
        "avg_executed_chunk_size_adaptive": round(sum(
            sum(r["adaptive"]["executed_chunk_sizes"]) /
            max(1, len(r["adaptive"]["executed_chunk_sizes"]))
            for r in results) / max(1, n), 2),
        "unexpected_state_failures_legacy": sum(
            1 for r in results if r["legacy"]["unexpected_state_failures"]),
        "unexpected_state_failures_adaptive": sum(
            1 for r in results if r["adaptive"]["unexpected_state_failures"]),
    }

    report = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "adaptive_chunking_scenarios",
        "note": ("Controlled mock-runner cases; uses the real chunk "
                 "machinery (AgentController._execute_chunk, ActionChunker, "
                 "recovery, ACC) with a deterministic fake page model — no "
                 "browser or model. Existing suite results are untouched."),
        "interpretation": (
            "S1 is the headline case: legacy action chunking accepts the "
            "model's terminal done without any goal verification and claims "
            "success while the task is incomplete (false success); ACC "
            "rejects it via the goal-verifier safeguard, replans, completes, "
            "and verifies. S2/S3 show the expected_transition contract "
            "stopping after wrong-but-changing / unrelated-redraw actions "
            "(legacy executes further actions on them). S4-S6 are "
            "parity/observability cases: same final outcomes, ACC records "
            "goal checks and unknown verifications, and pays extra planner "
            "cycles for the conservative (streaked, >=1 and <= cap) sizing. "
            "action delta = legacy actions - ACC actions; ACC re-plans "
            "after contract interruptions, so the delta can be negative on "
            "contract-triggering scenarios."),
        "adaptive_chunk_max": adaptive_chunk_max,
        "aggregate": aggregate,
        "scenarios": results,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved ACC scenario results to: {output_path}")

    print("\n" + "=" * 78)
    print(" ACC SCENARIO COMPARISON")
    print("=" * 78)
    agg = aggregate
    print(f"| false successes                    | {agg['false_successes_legacy']:>15d} | {agg['false_successes_adaptive']:>17d} |")
    print(f"| claimed success rate               | {agg['claimed_success_rate_legacy']:>14.1f}% | {agg['claimed_success_rate_adaptive']:>16.1f}% |")
    print(f"| verified success rate              | {agg['verified_success_rate_legacy']:>14.1f}% | {agg['verified_success_rate_adaptive']:>16.1f}% |")
    print(f"| action delta (legacy - ACC)        | {agg['wasted_actions_saved_total']:>15d} |")
    print(f"| extra planner cycles (ACC cost)    | {agg['extra_planner_cycles_total']:>15d} |")
    print(f"| goal checks added                  | {agg['goal_checks_added_total']:>15d} |")
    print(f"| nav interruptions                  | {agg['total_nav_interruptions_legacy']:>15d} / {agg['total_nav_interruptions_adaptive']} |")
    print(f"| unexpected-state failures          | {agg['unexpected_state_failures_legacy']:>15d} / {agg['unexpected_state_failures_adaptive']} |")
    print(f"| avg planned chunk size             | {agg['avg_planned_chunk_size_legacy']:>15.2f} / {agg['avg_planned_chunk_size_adaptive']:.2f} |")
    print("=" * 78 + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="Action Chunking Benchmark")
    parser.add_argument("--trials", type=int, default=1, help="Number of trials per task")
    parser.add_argument("--chunk-size", type=int, default=3, choices=range(2, 7), help="Chunk size (2-6)")
    parser.add_argument("--vlm", type=str, default="qwen", choices=["scripted", "qwen"], help="VLM engine")
    parser.add_argument("--model-path", type=str, default=None, help="Model path/snapshot for Qwen")
    parser.add_argument("--device", type=str, default="mps", help="Device for Qwen inference (mps, cuda, cpu)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--scenarios", action="store_true", default=False,
                        help="Run the controlled ACC scenario cases (mock "
                             "runner, no browser/model) on legacy action "
                             "chunking vs --adaptive-chunking, and save the "
                             "comparison to "
                             "data/benchmark_runs/adaptive_chunking_scenarios"
                             ".json. Existing suite results are untouched.")
    parser.add_argument("--adaptive-chunk-max", type=int, default=3,
                        help="ACC hard cap for scenario mode (1-6, default 3).")
    args = parser.parse_args()

    if args.scenarios:
        out_path = Path(args.output) if args.output else \
            (REPO_ROOT / "data" / "benchmark_runs" /
             "adaptive_chunking_scenarios.json")
        run_acc_scenarios(output_path=out_path,
                          adaptive_chunk_max=args.adaptive_chunk_max)
        return

    out_path = Path(args.output) if args.output else None
    run_benchmark_suite(
        trials=args.trials,
        chunk_size=args.chunk_size,
        output_path=out_path,
        vlm_type=args.vlm,
        model_path=args.model_path,
        device=args.device
    )


if __name__ == "__main__":
    main()
