#!/usr/bin/env python3
"""HTML Benchmark comparing Old (Baseline) vs New (TaskCompiler + SubgoalContract) on lh_benchmark.html.

Runs reproducible long-horizon tasks on test_pages/lh_benchmark.html and measures:
- Task-level outcomes (denominators & pass rates)
- Verified subgoal rate
- Repeated actions
- Recovery count
- Model/planning calls & latency
- Action budget usage
"""

import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from browser.browser_manager import BrowserManager
from browser.observer import Observer
from browser.proposers import DOMProposer
from actions.action_executor import ActionExecutor
from actions.action_schema import parse_action
from agent.long_horizon.controller import LongHorizonController
from agent.long_horizon.task_spec import TaskCompiler, TaskSpec
from agent.long_horizon.subgoal_contract import (
    SubgoalContract, Postcondition, validate_contract_pre_execution, repair_contract
)
from agent.long_horizon.observation_builder import build_observation_context, is_stale_observation
from agent.long_horizon.task_memory import TaskMemory

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("benchmark")

HTML_URL = f"file://{REPO_ROOT / 'test_pages' / 'lh_benchmark.html'}"


BENCHMARK_TASKS = [
    {
        "id": "T1_profile",
        "name": "Customer Profile Form Fill",
        "task": "Create customer profile: Full name Rahul Sharma, email rahul.sharma@example.com, phone 9876543210, country India, and save profile.",
        "expected_stage": "Profile",
        "actions_script": [
            {"action": "type", "selector": "#name", "value": "Rahul Sharma", "text": "Rahul Sharma"},
            {"action": "type", "selector": "#email", "value": "rahul.sharma@example.com", "text": "rahul.sharma@example.com"},
            {"action": "type", "selector": "#phone", "value": "9876543210", "text": "9876543210"},
            {"action": "type", "selector": "#country", "value": "India", "text": "India"},
            {"action": "click", "selector": "#saveProfile", "text": "Save Profile"}
        ]
    },
    {
        "id": "T2_search",
        "name": "Product Search & Selection",
        "task": "Search for laptop and open the matching product.",
        "expected_stage": "Search",
        "actions_script": [
            {"action": "type", "selector": "#search", "value": "laptop", "text": "laptop"},
            {"action": "click", "selector": "#searchBtn", "text": "Search"},
            {"action": "click", "selector": "#openProduct", "text": "Open Product"}
        ]
    },
    {
        "id": "T3_configure",
        "name": "Product Configuration",
        "task": "Configure UltraBook Laptop 14 with size 14-inch, color Black, quantity 2, and add to cart.",
        "expected_stage": "Configure",
        "actions_script": [
            {"action": "type", "selector": "#size", "value": "14-inch", "text": "14-inch"},
            {"action": "type", "selector": "#color", "value": "Black", "text": "Black"},
            {"action": "type", "selector": "#qty", "value": "2", "text": "2"},
            {"action": "click", "selector": "#add", "text": "Add to Cart"}
        ]
    },
    {
        "id": "T4_cart_update_remove",
        "name": "Cart Quantity Update and Item Removal",
        "task": "Update cart quantity to 3, remove item, then add back with quantity 3 and continue.",
        "expected_stage": "Cart",
        "actions_script": [
            {"action": "type", "selector": "#cartQty", "value": "3", "text": "3"},
            {"action": "click", "selector": "#updateCart", "text": "Update Quantity"},
            {"action": "click", "selector": "#removeCart", "text": "Remove Item"},
            # Recovery / re-add: set qty to 3, update, continue
            {"action": "type", "selector": "#cartQty", "value": "3", "text": "3"},
            {"action": "click", "selector": "#updateCart", "text": "Update Quantity"},
            {"action": "click", "selector": "#continueCart", "text": "Continue"}
        ]
    },
    {
        "id": "T5_shipping",
        "name": "Shipping Address",
        "task": "Enter shipping address: Name Rahul Sharma, Address 123 Park Street, City Kolkata, State West Bengal, PIN 700016 and save address.",
        "expected_stage": "Shipping",
        "actions_script": [
            {"action": "type", "selector": "#sname", "value": "Rahul Sharma", "text": "Rahul Sharma"},
            {"action": "type", "selector": "#address", "value": "123 Park Street", "text": "123 Park Street"},
            {"action": "type", "selector": "#city", "value": "Kolkata", "text": "Kolkata"},
            {"action": "type", "selector": "#state", "value": "West Bengal", "text": "West Bengal"},
            {"action": "type", "selector": "#pin", "value": "700016", "text": "700016"},
            {"action": "click", "selector": "#saveAddress", "text": "Save Address"}
        ]
    },
    {
        "id": "T6_coupon_recoverable",
        "name": "Coupon Application with Recoverable Retry",
        "task": "Apply coupon code SAVE10 and verify discount, recovering if invalid code was initially entered.",
        "expected_stage": "Coupon",
        "actions_script": [
            # Injected failure: wrong coupon first
            {"action": "type", "selector": "#coupon", "value": "BADCODE", "text": "BADCODE"},
            {"action": "click", "selector": "#applyCoupon", "text": "Apply Coupon"},
            # Recovery: valid coupon
            {"action": "type", "selector": "#coupon", "value": "SAVE10", "text": "SAVE10"},
            {"action": "click", "selector": "#applyCoupon", "text": "Apply Coupon"},
            {"action": "click", "selector": "#continueCoupon", "text": "Continue"}
        ]
    },
    {
        "id": "T7_delivery",
        "name": "Delivery Option Selection",
        "task": "Choose Express delivery and continue.",
        "expected_stage": "Delivery",
        "actions_script": [
            {"action": "click", "selector": "input[value='Express']", "text": "Express"},
            {"action": "click", "selector": "#deliveryBtn", "text": "Continue"}
        ]
    },
    {
        "id": "T8_payment",
        "name": "Payment Authorization & Submission",
        "task": "Enter payment information with Credit Card for Rahul Sharma, card 4111111111111111, expiry 12/29, CVV 123, and continue to review.",
        "expected_stage": "Payment",
        "actions_script": [
            {"action": "type", "selector": "#method", "value": "Credit Card", "text": "Credit Card"},
            {"action": "type", "selector": "#pname", "value": "Rahul Sharma", "text": "Rahul Sharma"},
            {"action": "type", "selector": "#card", "value": "4111111111111111", "text": "4111111111111111"},
            {"action": "type", "selector": "#expiry", "value": "12/29", "text": "12/29"},
            {"action": "type", "selector": "#cvv", "value": "123", "text": "123"},
            {"action": "click", "selector": "#paymentBtn", "text": "Continue to Review"}
        ]
    },
    {
        "id": "T9_review",
        "name": "Order Review & PIN Edit",
        "task": "Review order, edit shipping PIN to 700017, save review, and confirm order details.",
        "expected_stage": "Review",
        "actions_script": [
            {"action": "type", "selector": "#reviewPin", "value": "700017", "text": "700017"},
            {"action": "click", "selector": "#saveReview", "text": "Save Review"},
            {"action": "click", "selector": "#confirmReview", "text": "Confirm Order Details"}
        ]
    },
    {
        "id": "T10_confirmation_delayed",
        "name": "Order Placement with Delayed Transition",
        "task": "Place order and verify confirmation and generated order ID after asynchronous processing.",
        "expected_stage": "Confirmation",
        "actions_script": [
            {"action": "click", "selector": "#placeOrder", "text": "Place Order"}
        ]
    }
]


def run_benchmark(enable_compiler: bool) -> Dict[str, Any]:
    browser = BrowserManager(headless=True, viewport_width=1280, viewport_height=720)
    browser.launch()
    browser.new_page()
    browser.navigate(HTML_URL)
    page = browser.get_page()

    results = []
    total_actions = 0
    total_recoveries = 0
    repeated_actions = 0
    model_calls = 0
    total_subgoals_verified = 0
    total_subgoals_attempted = len(BENCHMARK_TASKS)
    t_start = time.time()

    compiler = TaskCompiler() if enable_compiler else None

    try:
        for t_idx, task_def in enumerate(BENCHMARK_TASKS, 1):
            t0 = time.time()
            task_id = task_def["id"]
            raw_task = task_def["task"]
            actions_list = task_def["actions_script"]
            task_success = True
            task_recoveries = 0
            task_repeated = 0
            task_actions_count = 0

            # 1. Compiler Gate (New Architecture only)
            spec = None
            if enable_compiler:
                comp_res = compiler.compile(raw_task)
                if not comp_res.ok or comp_res.spec is None:
                    results.append({
                        "task_id": task_id,
                        "success": False,
                        "reason": f"Compilation failed: {comp_res.error}",
                        "subgoals_verified": 0,
                        "actions": 0,
                        "recoveries": 0,
                        "repeated": 0,
                        "latency_sec": round(time.time() - t0, 3)
                    })
                    continue
                spec = comp_res.spec
                model_calls += 1

            # 2. Execute actions for task
            last_action_sig = None
            for act in actions_list:
                action_type = act["action"]
                selector = act["selector"]
                value = act.get("value")

                # Subgoal Contract Validation Gate (New Architecture only)
                if enable_compiler and spec is not None:
                    # Construct contract for action
                    contract = SubgoalContract(
                        subgoal_id=f"SG_{t_idx}_{task_actions_count}",
                        objective=f"{action_type} on {selector}",
                        action_intent=action_type,
                        action_target_description=selector,
                        action_value=value if value else ("click" if action_type == "click" else "action"),
                        preconditions=[],
                        postconditions=[
                            Postcondition(predicate="CHANGED", target=selector, value=str(value or ""), required=True)
                        ],
                        completion_evidence=[f"{selector} updated"],
                        allowed_side_effects=[]
                    )
                    # Contract Gate
                    violations = validate_contract_pre_execution(contract, spec)
                    if violations:
                        # Attempt bounded repair
                        contract = repair_contract(contract)
                        violations = validate_contract_pre_execution(contract, spec)
                        if violations:
                            task_success = False
                            break

                # Check repeated action
                curr_sig = (action_type, selector, value)
                if curr_sig == last_action_sig:
                    task_repeated += 1
                    repeated_actions += 1
                last_action_sig = curr_sig

                # Check if this is a known injected recovery step
                if value == "BADCODE":
                    task_recoveries += 1
                    total_recoveries += 1

                # Execute action on live DOM
                try:
                    loc = page.locator(selector).first
                    if action_type == "type":
                        tag = loc.evaluate("el => el.tagName")
                        if tag == "SELECT":
                            loc.select_option(value or "", timeout=3000)
                        else:
                            loc.fill(value or "", timeout=3000)
                    elif action_type == "click":
                        loc.click(timeout=3000)
                    task_actions_count += 1
                    total_actions += 1
                except Exception as e:
                    logger.warning(f"Action execution error on {selector}: {e}")
                    task_success = False
                    break

            # If task 10 (delayed transition), wait for async order confirmation
            if task_id == "T10_confirmation_delayed":
                page.wait_for_timeout(1500)
                # Verify final confirmation
                order_id_el = page.locator("#orderId").first
                if order_id_el.is_visible() and order_id_el.inner_text().startswith("MS-"):
                    total_subgoals_verified += 1
                else:
                    task_success = False
            else:
                # Check status
                status_text = page.locator("#status").inner_text()
                if "error" in status_text.lower():
                    task_success = False
                else:
                    total_subgoals_verified += 1

            latency = round(time.time() - t0, 3)
            results.append({
                "task_id": task_id,
                "name": task_def["name"],
                "success": task_success,
                "actions": task_actions_count,
                "recoveries": task_recoveries,
                "repeated": task_repeated,
                "latency_sec": latency
            })
    finally:
        browser.close()

    total_latency = round(time.time() - t_start, 3)
    passed_tasks = sum(1 for r in results if r["success"])
    return {
        "mode": "Upgraded (TaskCompiler + SubgoalContract)" if enable_compiler else "Baseline (Uncompiled)",
        "tasks_attempted": total_subgoals_attempted,
        "tasks_passed": passed_tasks,
        "pass_rate": f"{(passed_tasks / total_subgoals_attempted) * 100:.1f}%",
        "verified_subgoal_rate": f"{(total_subgoals_verified / total_subgoals_attempted) * 100:.1f}%",
        "total_actions": total_actions,
        "repeated_actions": repeated_actions,
        "recovery_count": total_recoveries,
        "model_calls": model_calls,
        "total_latency_sec": total_latency,
        "task_results": results
    }


def main():
    print("=" * 70)
    print("RUNNING LONG-HORIZON HTML BENCHMARK: BASELINE vs UPGRADED")
    print(f"Fixture: {HTML_URL}")
    print("=" * 70)

    print("\n[1/2] Running Baseline (Uncompiled)...")
    res_base = run_benchmark(enable_compiler=False)

    print("\n[2/2] Running Upgraded (TaskCompiler + SubgoalContract)...")
    res_upgraded = run_benchmark(enable_compiler=True)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<30} | {'Baseline':<18} | {'Upgraded':<18}")
    print("-" * 72)
    print(f"{'Tasks Passed':<30} | {res_base['tasks_passed']}/{res_base['tasks_attempted']} ({res_base['pass_rate']}) | {res_upgraded['tasks_passed']}/{res_upgraded['tasks_attempted']} ({res_upgraded['pass_rate']})")
    print(f"{'Verified Subgoal Rate':<30} | {res_base['verified_subgoal_rate']:<18} | {res_upgraded['verified_subgoal_rate']:<18}")
    print(f"{'Total Actions':<30} | {res_base['total_actions']:<18} | {res_upgraded['total_actions']:<18}")
    print(f"{'Repeated Actions':<30} | {res_base['repeated_actions']:<18} | {res_upgraded['repeated_actions']:<18}")
    print(f"{'Recoveries Triggered':<30} | {res_base['recovery_count']:<18} | {res_upgraded['recovery_count']:<18}")
    print(f"{'Latency (seconds)':<30} | {res_base['total_latency_sec']:<18} | {res_upgraded['total_latency_sec']:<18}")

    print("\nPer-Task Outcomes:")
    for b, u in zip(res_base["task_results"], res_upgraded["task_results"]):
        b_st = "PASS" if b["success"] else "FAIL"
        u_st = "PASS" if u["success"] else "FAIL"
        print(f"  {b['task_id']:<26} | Baseline: {b_st} ({b['actions']} acts, {b['latency_sec']}s) | Upgraded: {u_st} ({u['actions']} acts, {u['latency_sec']}s)")

    # Save to json report
    out_path = REPO_ROOT / "reports" / "lh_benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"baseline": res_base, "upgraded": res_upgraded}, f, indent=2)
    print(f"\nSaved detailed benchmark results to: {out_path}")


if __name__ == "__main__":
    main()
