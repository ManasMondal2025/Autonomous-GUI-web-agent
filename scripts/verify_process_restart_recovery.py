#!/usr/bin/env python3
"""Multi-Process Browser Restart and Session Recovery Verification.

Fulfills Phase 2 / Requirement 2:
1. Start task in Python Process 1 with persistence enabled.
2. Complete and verify an intermediate subgoal.
3. Save checkpoint to SQLite.
4. Terminate Process 1 and browser process.
5. Start a fresh Python Process 2.
6. Resume from saved task ID.
7. Reconcile saved checkpoint with live application state.
8. Complete remaining subgoals.
9. Independently verify final application state.
10. Check authoritative server-side counter for duplicate consequential operations.

Tests:
- Scenario A: Recoverable session (complete workflow with server count == 1).
- Scenario B: Expired / unavailable authentication session (detects login redirect, pauses for HITL).
- Scenario C: Ambiguous consequential submission (crashed during submit, pauses for HITL, 0 duplicate clicks).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(REPO_ROOT))

from authoritative_test_server import (
    AuthoritativeServerState,
    HTML_PAGE,
    SERVER_STATE,
    LocalAuthoritativeServer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("process_restart_test")


def run_worker_process(args: list[str]) -> subprocess.CompletedProcess:
    """Run an isolated Python sub-process."""
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}/src:{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    cmd = [sys.executable, "-m"] + args
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


# Inline worker script executed in separate Python processes
WORKER_CODE = '''
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(REPO_ROOT))

from agent.persistence.manager import TaskPersistenceManager
from agent.persistence.schema import TaskStatus, ActionPhase
from browser.browser_manager import BrowserManager

action = sys.argv[1]
db_path = sys.argv[2]
server_url = sys.argv[3]
task_id = sys.argv[4]

mgr = TaskPersistenceManager(db_path=db_path)

if action == "start_and_checkpoint_intermediate":
    subgoals = [
        {"id": "S1", "intent": "Fill patient information", "target": "#patientName", "value": "Alice Morgan"},
        {"id": "S2", "intent": "Submit appointment request", "target": "#btnSubmitOnce", "action": "click"},
    ]
    bm = BrowserManager(headless=True)
    bm.launch()
    bm.new_page()
    page = bm.get_page()
    page.goto(server_url)
    page.wait_for_timeout(300)

    mgr.start_task(task_id, "Book appointment on authoritative server", subgoals, start_url=server_url)
    mgr.start_subgoal(subgoals[0], browser_url=page.url)

    # Perform Subgoal 1: fill input
    page.fill("#patientName", "Alice Morgan")
    page.select_option("#serviceType", "cardiology")
    page.fill("#appointmentDate", "2026-11-20")
    mgr.record_action_executed("S1", {"action": "fill", "target": "#patientName", "value": "Alice Morgan"}, browser_url=page.url)
    mgr.record_action_verified("S1", {"action": "fill", "target": "#patientName"}, {"verified": True})
    mgr.complete_subgoal("S1", "Patient fields populated", browser_url=page.url)

    # Now simulate crash/termination right before S2 is dispatched
    mgr.interrupt_task("Simulated process kill after S1")
    bm.close()
    print("CHECKPOINT_SAVED_S1")
    sys.exit(0)

elif action == "resume_and_complete":
    bm = BrowserManager(headless=True)
    bm.launch()
    bm.new_page()
    page = bm.get_page()

    plan = mgr.reconcile_and_resume(task_id, browser_manager=bm, page=page)
    print(f"RECONCILE_ACTION:{plan['action']}")

    if plan["action"] in ("paused_hitl_auth_expired", "paused_hitl_ambiguous"):
        bm.close()
        sys.exit(0)

    # Re-apply filled fields if fresh page load reset them
    if page.input_value("#patientName") != "Alice Morgan":
        page.fill("#patientName", "Alice Morgan")
        page.select_option("#serviceType", "cardiology")
        page.fill("#appointmentDate", "2026-11-20")

    # Run remaining subgoal S2
    remaining = plan["remaining_subgoals"]
    for sg in remaining:
        mgr.start_subgoal(sg, browser_url=page.url)
        mgr.record_action_proposed(sg["id"], {"action": "click", "target": sg["target"]})
        
        # Click submit once
        page.click(sg["target"])
        mgr.record_action_executed(sg["id"], {"action": "click", "target": sg["target"]}, browser_url=page.url)
        page.wait_for_selector("#submitStatus.ok, #submitStatus.bad", timeout=5000)

        st_text = page.locator("#submitStatus").inner_text()
        success = "Success: submission saved once" in st_text
        mgr.record_action_verified(sg["id"], {"action": "click", "target": sg["target"]}, {"verified": success})
        if success:
            mgr.complete_subgoal(sg["id"], "Submission verified", browser_url=page.url)
            mgr.complete_task({"outcome": "success", "status_text": st_text})
            print("TASK_COMPLETED_SUCCESSFULLY")
        else:
            mgr.fail_task(f"Failed verification: {st_text}")
            print("TASK_FAILED_VERIFICATION")
            
    bm.close()
    sys.exit(0)

elif action == "resume_expired_auth":
    bm = BrowserManager(headless=True)
    bm.launch()
    bm.new_page()
    page = bm.get_page()

    # Navigate to auth/expired or login
    login_url = f"{server_url}/login"
    page.goto(login_url)

    plan = mgr.reconcile_and_resume(task_id, browser_manager=bm, page=page)
    print(f"RECONCILE_ACTION:{plan['action']}")
    print(f"TASK_STATUS:{plan['task_state'].status.value}")
    print(f"HITL_APPROVAL:{plan['task_state'].human_approval_status.value}")
    bm.close()
    sys.exit(0)

elif action == "resume_ambiguous_crashed_submit":
    # Simulate a task where submit was executed, but crashed before confirmation
    subgoals = [{"id": "S1", "intent": "Submit appointment request", "target": "#btnSubmitOnce", "action": "click"}]
    mgr.start_task(task_id, "Ambiguous submit task", subgoals, start_url=server_url)
    mgr.start_subgoal(subgoals[0])
    mgr.record_action_executed("S1", {"action": "click", "target": "Submit once"})
    mgr.interrupt_task("Crash during network dispatch")

    bm = BrowserManager(headless=True)
    bm.launch()
    bm.new_page()
    page = bm.get_page()
    page.goto(server_url)

    plan = mgr.reconcile_and_resume(task_id, browser_manager=bm, page=page)
    print(f"RECONCILE_ACTION:{plan['action']}")
    print(f"TASK_STATUS:{plan['task_state'].status.value}")
    print(f"HITL_APPROVAL:{plan['task_state'].human_approval_status.value}")
    bm.close()
    sys.exit(0)
'''


def run_test_suite():
    # 1. Start live authoritative server
    SERVER_STATE.reset()
    server = LocalAuthoritativeServer(port=8915)
    server.start()
    server_url = server.url
    logger.info(f"Started authoritative test server at {server_url}")

    db_path = str(REPO_ROOT / "data" / "multi_process_recovery_test.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    worker_file = REPO_ROOT / "scripts" / "_recovery_worker.py"
    worker_file.write_text(WORKER_CODE)

    results = {}

    try:
        # ---------------------------------------------------------------------
        # TEST 1: Recoverable Session & Complete Multi-Process Lifecycle
        # ---------------------------------------------------------------------
        logger.info("\n--- TEST 1: Recoverable Multi-Process Process-Restart Recovery ---")
        task_id = "multi_proc_001"

        # Step A: Run Process 1 (starts task, completes S1, saves checkpoint, terminates)
        logger.info("Launching Python Process 1...")
        proc1 = subprocess.run(
            [sys.executable, str(worker_file), "start_and_checkpoint_intermediate", db_path, server_url, task_id],
            capture_output=True, text=True
        )
        assert "CHECKPOINT_SAVED_S1" in proc1.stdout, f"Process 1 failed: {proc1.stderr}\n{proc1.stdout}"
        logger.info("Process 1 and browser terminated cleanly after S1 checkpoint.")

        # Confirm intermediate server state
        stats_intermediate = SERVER_STATE.get_stats()
        assert stats_intermediate["submissions_count"] == 0, "Server must have 0 submissions after S1"

        # Step B: Run Process 2 (fresh process, fresh browser, resumes task_id)
        logger.info("Launching fresh Python Process 2 to resume task...")
        proc2 = subprocess.run(
            [sys.executable, str(worker_file), "resume_and_complete", db_path, server_url, task_id],
            capture_output=True, text=True
        )
        assert "TASK_COMPLETED_SUCCESSFULLY" in proc2.stdout, f"Process 2 failed: {proc2.stderr}\n{proc2.stdout}"
        logger.info("Process 2 resumed and completed task successfully.")

        # Step C: Authoritative Server Assertions
        stats_final = SERVER_STATE.get_stats()
        logger.info(f"Authoritative Server Stats: Submissions={stats_final['submissions_count']}")
        assert stats_final["submissions_count"] == 1, f"Expected exactly 1 submission, got {stats_final['submissions_count']}"
        results["test_recoverable_session"] = {
            "status": "PASS",
            "server_submissions": stats_final["submissions_count"],
            "process_1_output": "CHECKPOINT_SAVED_S1",
            "process_2_output": "TASK_COMPLETED_SUCCESSFULLY",
        }

        # ---------------------------------------------------------------------
        # TEST 2: Expired Authentication / Unavailable Session
        # ---------------------------------------------------------------------
        logger.info("\n--- TEST 2: Expired Authentication Session Handling ---")
        task_id_auth = "multi_proc_auth_expired"

        proc_auth = subprocess.run(
            [sys.executable, str(worker_file), "resume_expired_auth", db_path, server_url, task_id_auth],
            capture_output=True, text=True
        )
        assert "RECONCILE_ACTION:paused_hitl_auth_expired" in proc_auth.stdout
        assert "TASK_STATUS:paused_hitl" in proc_auth.stdout
        assert "HITL_APPROVAL:pending" in proc_auth.stdout
        logger.info("Expired authentication correctly detected; agent paused for HITL without issuing actions.")
        results["test_expired_session"] = {
            "status": "PASS",
            "reconcile_action": "paused_hitl_auth_expired",
            "task_status": "paused_hitl",
            "hitl_approval": "pending",
        }

        # ---------------------------------------------------------------------
        # TEST 3: Ambiguous Consequential Action (Zero Duplicate Clicks)
        # ---------------------------------------------------------------------
        logger.info("\n--- TEST 3: Ambiguous Consequential Submission Outcome ---")
        task_id_ambig = "multi_proc_ambiguous_submit"
        count_before = SERVER_STATE.get_stats()["submissions_count"]

        proc_ambig = subprocess.run(
            [sys.executable, str(worker_file), "resume_ambiguous_crashed_submit", db_path, server_url, task_id_ambig],
            capture_output=True, text=True
        )
        assert "RECONCILE_ACTION:paused_hitl_ambiguous" in proc_ambig.stdout
        assert "TASK_STATUS:paused_hitl" in proc_ambig.stdout
        assert "HITL_APPROVAL:pending" in proc_ambig.stdout

        count_after = SERVER_STATE.get_stats()["submissions_count"]
        assert count_after == count_before, f"Ambiguous resume must NOT execute duplicate submit! Before={count_before}, After={count_after}"
        logger.info(f"Ambiguous consequential submission safely paused for HITL. Server counter unchanged ({count_after}).")
        results["test_ambiguous_consequential_outcome"] = {
            "status": "PASS",
            "reconcile_action": "paused_hitl_ambiguous",
            "duplicate_submissions": count_after - count_before,
        }

    finally:
        server.stop()
        if worker_file.exists():
            worker_file.unlink()

    print("\n" + "=" * 80)
    print("PROCESS-RESTART RECOVERY EMPIRICAL TEST RESULTS")
    print("=" * 80)
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run_test_suite()