"""Asynchronous UI and Condition-Based Waiting Manager.

Replaces blind 'networkidle' with observable DOM condition waiting.
Handles:
- C1: Delayed search results loading
- C2: Slow form submission (single submission, processing wait, success/fail detection)
- C3: Persistent background network traffic (does not hang waiting for idle)
- C4: Delayed UI state updates (distinguishes transient UPDATING vs final UPDATED)
- C6: Predictable timeout handling with diagnostics
- Duplicate submission prevention on actions with side effects
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AsyncWaiter:
    """Manages condition-based waiting and prevents duplicate submissions."""

    def __init__(self, default_timeout_ms: int = 10000, poll_interval_ms: int = 200):
        self.default_timeout_ms = default_timeout_ms
        self.poll_interval_ms = poll_interval_ms

    def wait_for_predicate(
        self,
        page: Any,
        predicate: Callable[[Any], Tuple[bool, Optional[str]]],
        timeout_ms: Optional[int] = None,
        description: str = "condition"
    ) -> Dict[str, Any]:
        """Poll a Python predicate function against the page until True or timeout.

        Predicate returns (is_done, failure_reason_if_any).
        """
        timeout = timeout_ms if timeout_ms is not None else self.default_timeout_ms
        t0 = time.time()
        deadline = t0 + (timeout / 1000.0)

        diagnostics: Dict[str, Any] = {}
        while time.time() < deadline:
            try:
                done, fail_reason = predicate(page)
                if fail_reason:
                    elapsed = int((time.time() - t0) * 1000)
                    return {
                        "status": "failed",
                        "success": False,
                        "elapsed_ms": elapsed,
                        "failure_reason": fail_reason,
                        "diagnostics": diagnostics
                    }
                if done:
                    elapsed = int((time.time() - t0) * 1000)
                    return {
                        "status": "completed",
                        "success": True,
                        "elapsed_ms": elapsed,
                        "failure_reason": None,
                        "diagnostics": diagnostics
                    }
            except Exception as e:
                diagnostics["last_error"] = str(e)

            page.wait_for_timeout(self.poll_interval_ms)

        elapsed = int((time.time() - t0) * 1000)
        try:
            diagnostics["url"] = page.url
            diagnostics["visible_text_snippet"] = page.inner_text("body")[:300]
        except Exception:
            pass

        return {
            "status": "timeout",
            "success": False,
            "elapsed_ms": elapsed,
            "failure_reason": f"Timed out waiting for {description} after {timeout}ms",
            "diagnostics": diagnostics
        }

    def wait_for_search_results(
        self,
        page: Any,
        query: str,
        results_selector: str = "#searchResults",
        status_selector: str = "#searchStatus",
        timeout_ms: int = 8000
    ) -> Dict[str, Any]:
        """Wait for delayed search results to load and populate."""
        def check_search(p):
            res_loc = p.locator(results_selector)
            stat_loc = p.locator(status_selector)

            stat_text = stat_loc.inner_text() if stat_loc.count() > 0 else ""
            if "error" in stat_text.lower() or "failed" in stat_text.lower():
                return False, f"Search error observed: {stat_text}"

            if res_loc.count() > 0:
                is_visible = res_loc.first.is_visible()
                cls = res_loc.first.get_attribute("class") or ""
                txt = res_loc.first.inner_text() or ""
                if is_visible and "hidden" not in cls.split() and len(txt.strip()) > 0:
                    return True, None

            if "completed" in stat_text.lower():
                return True, None

            return False, None

        return self.wait_for_predicate(page, check_search, timeout_ms=timeout_ms, description="search results")

    def wait_for_ui_update(
        self,
        page: Any,
        status_selector: str = "#updateStatus",
        target_state: str = "State: UPDATED",
        transient_state: str = "UPDATING",
        timeout_ms: int = 6000
    ) -> Dict[str, Any]:
        """Wait for asynchronous UI state to reach target (not transient state)."""
        def check_update(p):
            stat_loc = p.locator(status_selector)
            if stat_loc.count() == 0:
                return False, None
            txt = (stat_loc.first.inner_text() or "").strip()
            if target_state.lower() in txt.lower():
                return True, None
            # Do NOT treat transient UPDATING state as completed
            if transient_state.lower() in txt.lower():
                return False, None
            return False, None

        return self.wait_for_predicate(page, check_update, timeout_ms=timeout_ms, description=f"UI state {target_state}")

    def safe_submit_and_wait(
        self,
        page: Any,
        submit_button_selector: str,
        status_selector: str,
        expected_success_marker: str,
        failure_marker: str = "failed",
        timeout_ms: int = 10000
    ) -> Dict[str, Any]:
        """Prevent duplicate submission: submit once, wait for completion, check status.

        Refuses to click if button is already processing/disabled.
        """
        btn = page.locator(submit_button_selector)
        if btn.count() == 0:
            return {
                "status": "failed",
                "success": False,
                "submission_executed": False,
                "failure_reason": f"Submit button '{submit_button_selector}' not found."
            }

        # Check if already disabled (prevents duplicate submission)
        is_disabled = False
        try:
            if btn.first.is_disabled() or btn.first.get_attribute("disabled") is not None:
                is_disabled = True
            btn_text = (btn.first.inner_text() or "").lower()
            if "processing" in btn_text or "submitting" in btn_text:
                is_disabled = True
        except Exception:
            pass

        if is_disabled:
            logger.info("Submit button is already disabled/processing; preventing duplicate submission.")
            submission_executed = False
        else:
            btn.first.click()
            submission_executed = True
            page.wait_for_timeout(200)

        # Wait for observable completion condition
        def check_submission_result(p):
            stat_loc = p.locator(status_selector)
            if stat_loc.count() == 0:
                return False, None
            txt = (stat_loc.first.inner_text() or "").strip()

            # Check failure condition
            if failure_marker.lower() in txt.lower() or "error" in txt.lower():
                return False, f"Submission failed: {txt}"

            # Check success condition
            if expected_success_marker.lower() in txt.lower():
                return True, None

            return False, None

        outcome = self.wait_for_predicate(
            page,
            check_submission_result,
            timeout_ms=timeout_ms,
            description=f"submission status '{expected_success_marker}'"
        )
        outcome["submission_executed"] = submission_executed
        return outcome
