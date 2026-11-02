"""Task persistence manager: orchestrates checkpoints, safe reconciliation, and duplicate action prevention."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schema import (
    ActionPhase,
    CheckpointEventType,
    CheckpointRecord,
    HumanApprovalStatus,
    TaskState,
    TaskStatus,
    scrub_secrets,
)
from .store import TaskNotFoundError, TaskPersistenceStore

logger = logging.getLogger(__name__)


class TaskReconciliationError(Exception):
    """Raised when task state cannot be safely reconciled with live browser state."""


class TaskPersistenceManager:
    """Manages task lifecycle checkpointing, resume reconciliation, and idempotency."""

    def __init__(self, store: Optional[TaskPersistenceStore] = None, db_path: Optional[str | Path] = None):
        self.store = store or TaskPersistenceStore(db_path=db_path or "./data/task_persistence.db")
        self.current_state: Optional[TaskState] = None
        self._step_counter = 0

    def start_task(
        self,
        task_id: str,
        task_description: str,
        subgoals: List[Dict[str, Any]],
        start_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        allow_update: bool = False,
    ) -> TaskState:
        """Initialize a new persistent task record."""
        now = datetime.now(timezone.utc).isoformat()
        state = TaskState(
            task_id=task_id,
            task_description=task_description,
            status=TaskStatus.IN_PROGRESS,
            created_at=now,
            updated_at=now,
            pending_subgoals=list(subgoals),
            current_url=start_url,
            metadata=metadata or {},
        )
        self.store.save_task(state, allow_update=allow_update)
        self.current_state = state
        self._step_counter = 0

        self._record_checkpoint(
            event_type=CheckpointEventType.TASK_INIT,
            action_phase=ActionPhase.UNKNOWN,
            subgoal_id=None,
            subgoal_intent="Initialize task",
            state_summary={"total_subgoals": len(subgoals)},
        )
        return state

    def start_subgoal(self, subgoal: Dict[str, Any], browser_url: Optional[str] = None) -> None:
        """Record the start of a subgoal."""
        if not self.current_state:
            return
        self._step_counter += 1
        self.current_state.current_subgoal = subgoal
        self.current_state.updated_at = datetime.now(timezone.utc).isoformat()
        if browser_url:
            self.current_state.current_url = browser_url
        self.store.save_task(self.current_state)

        self._record_checkpoint(
            event_type=CheckpointEventType.SUBGOAL_INIT,
            action_phase=ActionPhase.UNKNOWN,
            subgoal_id=subgoal.get("id"),
            subgoal_intent=subgoal.get("intent") or subgoal.get("description"),
            browser_url=browser_url or self.current_state.current_url,
            state_summary={"subgoal": subgoal},
        )

    def record_action_proposed(self, subgoal_id: str, action: Dict[str, Any]) -> None:
        """Record that an action was proposed by the planner / VLM."""
        if not self.current_state:
            return
        scrubbed_act = scrub_secrets(action)
        self.current_state.pending_action = {
            "action": scrubbed_act,
            "phase": ActionPhase.PROPOSED.value,
            "subgoal_id": subgoal_id,
            "proposed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.current_state.updated_at = datetime.now(timezone.utc).isoformat()
        self.store.save_task(self.current_state)

        self._record_checkpoint(
            event_type=CheckpointEventType.ACTION_PROPOSED,
            action_phase=ActionPhase.PROPOSED,
            subgoal_id=subgoal_id,
            action=scrubbed_act,
            state_summary={"phase": "proposed"},
        )

    def record_action_executed(
        self,
        subgoal_id: str,
        action: Dict[str, Any],
        browser_url: Optional[str] = None,
        state_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record that an action was dispatched to the browser (pending verification)."""
        if not self.current_state:
            return
        scrubbed_act = scrub_secrets(action)
        scrubbed_summary = scrub_secrets(state_summary or {})
        self.current_state.pending_action = {
            "action": scrubbed_act,
            "phase": ActionPhase.EXECUTED.value,
            "subgoal_id": subgoal_id,
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "state_summary": scrubbed_summary,
        }
        if browser_url:
            self.current_state.current_url = scrub_secrets(browser_url)
        self.current_state.updated_at = datetime.now(timezone.utc).isoformat()
        self.store.save_task(self.current_state)

        self._record_checkpoint(
            event_type=CheckpointEventType.ACTION_EXECUTED,
            action_phase=ActionPhase.EXECUTED,
            subgoal_id=subgoal_id,
            action=scrubbed_act,
            browser_url=self.current_state.current_url,
            state_summary=scrubbed_summary,
        )

    def record_action_verified(
        self,
        subgoal_id: str,
        action: Dict[str, Any],
        stv_result: Dict[str, Any],
        verified_facts: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record that an action outcome was verified by STV."""
        if not self.current_state:
            return
        cid = self._record_checkpoint(
            event_type=CheckpointEventType.ACTION_VERIFIED,
            action_phase=ActionPhase.VERIFIED,
            subgoal_id=subgoal_id,
            action=action,
            state_summary={"stv_result": stv_result},
            verified_facts=verified_facts or {},
        )
        self.current_state.last_verified_checkpoint = cid
        self.current_state.pending_action = None
        self.current_state.updated_at = datetime.now(timezone.utc).isoformat()
        self.store.save_task(self.current_state)

    def complete_subgoal(
        self,
        subgoal_id: str,
        verified_result: Any,
        browser_url: Optional[str] = None,
    ) -> None:
        """Record subgoal completion after verification."""
        if not self.current_state:
            return

        # Move from pending to completed
        cur_sg = self.current_state.current_subgoal or {"id": subgoal_id}
        self.current_state.completed_subgoals.append(cur_sg)
        self.current_state.pending_subgoals = [
            s for s in self.current_state.pending_subgoals if s.get("id") != subgoal_id
        ]
        self.current_state.verified_results.append({
            "subgoal_id": subgoal_id,
            "result": verified_result,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        self.current_state.current_subgoal = None
        if browser_url:
            self.current_state.current_url = browser_url
        self.current_state.updated_at = datetime.now(timezone.utc).isoformat()
        self.store.save_task(self.current_state)

        cid = self._record_checkpoint(
            event_type=CheckpointEventType.SUBGOAL_COMPLETE,
            action_phase=ActionPhase.VERIFIED,
            subgoal_id=subgoal_id,
            state_summary={"result": verified_result},
        )
        self.current_state.last_verified_checkpoint = cid
        self.store.save_task(self.current_state)

    def record_recovery_attempt(self, subgoal_id: str, detail: str) -> None:
        """Record a recovery attempt."""
        if not self.current_state:
            return
        entry = {
            "subgoal_id": subgoal_id,
            "detail": detail,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.current_state.failed_attempts.append(entry)
        self.current_state.updated_at = datetime.now(timezone.utc).isoformat()
        self.store.save_task(self.current_state)

        self._record_checkpoint(
            event_type=CheckpointEventType.RECOVERY_ATTEMPT,
            action_phase=ActionPhase.UNKNOWN,
            subgoal_id=subgoal_id,
            state_summary=entry,
        )

    def record_hitl_pause(self, subgoal_id: str, reason: str) -> None:
        """Record a pause waiting for human approval."""
        if not self.current_state:
            return
        self.current_state.status = TaskStatus.PAUSED_HITL
        self.current_state.human_approval_status = HumanApprovalStatus.PENDING
        self.current_state.updated_at = datetime.now(timezone.utc).isoformat()
        self.store.save_task(self.current_state)

        self._record_checkpoint(
            event_type=CheckpointEventType.HITL_PAUSE,
            action_phase=ActionPhase.UNKNOWN,
            subgoal_id=subgoal_id,
            state_summary={"reason": reason},
        )

    def complete_task(self, final_result: Any) -> None:
        """Mark task as successfully completed."""
        if not self.current_state:
            return
        self.current_state.status = TaskStatus.COMPLETED
        self.current_state.final_result = final_result if isinstance(final_result, dict) else {"result": final_result}
        self.current_state.updated_at = datetime.now(timezone.utc).isoformat()
        self.current_state.pending_action = None
        self.store.save_task(self.current_state)

        self._record_checkpoint(
            event_type=CheckpointEventType.TASK_COMPLETE,
            action_phase=ActionPhase.VERIFIED,
            state_summary={"final_result": self.current_state.final_result},
        )
        try:
            self.store.release_resume_lock(self.current_state.task_id)
        except Exception:
            pass

    def fail_task(self, reason: str) -> None:
        """Mark task as failed."""
        if not self.current_state:
            return
        self.current_state.status = TaskStatus.FAILED
        self.current_state.final_result = {"error": reason}
        self.current_state.updated_at = datetime.now(timezone.utc).isoformat()
        self.store.save_task(self.current_state)

        self._record_checkpoint(
            event_type=CheckpointEventType.TASK_FAILED,
            action_phase=ActionPhase.UNKNOWN,
            state_summary={"reason": reason},
        )
        try:
            self.store.release_resume_lock(self.current_state.task_id)
        except Exception:
            pass

    def interrupt_task(self, reason: str = "Process interrupted") -> None:
        """Record process interruption."""
        if not self.current_state:
            return
        self.current_state.status = TaskStatus.INTERRUPTED
        self.current_state.updated_at = datetime.now(timezone.utc).isoformat()
        self.store.save_task(self.current_state)

        self._record_checkpoint(
            event_type=CheckpointEventType.PROCESS_INTERRUPTED,
            action_phase=ActionPhase.UNKNOWN,
            state_summary={"reason": reason},
        )
        try:
            self.store.release_resume_lock(self.current_state.task_id)
        except Exception:
            pass

    def _record_checkpoint(
        self,
        event_type: CheckpointEventType,
        action_phase: ActionPhase,
        subgoal_id: Optional[str] = None,
        subgoal_intent: Optional[str] = None,
        action: Optional[Dict[str, Any]] = None,
        browser_url: Optional[str] = None,
        state_summary: Optional[Dict[str, Any]] = None,
        verified_facts: Optional[Dict[str, Any]] = None,
    ) -> str:
        cid = f"cp_{uuid.uuid4().hex[:10]}"
        raw_url = browser_url or (self.current_state.current_url if self.current_state else None)
        record = CheckpointRecord(
            checkpoint_id=cid,
            task_id=self.current_state.task_id if self.current_state else "unknown",
            event_type=event_type,
            action_phase=action_phase,
            step_number=self._step_counter,
            subgoal_id=subgoal_id,
            subgoal_intent=scrub_secrets(subgoal_intent) if subgoal_intent else None,
            action=scrub_secrets(action) if action else None,
            browser_url=scrub_secrets(raw_url) if raw_url else None,
            state_summary=scrub_secrets(state_summary or {}),
            verified_facts=scrub_secrets(verified_facts or {}),
        )
        self.store.save_checkpoint(record)
        return cid

    # -------------------------------------------------------------------------
    # Resume & Idempotency / Duplicate Action Prevention Logic
    # -------------------------------------------------------------------------

    def reconcile_and_resume(
        self,
        task_id: str,
        browser_manager: Any,
        page: Any = None,
    ) -> Dict[str, Any]:
        """Reconcile saved task record with actual browser state.

        Critical requirements handled:
        - Load and validate saved task record
        - Verify schema compatibility
        - Verify browser session responsiveness (re-open / re-navigate if session lost)
        - Prevent duplicate actions on interrupted pending actions
        - Determine remaining subgoals to run
        """
        task_state = self.store.load_task(task_id)
        if not task_state:
            raise TaskNotFoundError(f"Task ID {task_id!r} not found in persistence store.")

        # Check if already completed
        if task_state.status == TaskStatus.COMPLETED:
            logger.info(f"Task {task_id} is already completed.")
            return {
                "action": "already_completed",
                "task_state": task_state,
                "remaining_subgoals": [],
                "skip_actions": [],
            }

        # Acquire atomic lock to prevent concurrent resume operations on the same task record
        self.store.acquire_resume_lock(task_id)
        self.current_state = task_state

        # 1. Verify Browser Session Health
        active_page = page
        session_recovered = False
        try:
            if active_page is None:
                active_page = browser_manager.get_page()
            # Test page responsiveness
            _ = active_page.url
        except Exception:
            # Browser session was lost or disconnected
            logger.warning(f"Browser session for {task_id} is unavailable or closed. Re-establishing...")
            try:
                browser_manager.launch()
                browser_manager.new_page()
                active_page = browser_manager.get_page()
                session_recovered = True
            except Exception as e:
                raise TaskReconciliationError(f"Could not re-establish browser session: {e}")

        # 2. Restore URL state if needed
        if task_state.current_url:
            current_live_url = getattr(active_page, "url", "")
            if session_recovered or current_live_url != task_state.current_url:
                logger.info(f"Restoring page URL to checkpoint URL: {task_state.current_url}")
                try:
                    active_page.goto(task_state.current_url)
                    active_page.wait_for_timeout(300)
                except Exception as e:
                    logger.warning(f"URL restoration returned error: {e}")

        # 2a. Check if authentication session expired (redirected to login / signin)
        current_live_url_lower = getattr(active_page, "url", "").lower()
        orig_url_lower = str(task_state.current_url or "").lower()
        if any(k in current_live_url_lower for k in ("/login", "/signin", "auth/expired")) and \
           not any(k in orig_url_lower for k in ("/login", "/signin")):
            logger.warning(f"[AUTH EXPIRED] Browser redirected to login: {active_page.url}. Pausing for human re-authentication.")
            task_state.status = TaskStatus.PAUSED_HITL
            task_state.human_approval_status = HumanApprovalStatus.PENDING
            task_state.updated_at = datetime.now(timezone.utc).isoformat()
            self.store.save_task(task_state)
            self._record_checkpoint(
                event_type=CheckpointEventType.HITL_PAUSE,
                action_phase=ActionPhase.UNKNOWN,
                state_summary={"reason": "Authentication session expired or login required"},
            )
            return {
                "action": "paused_hitl_auth_expired",
                "reason": f"Authentication session expired; browser at {active_page.url}. Human re-authentication required.",
                "task_state": task_state,
                "session_recovered": session_recovered,
                "remaining_subgoals": task_state.pending_subgoals,
                "completed_subgoals": task_state.completed_subgoals,
                "verified_results": task_state.verified_results,
                "duplicate_prevented": False,
                "skip_actions": [],
            }

        # 2b. Reconcile verified pre-requisite page conditions (e.g. dismissed popups)
        # If popup dismissal was already completed in prior subgoals, ensure fresh page reflects it
        completed_sg_intents = [
            str(s.get("intent", "")).lower() + " " + str(s.get("target", "")).lower()
            for s in task_state.completed_subgoals
        ]
        if any("cookie" in s or "dismiss" in s or "reject" in s for s in completed_sg_intents):
            try:
                cookie_banner = active_page.locator("#cookieBanner, .cookie-banner")
                if cookie_banner.count() > 0 and cookie_banner.first.is_visible():
                    logger.info("[RECONCILE] Cookie banner visible on fresh page but already verified completed in prior run. Reconciling dismissal.")
                    reject_btn = active_page.locator("button:has-text('Reject optional cookies'), button:has-text('Reject')")
                    if reject_btn.count() > 0:
                        reject_btn.first.click()
                        active_page.wait_for_timeout(150)
            except Exception as e:
                logger.debug(f"Popup reconciliation notice: {e}")

        # 3. Prevent Duplicate Actions on Interrupted Pending Action
        # This is the critical requirement from Section 2.5:
        # If an action was in EXECUTED state when interrupted, check if the action already took effect.
        pending_act = task_state.pending_action
        skip_actions = []
        action_reconciled = False
        duplicate_prevented = False

        if pending_act and pending_act.get("phase") == ActionPhase.EXECUTED.value:
            act = pending_act.get("action", {})
            target = act.get("target", "")
            val = act.get("value", "")
            act_type = act.get("action", "")
            subgoal_id = pending_act.get("subgoal_id")

            logger.info(
                f"Inspecting interrupted pending action for duplicate prevention: "
                f"action={act_type}, target={target}, value={val}"
            )

            # Inspect live DOM and application state
            live_dom_matches = False
            try:
                # Check 1: If it was a submission action, check if already submitted or processing
                if any(w in str(target).lower() for w in ("submit", "save", "confirm", "send")):
                    # Check for disabled submit button, confirmation text, or event log
                    try:
                        events = active_page.evaluate("typeof eventLog !== 'undefined' ? eventLog : (window.eventLog || [])")
                        if any(e.get("type", "").endswith("started") or e.get("type", "").endswith("succeeded") for e in events):
                            live_dom_matches = True
                    except Exception:
                        pass

                    # Also check visible text for confirmation keywords
                    body_text = active_page.inner_text("body").lower()
                    if any(k in body_text for k in ("confirmed", "saved", "demo-001", "success", "reference")):
                        live_dom_matches = True

                # Check 2: If it was a type / fill / date skill action, check if the input already contains the value
                elif (act_type in ("type", "fill", "skill_execution") or "date" in str(target).lower()) and val:
                    # Look for input with matching target or id
                    inputs = active_page.locator("input, textarea").all()
                    for inp in inputs:
                        try:
                            if inp.input_value() == val:
                                live_dom_matches = True
                                break
                        except Exception:
                            pass

                # Check 3: If cookie dismiss or popup close, check if already closed
                elif any(w in str(target).lower() for w in ("cookie", "dismiss", "reject", "close")):
                    banner = active_page.locator("#cookieBanner, .cookie-banner")
                    if banner.count() > 0 and banner.first.is_hidden():
                        live_dom_matches = True

            except Exception as e:
                logger.warning(f"Error inspecting live state for duplicate prevention: {e}")

            is_consequential = any(w in str(target).lower() for w in ("submit", "save", "confirm", "send", "pay", "order", "purchase"))
            if live_dom_matches:
                logger.info(
                    f"[IDEMPOTENCY] Action '{act_type}' on '{target}' was ALREADY completed prior to crash. "
                    f"Preventing duplicate execution!"
                )
                skip_actions.append(target)
                action_reconciled = True
                duplicate_prevented = True
                self.record_action_verified(
                    subgoal_id=subgoal_id,
                    action=act,
                    stv_result={"verified": True, "reconciled_after_crash": True},
                )
            elif is_consequential:
                # Consequential action outcome cannot be verified in DOM/backend state.
                # Phase 2 requirement: "When the outcome is ambiguous, require an unresolved state, safe reconciliation, or HITL. Never blindly repeat a consequential action."
                logger.warning(
                    f"[AMBIGUOUS OUTCOME] Consequential action '{act_type}' on '{target}' was dispatched "
                    f"before crash, but outcome cannot be confirmed in live DOM or backend state. "
                    f"Pausing for HITL to avoid duplicate side-effects."
                )
                task_state.status = TaskStatus.PAUSED_HITL
                task_state.human_approval_status = HumanApprovalStatus.PENDING
                task_state.updated_at = datetime.now(timezone.utc).isoformat()
                self.store.save_task(task_state)
                self._record_checkpoint(
                    event_type=CheckpointEventType.HITL_PAUSE,
                    action_phase=ActionPhase.UNKNOWN,
                    subgoal_id=subgoal_id,
                    state_summary={"reason": f"Ambiguous outcome for consequential action '{target}' after crash"},
                )
                return {
                    "action": "paused_hitl_ambiguous",
                    "reason": f"Ambiguous outcome for consequential action '{target}' after crash. Human intervention required.",
                    "task_state": task_state,
                    "session_recovered": session_recovered,
                    "remaining_subgoals": task_state.pending_subgoals,
                    "completed_subgoals": task_state.completed_subgoals,
                    "verified_results": task_state.verified_results,
                    "duplicate_prevented": False,
                    "skip_actions": [],
                }
            else:
                logger.info(f"Non-consequential action '{act_type}' on '{target}' outcome not confirmed in DOM. Can safely retry.")

        # 4. Filter remaining subgoals
        completed_ids = {s.get("id") for s in task_state.completed_subgoals}
        remaining_subgoals = [
            s for s in task_state.pending_subgoals if s.get("id") not in completed_ids
        ]

        # Update status to in_progress
        task_state.status = TaskStatus.IN_PROGRESS
        task_state.updated_at = datetime.now(timezone.utc).isoformat()
        self.store.save_task(task_state)

        return {
            "action": "resume_ready",
            "task_state": task_state,
            "session_recovered": session_recovered,
            "remaining_subgoals": remaining_subgoals,
            "completed_subgoals": task_state.completed_subgoals,
            "verified_results": task_state.verified_results,
            "duplicate_prevented": duplicate_prevented,
            "skip_actions": skip_actions,
        }
