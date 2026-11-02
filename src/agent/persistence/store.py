"""SQLite-backed durable task persistence store."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schema import (
    SCHEMA_VERSION,
    CheckpointRecord,
    TaskState,
    TaskStatus,
    scrub_secrets,
)

logger = logging.getLogger(__name__)


class TaskExistsError(Exception):
    """Raised when attempting to create a new task with an existing ID."""


class TaskNotFoundError(Exception):
    """Raised when a specified task ID cannot be found."""


class IncompatibleSchemaError(Exception):
    """Raised when loading a task record with an unsupported schema version."""


class CorruptedCheckpointError(Exception):
    """Raised when loading a task or checkpoint record with corrupted data."""


class TaskLockedError(Exception):
    """Raised when attempting to resume a task that is concurrently locked by another worker."""


class TaskPersistenceStore:
    """Manages SQLite storage for task states and checkpoints."""

    def __init__(self, db_path: Path | str = "./data/task_persistence.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    task_description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    checkpoint_version INTEGER NOT NULL,
                    current_url TEXT,
                    browser_session_reference TEXT,
                    last_verified_checkpoint TEXT,
                    human_approval_status TEXT NOT NULL,
                    current_subgoal_json TEXT,
                    completed_subgoals_json TEXT,
                    pending_subgoals_json TEXT,
                    verified_results_json TEXT,
                    failed_attempts_json TEXT,
                    pending_action_json TEXT,
                    final_result_json TEXT,
                    metadata_json TEXT,
                    is_locked INTEGER DEFAULT 0,
                    locked_at TEXT
                );
            """)
            try:
                conn.execute("ALTER TABLE tasks ADD COLUMN is_locked INTEGER DEFAULT 0;")
            except Exception:
                pass
            try:
                conn.execute("ALTER TABLE tasks ADD COLUMN locked_at TEXT;")
            except Exception:
                pass
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    checkpoint_version INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    action_phase TEXT NOT NULL,
                    step_number INTEGER NOT NULL,
                    subgoal_id TEXT,
                    subgoal_intent TEXT,
                    browser_url TEXT,
                    action_json TEXT,
                    state_summary_json TEXT,
                    verified_facts_json TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES tasks (task_id) ON DELETE CASCADE
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_task ON checkpoints(task_id);")
            conn.commit()

    def save_task(self, state: TaskState, allow_update: bool = True) -> None:
        """Save or update task state."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT task_id FROM tasks WHERE task_id = ?", (state.task_id,))
            exists = cur.fetchone() is not None

            if exists and not allow_update:
                raise TaskExistsError(f"Task with ID {state.task_id!r} already exists.")

            scrubbed_url = scrub_secrets(state.current_url) if state.current_url else None
            scrubbed_act = scrub_secrets(state.pending_action) if state.pending_action else None
            scrubbed_meta = scrub_secrets(state.metadata) if state.metadata else {}

            if exists:
                conn.execute("""
                    UPDATE tasks SET
                        task_description = ?,
                        status = ?,
                        updated_at = ?,
                        checkpoint_version = ?,
                        current_url = ?,
                        browser_session_reference = ?,
                        last_verified_checkpoint = ?,
                        human_approval_status = ?,
                        current_subgoal_json = ?,
                        completed_subgoals_json = ?,
                        pending_subgoals_json = ?,
                        verified_results_json = ?,
                        failed_attempts_json = ?,
                        pending_action_json = ?,
                        final_result_json = ?,
                        metadata_json = ?
                    WHERE task_id = ?
                """, (
                    state.task_description,
                    state.status.value,
                    state.updated_at,
                    state.checkpoint_version,
                    scrubbed_url,
                    state.browser_session_reference,
                    state.last_verified_checkpoint,
                    state.human_approval_status.value,
                    json.dumps(state.current_subgoal),
                    json.dumps(state.completed_subgoals),
                    json.dumps(state.pending_subgoals),
                    json.dumps(state.verified_results),
                    json.dumps(state.failed_attempts),
                    json.dumps(scrubbed_act),
                    json.dumps(state.final_result),
                    json.dumps(scrubbed_meta),
                    state.task_id,
                ))
            else:
                conn.execute("""
                    INSERT INTO tasks (
                        task_id, task_description, status, created_at, updated_at,
                        checkpoint_version, current_url, browser_session_reference,
                        last_verified_checkpoint, human_approval_status,
                        current_subgoal_json, completed_subgoals_json,
                        pending_subgoals_json, verified_results_json,
                        failed_attempts_json, pending_action_json,
                        final_result_json, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    state.task_id,
                    state.task_description,
                    state.status.value,
                    state.created_at,
                    state.updated_at,
                    state.checkpoint_version,
                    scrubbed_url,
                    state.browser_session_reference,
                    state.last_verified_checkpoint,
                    state.human_approval_status.value,
                    json.dumps(state.current_subgoal),
                    json.dumps(state.completed_subgoals),
                    json.dumps(state.pending_subgoals),
                    json.dumps(state.verified_results),
                    json.dumps(state.failed_attempts),
                    json.dumps(scrubbed_act),
                    json.dumps(state.final_result),
                    json.dumps(scrubbed_meta),
                ))
            conn.commit()

    def acquire_resume_lock(self, task_id: str) -> None:
        """Acquire atomic lock to prevent concurrent resume operations on the same task."""
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cur = conn.execute(
                "UPDATE tasks SET is_locked = 1, locked_at = ? WHERE task_id = ? AND is_locked = 0",
                (now_iso, task_id)
            )
            conn.commit()
            if cur.rowcount == 0:
                # Determine reason: non-existent task or concurrent lock
                t_cur = conn.execute("SELECT is_locked, locked_at FROM tasks WHERE task_id = ?", (task_id,))
                row = t_cur.fetchone()
                if not row:
                    raise TaskNotFoundError(f"Task with ID {task_id!r} not found.")
                raise TaskLockedError(f"Task {task_id!r} is currently locked by another concurrent process (locked at {row['locked_at']}).")

    def release_resume_lock(self, task_id: str) -> None:
        """Release atomic resume lock."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE tasks SET is_locked = 0, locked_at = NULL WHERE task_id = ?",
                (task_id,)
            )
            conn.commit()

    def load_task(self, task_id: str) -> Optional[TaskState]:
        """Load task state by task_id."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
            row = cur.fetchone()
            if not row:
                return None

            version = row["checkpoint_version"]
            if version > SCHEMA_VERSION:
                raise IncompatibleSchemaError(
                    f"Saved task has version {version}, current supported version is {SCHEMA_VERSION}"
                )

            try:
                cur_sg = json.loads(row["current_subgoal_json"]) if row["current_subgoal_json"] else None
                comp_sg = json.loads(row["completed_subgoals_json"]) if row["completed_subgoals_json"] else []
                pend_sg = json.loads(row["pending_subgoals_json"]) if row["pending_subgoals_json"] else []
                ver_res = json.loads(row["verified_results_json"]) if row["verified_results_json"] else []
                fail_att = json.loads(row["failed_attempts_json"]) if row["failed_attempts_json"] else []
                pend_act = json.loads(row["pending_action_json"]) if row["pending_action_json"] else None
                fin_res = json.loads(row["final_result_json"]) if row["final_result_json"] else None
                meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
            except Exception as e:
                raise CorruptedCheckpointError(f"Corrupted task record for {task_id}: {e}")

            return TaskState(
                task_id=row["task_id"],
                task_description=row["task_description"],
                status=TaskStatus(row["status"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                checkpoint_version=row["checkpoint_version"],
                current_url=row["current_url"],
                browser_session_reference=row["browser_session_reference"],
                last_verified_checkpoint=row["last_verified_checkpoint"],
                human_approval_status=row["human_approval_status"],
                current_subgoal=cur_sg,
                completed_subgoals=comp_sg,
                pending_subgoals=pend_sg,
                verified_results=ver_res,
                failed_attempts=fail_att,
                pending_action=pend_act,
                final_result=fin_res,
                is_locked=bool(row["is_locked"]) if "is_locked" in row.keys() else False,
                locked_at=row["locked_at"] if "locked_at" in row.keys() else None,
                metadata=meta,
            )

    def list_tasks(self) -> List[TaskState]:
        """List all saved tasks."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT task_id FROM tasks ORDER BY created_at DESC")
            task_ids = [r["task_id"] for r in cur.fetchall()]
        out = []
        for tid in task_ids:
            t = self.load_task(tid)
            if t:
                out.append(t)
        return out

    def delete_task(self, task_id: str) -> bool:
        """Delete task and its checkpoints."""
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
            conn.commit()
            return cur.rowcount > 0

    def save_checkpoint(self, record: CheckpointRecord) -> None:
        """Save a new checkpoint record."""
        scrubbed_cp_url = scrub_secrets(record.browser_url) if record.browser_url else None
        scrubbed_cp_act = scrub_secrets(record.action) if record.action else None
        scrubbed_cp_state = scrub_secrets(record.state_summary) if record.state_summary else {}
        scrubbed_cp_facts = scrub_secrets(record.verified_facts) if record.verified_facts else {}

        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO checkpoints (
                    checkpoint_id, task_id, checkpoint_version, event_type,
                    action_phase, step_number, subgoal_id, subgoal_intent,
                    browser_url, action_json, state_summary_json,
                    verified_facts_json, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.checkpoint_id,
                record.task_id,
                record.checkpoint_version,
                record.event_type.value,
                record.action_phase.value,
                record.step_number,
                record.subgoal_id,
                record.subgoal_intent,
                scrubbed_cp_url,
                json.dumps(scrubbed_cp_act),
                json.dumps(scrubbed_cp_state),
                json.dumps(scrubbed_cp_facts),
                record.timestamp,
            ))
            conn.commit()

    def get_checkpoints(self, task_id: str) -> List[CheckpointRecord]:
        """Get all checkpoints for a task ordered by step_number and timestamp."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM checkpoints WHERE task_id = ? ORDER BY step_number ASC, timestamp ASC",
                (task_id,)
            )
            rows = cur.fetchall()
            records = []
            for r in rows:
                try:
                    act = json.loads(r["action_json"]) if r["action_json"] else None
                    state_sum = json.loads(r["state_summary_json"]) if r["state_summary_json"] else {}
                    ver_facts = json.loads(r["verified_facts_json"]) if r["verified_facts_json"] else {}
                except Exception as e:
                    raise CorruptedCheckpointError(f"Corrupted checkpoint record {r['checkpoint_id']}: {e}")
                records.append(
                    CheckpointRecord(
                        checkpoint_id=r["checkpoint_id"],
                        task_id=r["task_id"],
                        checkpoint_version=r["checkpoint_version"],
                        event_type=r["event_type"],
                        action_phase=r["action_phase"],
                        step_number=r["step_number"],
                        subgoal_id=r["subgoal_id"],
                        subgoal_intent=r["subgoal_intent"],
                        browser_url=r["browser_url"],
                        action=act,
                        state_summary=state_sum,
                        verified_facts=ver_facts,
                        timestamp=r["timestamp"],
                    )
                )
            return records

    def get_latest_checkpoint(self, task_id: str) -> Optional[CheckpointRecord]:
        cps = self.get_checkpoints(task_id)
        return cps[-1] if cps else None

