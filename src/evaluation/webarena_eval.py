"""Task-success evaluation over a WebArena task list using the core agent.

For each task: env.reset() → AgentController.run(goal) → env.check_success().
Writes per-task records (JSONL) + summary (JSON). Success comes from the
programmatic checker, never from the agent's self-report.
"""

import logging
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json

from .webarena_env import WebArenaEnvironment, load_tasks

logger = logging.getLogger(__name__)

CHECKPOINT_NAME = "webarena_checkpoint.jsonl"
RECORDS_NAME = "webarena_records.jsonl"
SUMMARY_NAME = "webarena_summary.json"


class WebArenaEvaluator:
    def __init__(self, env: WebArenaEnvironment, controller_factory,
                 output_dir: Optional[Path] = None, resume: bool = False):
        """controller_factory(task) -> configured AgentController.

        resume=True: skip tasks already present in the checkpoint file and
        merge their records into the final report in input order.
        """
        self.env = env
        self.controller_factory = controller_factory
        self.output_dir = Path(output_dir) if output_dir else None
        self.resume = resume
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _termination_info(self, task: Dict[str, Any],
                          run_succeeded: bool = False) -> Dict[str, Any]:
        """Read per-step goal_checks from the task trajectory (if logged)."""
        from verification.goal_verifier import classify_termination
        info: Dict[str, Any] = {"termination_category": "unknown",
                                "first_achieved_step": None}
        if not self.output_dir:
            return info
        traj_base = self.output_dir / "trajectories"
        if not traj_base.exists():
            return info
        candidates = sorted(traj_base.glob(f"{task.get('task_id', '')}*"))
        if not candidates:
            return info
        actions_path = candidates[-1] / "actions.json"
        try:
            actions = json.load(open(actions_path))
        except Exception:
            return info
        checks = [(a.get("extra") or {}).get("goal_check") for a in actions
                  if isinstance(a, dict)]
        # Drop the terminal "finish" pseudo-step, which carries no goal check.
        checks = [gc for gc in checks
                  if isinstance(gc, dict) and "achieved" in gc]
        achieved = [i + 1 for i, gc in enumerate(checks)
                    if isinstance(gc, dict) and gc.get("achieved")]
        info["first_achieved_step"] = achieved[0] if achieved else None
        info["termination_category"] = classify_termination(checks, run_succeeded)
        return info

    # -- incremental checkpointing --------------------------------------
    def _checkpoint_path(self) -> Optional[Path]:
        return self.output_dir / CHECKPOINT_NAME if self.output_dir else None

    def _load_checkpoint(self) -> Dict[Any, Dict[str, Any]]:
        """task_id -> record for previously completed tasks (resume only)."""
        done: Dict[Any, Dict[str, Any]] = {}
        path = self._checkpoint_path()
        if not (self.resume and path and path.exists()):
            return done
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("task_id") is not None:
                    done[record["task_id"]] = record
        logger.info(f"Resume: {len(done)} completed task(s) in checkpoint")
        return done

    def _append_checkpoint(self, record: Dict[str, Any]) -> None:
        """Crash-safe append: one fsynced line per completed task."""
        path = self._checkpoint_path()
        if path is None:
            return
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _write_final(self, records: List[Dict[str, Any]],
                     summary: Dict[str, Any]) -> None:
        if not self.output_dir:
            return
        tmp = self.output_dir / (RECORDS_NAME + ".tmp")
        with open(tmp, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.output_dir / RECORDS_NAME)
        with open(self.output_dir / SUMMARY_NAME, "w") as f:
            json.dump(summary, f, indent=2)

    def _run_single(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run one task; exceptions become failure records (run continues)."""
        try:
            self.env.reset(task)
            controller = self.controller_factory(task)
            outcome = controller.run(task.get("goal", ""))
            check = self.env.check_success(task)
            term_info = self._termination_info(task, bool(outcome.get("success")))
            return {
                "task_id": task.get("task_id"),
                "goal": task.get("goal"),
                "agent_success": bool(outcome.get("success")),
                "agent_steps": outcome.get("steps"),
                "agent_termination": outcome.get("termination", "unknown"),
                "check_success": bool(check.get("success")),
                "check_reason": check.get("reason"),
                # official metric: programmatic check wins
                "success": bool(check.get("success")),
                **term_info,
            }
        except Exception as e:
            logger.exception(f"Task {task.get('task_id')} crashed: {e}")
            return {
                "task_id": task.get("task_id"),
                "goal": task.get("goal"),
                "agent_success": False,
                "agent_steps": 0,
                "agent_termination": "error",
                "check_success": False,
                "check_reason": f"task error: {e}",
                "success": False,
                "termination_category": "not_achieved",
                "first_achieved_step": None,
            }

    def evaluate(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.output_dir and not self.resume:
            # Fresh run: drop any stale checkpoint so a later --resume
            # starts from this run, not an older one.
            stale = self._checkpoint_path()
            if stale.exists():
                stale.unlink()
        done = self._load_checkpoint()

        # Tasks without a unique id can't be resumed reliably: always run
        # them and keep them in input order.
        counts = Counter(t.get("task_id") for t in tasks)
        fresh: List[Tuple[int, Dict[str, Any]]] = []  # (input index, record)
        fresh_by_id: Dict[Any, Dict[str, Any]] = {}
        for i, task in enumerate(tasks):
            tid = task.get("task_id")
            if self.resume and tid is not None and counts[tid] == 1 \
                    and tid in done:
                continue
            record = self._run_single(task)
            self._append_checkpoint(record)
            if tid is None or counts[tid] > 1:
                fresh.append((i, record))
            else:
                fresh_by_id[tid] = record

        # Deterministic merge in input order: fresh run > checkpoint.
        records = []
        for i, task in enumerate(tasks):
            tid = task.get("task_id")
            hit = next((r for j, r in fresh if j == i), None)
            if hit is not None:
                records.append(hit)
            elif tid in fresh_by_id:
                records.append(fresh_by_id[tid])
            else:
                records.append(done[tid])

        n = len(records)
        term_counts = Counter(r.get("termination_category", "unknown")
                              for r in records)
        summary = {
            "n": n,
            "task_success_rate": round(sum(r["success"] for r in records) / max(n, 1), 4),
            "agent_reported_success_rate": round(
                sum(r["agent_success"] for r in records) / max(n, 1), 4),
            "avg_steps": round(sum(r.get("agent_steps", 0) or 0 for r in records) / max(n, 1), 2),
            "termination_categories": dict(term_counts),
        }
        self._write_final(records, summary)
        summary["records"] = records
        return summary
