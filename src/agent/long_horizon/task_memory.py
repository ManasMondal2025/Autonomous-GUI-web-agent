"""Working Task Memory: structured, deterministic per-task state.

Not another LLM. All transitions are explicit method calls; facts carry a
status tier (BELIEF < UNVERIFIED < VERIFIED) and only STV-shaped results
(evidence-backed commits) may promote a fact to VERIFIED.

Extended (Phase 5): stores a validated TaskSpec alongside the existing
state. The spec is the canonical authority on the task's goal, constraints,
permissions, and completion criteria. Facts are tagged with their provenance
(USER_PROVIDED, INFERRED, DEFAULT, STV_VERIFIED) so planners and auditors
can distinguish what the user stated from what was derived.
"""

import json
import logging
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

SUBGOAL_STATUSES = ("PENDING", "ACTIVE", "VERIFIED", "BLOCKED", "FAILED")
FACT_TIERS = ("BELIEF", "UNVERIFIED", "VERIFIED")
# Provenance tags — distinguish where a fact came from
FACT_PROVENANCE = ("USER_PROVIDED", "INFERRED", "DEFAULT", "STV_VERIFIED")
MEMORY_SCHEMA_VERSION = "1.1"


class TaskMemory:
    """Per-task verified state. JSON-serializable throughout."""

    def __init__(self) -> None:
        self.task: Dict[str, Any] = {"id": "", "goal": "",
                                     "success_conditions": []}
        self.verified_subgoals: Dict[str, Dict[str, Any]] = {}
        self.current_subgoal: Optional[str] = None
        self.verified_stage: str = "unknown"
        self.key_facts: Dict[str, Dict[str, Any]] = {}
        self.current_state: Dict[str, Any] = {}
        self.recent_failures: List[Dict[str, Any]] = []
        self.checkpoints: List[Dict[str, Any]] = []
        self._max_failures = 20
        # Phase 5 extensions
        self.task_spec: Optional[Dict[str, Any]] = None  # serialised TaskSpec
        self.schema_version: str = MEMORY_SCHEMA_VERSION

    # -- lifecycle ------------------------------------------------------
    def initialize(self, task_id: str, goal: str,
                   success_conditions: Optional[List[str]] = None,
                   subgoals: Optional[List[Dict[str, Any]]] = None,
                   task_spec=None) -> None:
        """Initialise working memory for a new task.

        task_spec: a TaskSpec instance (Phase 1) or None. When provided it is
        serialised and stored for resumption and provenance auditing.
        """
        self.task = {"id": task_id, "goal": goal,
                     "success_conditions": list(success_conditions or [])}
        # Persist a validated TaskSpec when provided
        if task_spec is not None:
            try:
                self.task_spec = task_spec.to_json() \
                    if hasattr(task_spec, "to_json") else dict(task_spec)
            except Exception as e:
                logger.warning(f"Could not serialise TaskSpec: {e}")
                self.task_spec = None
        self.verified_subgoals = {}
        for sg in subgoals or []:
            sid = sg["id"]
            self.verified_subgoals[sid] = {
                "description": sg.get("description", ""),
                "success_condition": sg.get("success_condition", ""),
                "status": "PENDING", "evidence": [],
                "intent": sg.get("intent", ""),
                "target": sg.get("target", ""),
                "stage": sg.get("stage", ""),
                "parameters": dict(sg.get("parameters", {}) or {}),
                "preconditions": list(sg.get("preconditions", []) or []),
                "success_predicates": list(
                    sg.get("success_predicates", []) or []),
                "dependencies": [str(d) for d in
                                 (sg.get("dependencies", []) or [])],
                "terminal_predicates": list(
                    sg.get("terminal_predicates", []) or []),
                "required_facts": [str(r) for r in
                                   (sg.get("required_facts", []) or [])],
                "attempt_count": int(sg.get("attempt_count", 0) or 0),
                "failure_count": int(sg.get("failure_count", 0) or 0),
                "literals": [str(x) for x in
                             (sg.get("literals", []) or [])][:24],
                "pre_state": dict(sg.get("pre_state", {}) or {})
                if isinstance(sg.get("pre_state", {}), dict) else {},
                "post_state": dict(sg.get("post_state", {}) or {})
                if isinstance(sg.get("post_state", {}), dict) else {},
            }
        self.current_subgoal = next(iter(self.verified_subgoals), None)
        if self.current_subgoal:
            self.verified_subgoals[self.current_subgoal]["status"] = "ACTIVE"
        self.key_facts, self.current_state = {}, {}
        self.recent_failures, self.checkpoints = [], []
        self.verified_stage = "unknown"

    # -- subgoals -------------------------------------------------------
    def add_subgoal(self, sid: str, description: str,
                    success_condition: str = "",
                    intent: str = "", target: str = "", stage: str = "",
                    parameters: Optional[Dict[str, Any]] = None,
                    preconditions: Optional[List[Any]] = None,
                    success_predicates: Optional[List[Any]] = None,
                    dependencies: Optional[List[str]] = None,
                    terminal_predicates: Optional[List[Any]] = None,
                    required_facts: Optional[List[str]] = None,
                    pre_state: Optional[Dict[str, Any]] = None,
                    post_state: Optional[Dict[str, Any]] = None,
                    literals: Optional[List[str]] = None) -> None:
        if sid in self.verified_subgoals:
            raise ValueError(f"duplicate subgoal id: {sid}")
        self.verified_subgoals[sid] = {
            "description": description, "success_condition": success_condition,
            "status": "PENDING", "evidence": [],
            "intent": intent, "target": target, "stage": stage,
            "parameters": dict(parameters or {}),
            "preconditions": list(preconditions or []),
            "success_predicates": list(success_predicates or []),
            "dependencies": [str(d) for d in (dependencies or [])],
            "terminal_predicates": list(terminal_predicates or []),
            "required_facts": [str(r) for r in (required_facts or [])],
            "attempt_count": 0, "failure_count": 0,
            "pre_state": dict(pre_state or {}),
            "post_state": dict(post_state or {}),
            "literals": [str(x) for x in (literals or [])][:24],
        }

    def set_verified_stage(self, stage: str) -> None:
        """Record the stage of the most recently verified subgoal.

        Called ONLY from the subgoal-verified path: the label is
        planner-vocabulary confirmed by STV verification, never a
        prediction about unvisited state.
        """
        if str(stage or "").strip():
            self.verified_stage = str(stage).strip()[:120]

    def set_current_subgoal(self, sid: str) -> None:
        if sid not in self.verified_subgoals:
            raise ValueError(f"unknown subgoal id: {sid}")
        self.current_subgoal = sid
        if self.verified_subgoals[sid]["status"] == "PENDING":
            self.verified_subgoals[sid]["status"] = "ACTIVE"

    def mark_subgoal_verified(self, sid: str,
                              evidence: List[str]) -> None:
        """Requires non-empty evidence: unverified marks are rejected."""
        if sid not in self.verified_subgoals:
            raise ValueError(f"unknown subgoal id: {sid}")
        if not evidence:
            raise ValueError(
                f"refusing to verify subgoal {sid} without evidence")
        self.verified_subgoals[sid]["status"] = "VERIFIED"
        self.verified_subgoals[sid]["evidence"] = list(evidence)

    def fail_subgoal(self, sid: str, reason: str) -> None:
        if sid not in self.verified_subgoals:
            raise ValueError(f"unknown subgoal id: {sid}")
        self.verified_subgoals[sid]["status"] = "FAILED"
        self.record_failure({"scope": "subgoal", "id": sid,
                             "reason": reason})

    def advance_to_next_subgoal(self) -> Optional[str]:
        for sid, sg in self.verified_subgoals.items():
            if sg["status"] in ("PENDING", "ACTIVE"):
                self.set_current_subgoal(sid)
                return sid
        self.current_subgoal = None
        return None

    # -- facts (tiered) ---------------------------------------------------
    def _check_fact(self, name: str, value: Any, status: str,
                    source: str, evidence: Any,
                    provenance: str = "INFERRED") -> Dict[str, Any]:
        if status not in FACT_TIERS:
            raise ValueError(f"bad fact tier: {status}")
        if provenance not in FACT_PROVENANCE:
            provenance = "INFERRED"
        return {"value": value, "status": status, "source": source,
                "evidence": evidence, "provenance": provenance}

    def set_fact(self, name: str, value: Any, status: str = "BELIEF",
                 source: str = "planner", evidence: Any = None,
                 provenance: str = "INFERRED") -> None:
        """Planner/executor hints land as BELIEF (default) or UNVERIFIED.

        provenance: one of FACT_PROVENANCE — distinguishes user-stated facts
        (USER_PROVIDED) from model hypotheses (INFERRED) and defaults (DEFAULT).
        """
        if status == "VERIFIED":
            raise ValueError(
                "use commit_verified_fact() for VERIFIED facts; "
                "direct VERIFIED writes are forbidden")
        self.key_facts[name] = self._check_fact(
            name, value, status, source, evidence, provenance)

    def commit_verified_fact(self, name: str, value: Any, source: str,
                             evidence: Any) -> None:
        """The ONLY path to VERIFIED. Callers must pass STV evidence;
        empty evidence is rejected so execution alone can never verify."""
        if not evidence:
            raise ValueError(
                f"refusing to verify fact {name!r} without STV evidence")
        if not str(source).upper().startswith("STV"):
            raise ValueError(
                f"VERIFIED facts must come from STV (got source={source!r})")
        self.key_facts[name] = self._check_fact(
            name, value, "VERIFIED", source, evidence,
            provenance="STV_VERIFIED")

    def set_state(self, state: Dict[str, Any]) -> None:
        self.current_state = dict(state)

    # -- failures / checkpoints -------------------------------------------
    def record_failure(self, failure: Dict[str, Any]) -> None:
        entry = dict(failure)
        entry.setdefault("at", datetime.now(timezone.utc).isoformat())
        self.recent_failures.append(entry)
        del self.recent_failures[:-self._max_failures]

    def add_checkpoint_ref(self, ref: Dict[str, Any]) -> None:
        self.checkpoints.append(dict(ref))

    def latest_checkpoint_ref(self) -> Optional[Dict[str, Any]]:
        return self.checkpoints[-1] if self.checkpoints else None

    # -- planner context (§28: compact, verified-first) ----------------------
    def get_planner_context(self) -> Dict[str, Any]:
        verified = {k: {"value": f["value"], "evidence": f["evidence"]}
                    for k, f in self.key_facts.items()
                    if f.get("status") == "VERIFIED"}
        cur = self.verified_subgoals.get(self.current_subgoal or "", {})
        return {
            "goal": self.task.get("goal", ""),
            "verified_subgoals": {
                sid: {"description": sg["description"], "status": sg["status"]}
                for sid, sg in self.verified_subgoals.items()},
            "current_subgoal": self.current_subgoal,
            "current_subgoal_description": cur.get("description", ""),
            "verified_key_facts": verified,
            "current_state": deepcopy(self.current_state),
            "recent_failures": deepcopy(self.recent_failures[-5:]),
            "checkpoints": [c.get("id") for c in self.checkpoints],
        }

    def verified_state_summary(self) -> Dict[str, Any]:
        """CURRENT VERIFIED STATE for state-conditioned planning.

        Built strictly from STV-verified material: VERIFIED facts, VERIFIED
        subgoal ids, the active subgoal's contract, and the confirmed
        stage. A planner prediction never enters here (§6:
        prediction != verified state).
        """
        verified = {k: f.get("value")
                    for k, f in self.key_facts.items()
                    if f.get("status") == "VERIFIED"}
        cur = self.current_subgoal
        cur_sg = self.verified_subgoals.get(cur or "", {})
        active = None
        if cur:
            active = {"id": cur,
                      "intent": cur_sg.get("intent", ""),
                      "target": cur_sg.get("target", ""),
                      "description": cur_sg.get("description", ""),
                      "parameters": deepcopy(cur_sg.get("parameters", {})),
                      "success_predicates": deepcopy(
                          cur_sg.get("success_predicates", []))}
        return {
            "url": self.current_state.get("url", ""),
            "verified_stage": self.verified_stage,
            "active_subgoal": active,
            "completed_subgoals": [
                {"id": sid, "intent": sg.get("intent", ""),
                 "description": sg.get("description", "")}
                for sid, sg in self.verified_subgoals.items()
                if sg.get("status") == "VERIFIED"],
            "verified_milestones": [
                sid for sid, sg in self.verified_subgoals.items()
                if sg.get("status") == "VERIFIED"],
            "remaining_requirements": self._remaining_requirements(cur),
            "verified_facts": verified,
            "known_ui_state": deepcopy(self.current_state),
            "recent_failure_trace": deepcopy(
                self.recent_failures[-1]) if self.recent_failures
            else None,
        }

    def _remaining_requirements(self, sid: Optional[str]) -> List[str]:
        """Unverified terminal requirements of one subgoal/milestone.

        Required terminal predicates minus those whose value already
        appears among VERIFIED facts. Deterministic planning signal only.
        """
        if not sid:
            return []
        sg = self.verified_subgoals.get(sid, {})
        preds = sg.get("terminal_predicates", []) or \
            sg.get("success_predicates", []) or []
        facts = {str(k).lower()
                 for k, f in self.key_facts.items()
                 if f.get("status") == "VERIFIED"}
        facts.update(str(v).lower() for v in
                     (f.get("value") for f in self.key_facts.values()
                      if f.get("status") == "VERIFIED") if v)
        remaining = []
        for p in preds:
            if isinstance(p, dict) and p.get("predicate"):
                label = (f"{p.get('target', '?')} "
                         f"{p.get('predicate')} {p.get('value', '')}")
                probe = str(p.get("value", "") or "").lower()
            else:
                label, probe = str(p), str(p).lower()
            if not probe or probe not in facts:
                remaining.append(label[:160])
        return remaining

    # -- persistence --------------------------------------------------------
    def to_json(self) -> Dict[str, Any]:
        return {"schema_version": self.schema_version,
                "task": deepcopy(self.task),
                "verified_subgoals": deepcopy(self.verified_subgoals),
                "current_subgoal": self.current_subgoal,
                "verified_stage": self.verified_stage,
                "key_facts": deepcopy(self.key_facts),
                "current_state": deepcopy(self.current_state),
                "recent_failures": deepcopy(self.recent_failures),
                "checkpoints": deepcopy(self.checkpoints),
                "task_spec": deepcopy(self.task_spec)}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2, default=str))

    @classmethod
    def load(cls, path: Path) -> "TaskMemory":
        data = json.loads(Path(path).read_text())
        stored_ver = data.get("schema_version", "unknown")
        if stored_ver != MEMORY_SCHEMA_VERSION:
            logger.warning(
                f"TaskMemory schema version mismatch: stored={stored_ver!r}, "
                f"current={MEMORY_SCHEMA_VERSION!r}; loading with best-effort")
        m = cls()
        m.task = data.get("task", m.task)
        m.verified_subgoals = data.get("verified_subgoals", {})
        m.current_subgoal = data.get("current_subgoal")
        m.verified_stage = data.get("verified_stage", "unknown")
        m.key_facts = data.get("key_facts", {})
        m.current_state = data.get("current_state", {})
        m.recent_failures = data.get("recent_failures", [])
        m.checkpoints = data.get("checkpoints", [])
        m.task_spec = data.get("task_spec")  # None for old records
        m.schema_version = stored_ver
        return m

    def get_task_spec(self):
        """Return the stored TaskSpec instance, or None if not present.

        Handles backward compatibility: old memory records have no task_spec.
        """
        if self.task_spec is None:
            return None
        try:
            from .task_spec import TaskSpec
            return TaskSpec.from_json(self.task_spec)
        except Exception as e:
            logger.warning(f"Could not deserialise stored TaskSpec: {e}")
            return None
