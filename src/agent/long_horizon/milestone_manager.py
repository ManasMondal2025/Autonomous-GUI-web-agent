"""Milestone manager: dependency-ordered views over verified subgoals.

A milestone IS a subgoal dict in TaskMemory.verified_subgoals (no parallel
store) plus dependency/terminal metadata. The manager never invents state:
activation requires satisfied dependencies, verification requires STV
evidence (delegated to TaskMemory.mark_subgoal_verified), and VERIFIED
milestones are immutable except for additive evidence metadata -- no later
planner may silently invalidate them.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

MILESTONE_STATUSES = ("PENDING", "ACTIVE", "VERIFIED", "BLOCKED", "FAILED")


class MilestoneManager:
    """Dependency graph controller over TaskMemory subgoal dicts."""

    def __init__(self, memory=None):
        self.memory = memory

    # -- queries ----------------------------------------------------------
    def _sgs(self) -> Dict[str, Dict[str, Any]]:
        if self.memory is None:
            return {}
        return self.memory.verified_subgoals or {}

    def get(self, sid: str) -> Dict[str, Any]:
        return self._sgs().get(sid, {})

    def dependencies(self, sid: str) -> List[str]:
        deps = self.get(sid).get("dependencies", []) or []
        return [str(d) for d in deps]

    def is_dependency_satisfied(self, sid: str) -> bool:
        """Every dependency VERIFIED. Unknown ids fail closed."""
        sgs = self._sgs()
        for dep in self.dependencies(sid):
            if sgs.get(dep, {}).get("status") != "VERIFIED":
                return False
        return True

    def current_milestone(self) -> Optional[str]:
        """ACTIVE first, else first dependency-ready PENDING, else None."""
        sgs = self._sgs()
        for sid, sg in sgs.items():
            if sg.get("status") == "ACTIVE":
                return sid
        for sid, sg in sgs.items():
            if sg.get("status") == "PENDING" and \
                    self.is_dependency_satisfied(sid):
                return sid
        return None

    def remaining_milestones(self) -> List[str]:
        return [sid for sid, sg in self._sgs().items()
                if sg.get("status") in ("PENDING", "ACTIVE", "BLOCKED")]

    def all_verified(self) -> bool:
        sgs = self._sgs()
        return bool(sgs) and all(
            sg.get("status") == "VERIFIED" for sg in sgs.values())

    def terminal_predicates(self, sid: str) -> List[Any]:
        preds = self.get(sid).get("terminal_predicates", []) or []
        if preds:
            return list(preds)
        return list(self.get(sid).get("success_predicates", []) or [])

    def required_facts(self, sid: str) -> List[str]:
        req = self.get(sid).get("required_facts", []) or []
        return [str(r) for r in req]

    def distance(self, sid: str, observation=None) -> Dict[str, Any]:
        """Deterministic required-vs-verified distance (planning signal)."""
        from . import progress as _progress
        return _progress.distance(self.memory, sid, observation)

    # -- transitions --------------------------------------------------------
    def activate_milestone(self, sid: str) -> bool:
        """Activate only when dependencies are satisfied; else BLOCKED.

        Returns True when the milestone is ACTIVE afterwards.
        """
        sgs = self._sgs()
        if sid not in sgs:
            raise ValueError(f"unknown milestone id: {sid}")
        if sgs[sid].get("status") == "VERIFIED":
            return True
        if not self.is_dependency_satisfied(sid):
            sgs[sid]["status"] = "BLOCKED"
            logger.info(f"Milestone {sid} BLOCKED: dependencies "
                        f"{self.dependencies(sid)} unmet")
            return False
        sgs[sid]["status"] = "ACTIVE"
        if self.memory is not None:
            try:
                self.memory.set_current_subgoal(sid)
            except Exception:
                pass
        return True

    def mark_verified(self, sid: str, evidence: List[str]) -> None:
        """STV-evidence-backed verification (delegated; never predictive)."""
        if self.memory is None:
            raise ValueError("no memory attached")
        self.memory.mark_subgoal_verified(sid, evidence)
        try:
            self.memory.set_verified_stage(
                self.get(sid).get("stage", ""))
        except Exception:
            pass

    def append_evidence(self, sid: str, evidence: List[str]) -> None:
        """The ONLY permitted mutation of a VERIFIED milestone: additive
        evidence metadata. Status, contract and facts are never rewritten.
        """
        sg = self.get(sid)
        if not sg:
            raise ValueError(f"unknown milestone id: {sid}")
        if sg.get("status") != "VERIFIED":
            raise ValueError(f"append_evidence only for VERIFIED "
                             f"milestones (got {sg.get('status')})")
        sg.setdefault("evidence", []).extend(list(evidence or []))

    def mark_failed(self, sid: str, reason: str) -> None:
        sgs = self._sgs()
        if sid not in sgs:
            raise ValueError(f"unknown milestone id: {sid}")
        if sgs[sid].get("status") == "VERIFIED":
            raise ValueError(f"refusing to fail VERIFIED milestone {sid}: "
                             f"verified knowledge is immutable ({reason})")
        sgs[sid]["status"] = "FAILED"
        if self.memory is not None:
            self.memory.record_failure({"scope": "milestone", "id": sid,
                                        "reason": reason})

    def mark_blocked(self, sid: str, reason: str) -> None:
        sgs = self._sgs()
        if sid not in sgs:
            raise ValueError(f"unknown milestone id: {sid}")
        if sgs[sid].get("status") == "VERIFIED":
            raise ValueError(f"refusing to block VERIFIED milestone {sid}")
        sgs[sid]["status"] = "BLOCKED"
        logger.info(f"Milestone {sid} BLOCKED: {reason}")

    def record_attempt(self, sid: str, failed: bool = False) -> None:
        sg = self.get(sid)
        if not sg:
            return
        try:
            sg["attempt_count"] = int(sg.get("attempt_count", 0)) + 1
            if failed:
                sg["failure_count"] = int(sg.get("failure_count", 0)) + 1
        except Exception:
            pass
