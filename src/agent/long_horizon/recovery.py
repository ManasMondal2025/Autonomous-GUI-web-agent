"""Recovery policy: ordered STV-decision ladder with budgets.

WAIT -> RETRY -> REGROUND -> LOCAL REPAIR -> LOCAL REPLAN -> GLOBAL REPLAN
-> ROLLBACK, aborting when budgets exhaust. Overall attempt budget is
delegated to the existing RecoveryManager (should_recover/record_recovery);
per-rung counters live here for metrics and logging.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

RUNGS = ("wait", "retry", "reground", "local_repair", "local_replan",
         "global_replan", "rollback")


class RecoveryPolicy:
    def __init__(self, recovery=None, max_waits: int = 3,
                 max_retries: int = 2, max_regrounds: int = 2,
                 max_local_repairs: int = 2, max_local_replans: int = 2,
                 max_global_replans: int = 1, max_rollbacks: int = 1):
        from agent.recovery import RecoveryManager
        self.recovery = recovery or RecoveryManager()
        self.budgets = {"wait": max_waits, "retry": max_retries,
                        "reground": max_regrounds,
                        "local_repair": max_local_repairs,
                        "local_replan": max_local_replans,
                        "global_replan": max_global_replans,
                        "rollback": max_rollbacks}
        self.counters = {r: 0 for r in RUNGS}

    def _rung_available(self, rung: str) -> bool:
        return self.counters[rung] < self.budgets[rung]

    def next(self, decision: str,
             context: Dict[str, Any]) -> Dict[str, Any]:
        """Map an STV decision to a recovery instruction.

        Returns {"instruction": <rung|proceed|abort>, "detail": str}.
        `context` may carry: repeat_same_action (bool), has_checkpoint
        (bool), reground_possible (bool).
        """
        if decision == "PASS":
            return {"instruction": "proceed",
                    "detail": "STV PASS; no recovery needed"}
        order = {"WAIT": ["wait", "retry"],
                 "RETRY": ["retry", "reground"],
                 "REPAIR": ["reground", "local_repair"],
                 "REPLAN": ["local_replan", "global_replan"],
                 "ROLLBACK": ["rollback", "global_replan"]}.get(
            decision, ["local_replan", "global_replan"])
        for rung in order:
            if rung == "rollback" and not context.get("has_checkpoint"):
                continue
            if rung == "reground" and not context.get("reground_possible",
                                                      True):
                continue
            if self._rung_available(rung) and self.recovery.should_recover(
                    {"success": False}):
                self.counters[rung] += 1
                self.recovery.record_recovery(f"{decision} -> {rung}")
                return {"instruction": rung,
                        "detail": f"{decision} escalated to {rung} "
                                  f"(#{self.counters[rung]})"}
        return {"instruction": "abort",
                "detail": f"{decision} with recovery budgets exhausted"}

    def reset_step_counters(self) -> None:
        for rung in ("wait", "retry", "reground"):
            self.counters[rung] = 0
        try:
            self.recovery.reset_step_counters()
        except Exception:
            pass

    def counts(self) -> Dict[str, int]:
        return dict(self.counters)
