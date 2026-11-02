"""Experimental Action Chunking helper for GUI Web Agent.

Allows executing several predictable browser actions within a single
planning cycle instead of invoking the planner after every individual action.

Opt-in experimental feature: disabled by default, activated via CLI flag
`--action-chunking` with optional `--chunk-size` (2-6, default 3).
"""

import logging
import re
import time
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default and supported range
DEFAULT_CHUNK_SIZE = 3
MIN_CHUNK_SIZE = 2
MAX_CHUNK_SIZE = 6

# Actions/targets that represent chunk boundaries requiring immediate
# verification or full observation before proceeding.
BOUNDARY_KEYWORDS = (
    "submit", "save", "checkout", "place order", "pay", "purchase",
    "delete", "remove", "confirm", "login", "sign in", "register",
    "logout", "send", "book now", "apply", "finalize"
)


class ChunkMetrics:
    """Tracks metrics for action chunking evaluation."""

    def __init__(self):
        self.chunks_proposed: int = 0
        self.chunks_executed: int = 0
        self.actions_proposed: int = 0
        self.actions_executed: int = 0
        self.chunks_interrupted: int = 0
        self.replanning_calls: int = 0
        self.vlm_calls: int = 0
        self.time_planning: float = 0.0
        self.time_executing: float = 0.0
        self.time_verifying: float = 0.0
        self.interruption_reasons: Dict[str, int] = {}

    def record_interruption(self, reason: str):
        self.chunks_interrupted += 1
        self.interruption_reasons[reason] = self.interruption_reasons.get(reason, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunks_proposed": self.chunks_proposed,
            "chunks_executed": self.chunks_executed,
            "actions_proposed": self.actions_proposed,
            "actions_executed": self.actions_executed,
            "chunks_interrupted": self.chunks_interrupted,
            "replanning_calls": self.replanning_calls,
            "vlm_calls": self.vlm_calls,
            "time_planning_s": round(self.time_planning, 3),
            "time_executing_s": round(self.time_executing, 3),
            "time_verifying_s": round(self.time_verifying, 3),
            "interruption_reasons": dict(self.interruption_reasons),
        }


class ActionChunker:
    """Manages action chunk lifecycle, interruption policies, and chunk logging."""

    def __init__(self, enabled: bool = False, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self.enabled = bool(enabled)
        self.chunk_size = max(MIN_CHUNK_SIZE, min(int(chunk_size or DEFAULT_CHUNK_SIZE), MAX_CHUNK_SIZE))
        self.current_chunk_id: int = 0
        self.metrics = ChunkMetrics()

    def next_chunk_id(self) -> int:
        self.current_chunk_id += 1
        return self.current_chunk_id

    def log_chunk_start(self, chunk_id: int, proposed: int):
        """Log concise start event per specification."""
        msg = f"[ACTION_CHUNK] chunk_id={chunk_id} proposed={proposed}"
        logger.info(msg)
        print(msg)

    def log_chunk_result(self, executed: int, interrupted: bool, reason: str = ""):
        """Log concise execution event per specification."""
        reason_str = f" reason={reason}" if reason else ""
        msg = f"[ACTION_CHUNK] executed={executed} interrupted={str(interrupted).lower()}{reason_str}"
        logger.info(msg)
        print(msg)
        if interrupted:
            replan_msg = "[ACTION_CHUNK] replanning=true"
            logger.info(replan_msg)
            print(replan_msg)

    def is_boundary_action(self, action_dict: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if an action is a boundary action that should terminate the chunk."""
        if not isinstance(action_dict, dict):
            return False, ""
        atype = str(action_dict.get("action", action_dict.get("action_type", ""))).lower()
        if atype in ("done", "fail"):
            return True, "terminal_action"
        if atype == "ask_user":
            return True, "hitl_ask_user"

        target = str(action_dict.get("target", "") or "").lower()
        reason = str(action_dict.get("reason", "") or "").lower()
        combined = f"{target} {reason}"

        for kw in BOUNDARY_KEYWORDS:
            if re.search(r"\b" + re.escape(kw) + r"\b", combined):
                return True, f"boundary_action({kw})"

        return False, ""

    def validate_action(self, action_dict: Dict[str, Any]) -> Tuple[bool, str]:
        """Pre-execution validation of an individual action."""
        if not isinstance(action_dict, dict):
            return False, "action is not a dictionary"
        atype = str(action_dict.get("action", action_dict.get("action_type", ""))).lower()
        valid_actions = {"click", "type", "press", "scroll", "wait", "done", "fail", "ask_user"}
        if atype not in valid_actions:
            return False, f"unsupported action type {atype!r}"
        if atype == "type" and "value" in action_dict and action_dict["value"] is None:
            # Type action without value is not valid unless defaulted
            pass
        return True, ""
