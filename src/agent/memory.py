"""Short-term agent memory with loop detection."""

import logging
from typing import List, Dict, Any, Optional
from collections import deque

logger = logging.getLogger(__name__)


class AgentMemory:
    """Keeps recent (action, result) summaries. Screenshots are NOT stored
    here — they live on disk via the trajectory logger."""

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self._history: deque = deque(maxlen=max_history)
        self._step = 0

    def add(
        self,
        action: Dict[str, Any],
        result: Dict[str, Any],
        url: str = "",
        dom_count_before: Optional[int] = None,
    ) -> None:
        self._step += 1
        resolved = action.get("resolved_element") or {}
        elem_text = resolved.get("text") or action.get("text") or ""
        elem_tag = resolved.get("tag") or action.get("tag") or ""
        entry = {
            "step": self._step,
            "action": action.get("action_type", action.get("action", "?")),
            "target_id": action.get("target_id"),
            "element_text": str(elem_text)[:100] if elem_text else "",
            "element_tag": str(elem_tag)[:30] if elem_tag else "",
            "coordinates": (
                [action.get("x"), action.get("y")]
                if action.get("x") is not None else None
            ),
            "value": action.get("value"),
            "question": action.get("question"),
            "human_answer": action.get("human_answer"),
            "reason": str(action.get("reason", ""))[:500],
            "result": "success" if result.get("success") else f"failed: {result.get('error', '?')}",
            "url": url,
            "verification": result.get("verification"),
            "_dom_count_before": dom_count_before,
        }
        self._history.append(entry)

    def recent(self, n: int = 3) -> List[Dict[str, Any]]:
        return list(self._history)[-n:]

    def all(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def detect_loop(self, repeat_threshold: int = 3) -> Optional[Dict[str, Any]]:
        """Detect if the same (action, target) repeated with no progress."""
        if len(self._history) < repeat_threshold:
            return None
        tail = list(self._history)[-repeat_threshold:]
        # ask_user turns share (None, None) coords by design and must never
        # trip loop detection: repeated questions are a VLM behavior issue
        # (bounded by max_asks), not a browser-state loop.
        if all(e.get("action") == "ask_user" for e in tail):
            return None
        keys = [
            (e["action"], e["target_id"], e["coordinates"] and tuple(e["coordinates"]))
            for e in tail
        ]
        if len(set(keys)) == 1 and all(
            e["verification"] in (False, None) or "failed" in str(e["result"])
            for e in tail
        ):
            return {
                "loop_detected": True,
                "repeated": keys[0],
                "count": repeat_threshold,
            }
        # also detect exact repeat regardless of outcome
        if len(set(keys)) == 1:
            return {
                "loop_detected": True,
                "repeated": keys[0],
                "count": repeat_threshold,
                "note": "same action repeated; check state change",
            }
        return None

    def summary(self) -> str:
        lines = []
        for e in self._history:
            lines.append(
                f"Step {e['step']}: {e['action']} target={e['target_id']} "
                f"→ {e['result']}"
            )
        return "\n".join(lines)

    def clear(self) -> None:
        self._history.clear()
        self._step = 0
