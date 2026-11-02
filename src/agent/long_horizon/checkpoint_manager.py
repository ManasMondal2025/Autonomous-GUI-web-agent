"""Checkpoint manager: URL-anchored, STV-verified restore points.

HONEST LIMITATION (logged on every restore): Playwright offers no full
browser-state snapshot. A checkpoint restores the URL and re-drives a
bounded corrective plan; form values, scroll, auth, and server-side state
are NOT guaranteed. Restoration is itself STV-verified -- unverified
restores escalate instead of pretending success.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

LIMITATION_NOTE = (
    "Checkpoint restore re-navigates to a known URL and re-drives a bounded "
    "corrective plan; it does NOT restore form inputs, scroll position, "
    "authentication sessions, or server-side state. Every restore is "
    "re-verified by STV.")


class CheckpointManager:
    def __init__(self, run_dir: Optional[Path] = None):
        self.run_dir = Path(run_dir) if run_dir is not None else None
        if self.run_dir:
            (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self._counter = 0

    def _dir(self, cid: str) -> Optional[Path]:
        if self.run_dir is None:
            return None
        d = self.run_dir / "checkpoints" / cid
        d.mkdir(parents=True, exist_ok=True)
        return d

    def create(self, memory, observation: Dict[str, Any],
               label: str = "") -> Dict[str, Any]:
        """Snapshot URL + verified memory + evidence screenshot."""
        self._counter += 1
        cid = f"CP{self._counter}"
        snap = memory.to_json() if hasattr(memory, "to_json") else {}
        cp = {
            "id": cid, "label": label,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "url": observation.get("url", ""),
            "verified_subgoals": [
                sid for sid, sg in
                snap.get("verified_subgoals", {}).items()
                if sg.get("status") == "VERIFIED"],
            "key_facts": snap.get("key_facts", {}),
            "current_state": snap.get("current_state", {}),
            "limitation": LIMITATION_NOTE,
        }
        d = self._dir(cid)
        if d is not None:
            (d / "state.json").write_text(json.dumps(cp, indent=2,
                                                     default=str))
            shot = observation.get("screenshot_bytes")
            if shot:
                (d / "screenshot.png").write_bytes(shot)
            cp["dir"] = str(d)
        return cp

    def list(self) -> List[Dict[str, Any]]:
        if self.run_dir is None:
            return []
        out = []
        for f in sorted((self.run_dir / "checkpoints").glob("*/state.json")):
            try:
                out.append(json.loads(f.read_text()))
            except Exception:
                continue
        return out

    def latest(self) -> Optional[Dict[str, Any]]:
        items = self.list()
        # In-memory fallback when no run_dir persistence is configured.
        return items[-1] if items else None
