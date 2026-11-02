"""Per-run trajectory logger: step PNGs + actions.json + metadata/result."""

import json
import logging
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TrajectoryLogger:
    def __init__(self, base_dir: Path, run_id: Optional[str] = None):
        self.base_dir = Path(base_dir)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self.run_dir = self.base_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._actions = []
        self._metadata: Dict[str, Any] = {"run_id": self.run_id}

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        self._metadata.update(metadata)
        with open(self.run_dir / "metadata.json", "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

    def log_step(
        self,
        step: int,
        screenshot_bytes: bytes,
        action: Dict[str, Any],
        verification: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        png_path = self.run_dir / f"step_{step:03d}.png"
        with open(png_path, "wb") as f:
            f.write(screenshot_bytes)

        # annotated screenshot if present (SoM)
        ann = (extra or {}).get("annotated_screenshot")
        if ann is not None:
            try:
                ann.save(self.run_dir / f"step_{step:03d}_som.png")
            except Exception:
                pass

        entry = {
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": {k: v for k, v in action.items()
                       if k not in ("screenshot_bytes", "annotated_screenshot", "mapping", "raw_prediction")},
            "verification": {k: v for k, v in verification.items()
                             if k != "screenshot_bytes"},
        }
        if extra:
            entry["extra"] = {k: v for k, v in extra.items()
                              if k not in ("annotated_screenshot", "screenshot_bytes", "mapping")}
        self._actions.append(entry)
        with open(self.run_dir / "actions.json", "w") as f:
            json.dump(self._actions, f, indent=2, default=str)

    def log_result(self, result: Dict[str, Any]) -> None:
        with open(self.run_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

    @property
    def path(self) -> Path:
        return self.run_dir
