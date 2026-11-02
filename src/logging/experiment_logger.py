"""Experiment logger: reproducibility metadata for every run."""

import json
import logging
import platform
import random
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def collect_reproducibility_metadata(
    config: Dict[str, Any],
    model_name: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Gather seed, model, viewport, browser, dataset versions."""
    meta: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": config.get("seed", 42),
        "model": {
            "name": model_name or config.get("model", {}).get("name", ""),
            "temperature": config.get("model", {}).get("temperature", 0.0),
        },
        "viewport": config.get("viewport", {"width": 1280, "height": 720}),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "dataset_version": config.get("dataset", {}).get("version", "unknown"),
        "config_snapshot": config,
    }
    try:
        import playwright
        meta["playwright_version"] = getattr(playwright, "__version__", "unknown")
    except Exception:
        meta["playwright_version"] = "unknown"
    if extra:
        meta.update(extra)
    random.seed(meta["random_seed"])
    try:
        import numpy as np
        np.random.seed(meta["random_seed"] % (2 ** 32))
    except Exception:
        pass
    return meta


class ExperimentLogger:
    """Writes one metadata.json per experiment dir."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log(self, config: Dict[str, Any], model_name: str = "",
            extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        meta = collect_reproducibility_metadata(config, model_name, extra)
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
        return meta
