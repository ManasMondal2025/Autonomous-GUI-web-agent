"""Agent state snapshot (per-step, JSON-serializable, no screenshots)."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class AgentState:
    step: int = 0
    url: str = ""
    viewport: Dict[str, int] = field(default_factory=lambda: {"width": 1280, "height": 720})
    scroll: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0})
    num_elements: int = 0
    last_action: str = ""
    last_verification: Optional[bool] = None
    finished: bool = False

    @classmethod
    def from_observation(cls, observation: Dict[str, Any], step: int = 0) -> "AgentState":
        return cls(
            step=step,
            url=observation.get("url", ""),
            viewport=observation.get("viewport", {"width": 1280, "height": 720}),
            scroll=observation.get("scroll_position", {"x": 0, "y": 0}),
            num_elements=len(observation.get("dom_elements", [])),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
