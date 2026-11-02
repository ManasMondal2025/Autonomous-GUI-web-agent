"""Grounding strategies: turn (screenshot, instruction) into an actionable target."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from PIL import Image


class BaseGrounder(ABC):
    """Abstract base for all grounding strategies.

    A grounder takes the current observation (screenshot + DOM) and an
    instruction, and produces a *grounded prediction* — a structured dict
    that can be turned into a concrete Playwright action.

    Two concrete implementations are provided:
        - CoordinateGrounder: predicts (x, y) directly
        - SoMGrounder: predicts a marker_id referencing a pre-annotated DOM element
    """

    def __init__(self, name: str = "base"):
        self.name = name

    @abstractmethod
    def ground(
        self,
        screenshot: Image.Image,
        instruction: str,
        dom_elements: list,
        history: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Produce a grounding prediction.

        Returns a dict that always contains at least:
            - "action": str  (click, type, press, scroll, wait, done, fail)
            - "reason": str (model's explanation)
            - "method": str (grounder name)
        Additional fields depend on the method:
            - CoordinateGrounder: {"x": int, "y": int}
            - SoMGrounder: {"target_id": int}
        """
        raise NotImplementedError

    @abstractmethod
    def to_action(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a grounding prediction into an Action dict.

        For coordinate grounding: pass through with x/y.
        For SoM grounding: resolve marker_id → DOM element → center coordinates.
        """
        raise NotImplementedError
