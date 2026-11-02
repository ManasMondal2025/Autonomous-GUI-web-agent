"""Proposal fallback: DOM first, vision detector only when DOM is insufficient.

Rationale: a detector (e.g. OmniParser) costs a full inference pass per
screenshot. Running it on every step wastes time on ordinary web pages where
DOM extraction is free and exact. So the detector is a *fallback*: it runs
only when the DOM yields fewer than `min_proposals` elements.

No detector weights are required by this module. The OmniParser backend
lazy-imports its dependencies and raises an informative error when they or
the weights are missing — same pattern as the QwenVL backend.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class DetectorProposalSource(ABC):
    """Vision-only proposer: image in, element dicts out."""

    @abstractmethod
    def propose(self, image) -> List[Dict[str, Any]]:
        """Return element dicts with at least 'bbox' {x,y,width,height}.

        Optional keys honored downstream: 'tag', 'text', 'ariaLabel',
        'confidence'. IDs are (re)assigned by FallbackProposer.
        """
        raise NotImplementedError


class OmniParserProposalSource(DetectorProposalSource):
    """OmniParser (icon detect + caption) backend.

    Expected layout: a directory containing the detection weights
    (e.g. icon_detect.pt) and, optionally, the caption model directory.
    Nothing is imported or loaded until the first propose() call.
    """

    def __init__(self, model_dir: Optional[str] = None,
                 box_threshold: float = 0.05,
                 caption: bool = True):
        self.model_dir = Path(model_dir) if model_dir else None
        self.box_threshold = box_threshold
        self.caption = caption
        self._loaded = False
        self._detector = None
        self._captioner = None

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            from ultralytics import YOLO  # noqa
        except ImportError as e:
            raise ImportError(
                "OmniParser backend requires `ultralytics` (and torch). "
                "Install with: pip install ultralytics. "
                "No weights were downloaded."
            ) from e
        if self.model_dir is None or not self.model_dir.exists():
            raise FileNotFoundError(
                f"OmniParser weights not found at {self.model_dir}. "
                "Download the OmniParser weights and pass model_dir=.... "
                "No weights were downloaded."
            )
        raise NotImplementedError(
            "OmniParser weight wiring is environment-specific (detector + "
            "caption checkpoints, preprocessing). Subclass "
            "DetectorProposalSource or extend _load() for your checkpoint "
            "layout; the FallbackProposer below works with any backend."
        )

    def propose(self, image) -> List[Dict[str, Any]]:
        self._load()
        raise AssertionError("unreachable")


class FallbackProposer:
    """Use DOM elements; fall back to the detector only when DOM is thin.

    - len(primary) >= min_proposals -> primary untouched, detector NEVER runs.
    - otherwise -> detector runs; its output is used only if it yields
      strictly more elements than the DOM did.
    - detector error -> primary returned unchanged (fail-soft).
    Output IDs are normalized to 0..N-1 so marker numbering is stable.
    """

    def __init__(self, detector: Optional[DetectorProposalSource] = None,
                 min_proposals: int = 1):
        self.detector = detector
        self.min_proposals = max(1, min_proposals)

    def propose(self, primary: List[Dict[str, Any]],
                image=None) -> List[Dict[str, Any]]:
        primary = list(primary or [])
        if len(primary) >= self.min_proposals or self.detector is None:
            return primary
        logger.info(
            f"DOM yielded {len(primary)} element(s) "
            f"(< {self.min_proposals}); running fallback detector")
        try:
            detected = self.detector.propose(image) or []
        except Exception as e:
            logger.warning(f"Fallback detector failed ({e}); keeping DOM output")
            return primary
        if len(detected) > len(primary):
            logger.info(f"Fallback detector found {len(detected)} element(s)")
            return self._normalize_ids(detected)
        return primary

    @staticmethod
    def _normalize_ids(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for i, el in enumerate(elements):
            el = dict(el)
            el["id"] = i
            el.setdefault("visible", True)
            el.setdefault("enabled", True)
            out.append(el)
        return out
