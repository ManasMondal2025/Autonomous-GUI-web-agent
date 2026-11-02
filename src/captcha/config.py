"""Configuration for the standalone CAPTCHA detector.

Follows repo conventions (YAML dict + dot-notation friendly) without
introducing a new framework. All weights/thresholds configurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


DEFAULT_WEIGHTS: Dict[str, float] = {
    "known_iframe": 0.60,
    "dom_widget": 0.60,
    "provider_frame": 0.60,
    "challenge_text": 0.50,
    "interactive_structure": 0.40,
    "ocr_text": 0.40,
    "vlm": 0.70,
}


@dataclass
class CaptchaConfig:
    enabled: bool = True
    dom_detection: bool = True
    frame_detection: bool = True
    text_detection: bool = True
    structural_detection: bool = True
    ocr_detection: bool = False  # opt-in; heavy dep, lazy import
    vlm_detection: bool = False  # opt-in; needs injected detector

    confidence_threshold: float = 0.60  # detected=True at/above
    blocking_threshold: float = 0.85  # blocking=True at/above + visible proof
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    max_text_chars: int = 8000

    @classmethod
    def from_dict(cls, d: Dict[str, Any] | None) -> "CaptchaConfig":
        d = dict(d or {})
        weights = dict(DEFAULT_WEIGHTS)
        weights.update(dict(d.get("weights") or {}))
        return cls(
            enabled=bool(d.get("enabled", True)),
            dom_detection=bool(d.get("dom_detection", True)),
            frame_detection=bool(d.get("frame_detection", True)),
            text_detection=bool(d.get("text_detection", True)),
            structural_detection=bool(d.get("structural_detection", True)),
            ocr_detection=bool(d.get("ocr_detection", False)),
            vlm_detection=bool(d.get("vlm_detection", False)),
            confidence_threshold=float(d.get("confidence_threshold", 0.60)),
            blocking_threshold=float(d.get("blocking_threshold", 0.85)),
            weights=weights,
            max_text_chars=int(d.get("max_text_chars", 8000)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "dom_detection": self.dom_detection,
            "frame_detection": self.frame_detection,
            "text_detection": self.text_detection,
            "structural_detection": self.structural_detection,
            "ocr_detection": self.ocr_detection,
            "vlm_detection": self.vlm_detection,
            "confidence_threshold": self.confidence_threshold,
            "blocking_threshold": self.blocking_threshold,
            "weights": dict(self.weights),
            "max_text_chars": self.max_text_chars,
        }
