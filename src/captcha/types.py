"""Controlled vocabularies + structured result for standalone CAPTCHA detection.

No imports from agent/browser/HITL. Pure data types only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


CAPTCHA_TYPES = [
    "recaptcha",
    "hcaptcha",
    "turnstile",
    "cloudflare_challenge",
    "image_captcha",
    "text_captcha",
    "generic_human_verification",
    "unknown",
]

PROVIDERS = [
    "google",
    "hcaptcha",
    "cloudflare",
    "unknown",
]


@dataclass
class CaptchaDetectionResult:
    """Structured detector output (see integration contract)."""

    detected: bool = False
    blocking: bool = False
    confidence: float = 0.0
    captcha_type: Optional[str] = None
    provider: Optional[str] = None
    signals: List[str] = field(default_factory=list)
    evidence: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "detected": bool(self.detected),
            "blocking": bool(self.blocking),
            "confidence": round(float(self.confidence), 4),
            "captcha_type": self.captcha_type,
            "provider": self.provider,
            "signals": list(self.signals or []),
            "evidence": dict(self.evidence or {}),
        }
