"""Standalone CAPTCHA detector — public API.

Future HITL integration contract::

    from captcha.detector import CaptchaDetector

    result = detector.detect(page)
    if result.blocking:
        ...  # existing HITL / noVNC takeover (out of scope here)
"""

from .config import CaptchaConfig, DEFAULT_WEIGHTS
from .detector import CaptchaDetector
from .types import (
    CAPTCHA_TYPES,
    PROVIDERS,
    CaptchaDetectionResult,
)

__all__ = [
    "CaptchaDetector",
    "CaptchaConfig",
    "CaptchaDetectionResult",
    "CAPTCHA_TYPES",
    "PROVIDERS",
    "DEFAULT_WEIGHTS",
]
