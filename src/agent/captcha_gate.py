"""CAPTCHA safety gate: adapter between the standalone detector and HITL.

The gate implements NO detection itself -- it consumes
``captcha.detector.CaptchaDetector`` results and decides whether the
existing HITL mechanism must pause autonomous execution.

Routing rule (deliberate): only ``result.blocking == True`` interrupts.
``detected=True, blocking=False`` (hidden/footer/config-only widgets)
flows through untouched.

Intervention reason is always ``captcha_detected`` -- never overloads
``missing_user_information``. The human solves the challenge directly in
the existing browser/noVNC window; the agent never asks for (or handles)
challenge answers.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CAPTCHA_REASON = "captcha_detected"

CAPTCHA_QUESTION = (
    "A CAPTCHA/human-verification challenge is blocking the browser. "
    "Please solve it directly in the browser window, then let the agent "
    "continue. Do not type the answer here."
)

STATUS_CLEAR = "clear"
STATUS_BLOCKING = "blocking"
STATUS_ERROR = "error"
STATUS_DISABLED = "disabled"


@dataclass
class CaptchaGateResult:
    """Outcome of one ``check()``. Never raises out of the gate."""

    status: str = STATUS_CLEAR
    result: Any = None   # CaptchaDetectionResult when available
    error: str = ""      # set only for STATUS_ERROR


class CaptchaGate:
    """Thin adapter: detector result -> HITL routing decision.

    ``detector`` may be a real ``CaptchaDetector`` or a test double with
    ``detect(page)``. When None and the gate is enabled, the real
    detector is constructed lazily (keeps ``agent`` importable without
    the captcha package on sys.path).
    """

    def __init__(self, detector=None, config=None, enabled: bool = True,
                 max_rounds: int = 3):
        self._detector = detector
        self._config = config
        self.enabled = bool(enabled)
        self.max_rounds = max(1, int(max_rounds))
        self.checks = 0
        self.blocks = 0
        self.errors = 0

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]] = None) -> "CaptchaGate":
        d = dict(d or {})
        config = None
        if any(k in d for k in ("confidence_threshold", "blocking_threshold",
                                "weights", "dom_detection", "frame_detection",
                                "text_detection", "structural_detection",
                                "ocr_detection", "vlm_detection")):
            try:
                from captcha.config import CaptchaConfig
                config = CaptchaConfig.from_dict(d)
            except Exception as e:
                logger.warning("[CAPTCHA_GATE] bad captcha config (%s); "
                               "using defaults", e)
                config = None
        return cls(config=config,
                   enabled=bool(d.get("enabled", True)),
                   max_rounds=int(d.get("max_rounds", 3)))

    @property
    def detector(self):
        if self._detector is None:
            from captcha.detector import CaptchaDetector
            kwargs: Dict[str, Any] = {}
            if self._config is not None:
                kwargs["config"] = self._config
            self._detector = CaptchaDetector(**kwargs)
        return self._detector

    def check(self, page: Any) -> CaptchaGateResult:
        """Classify current browser state. Never raises."""
        if not self.enabled:
            return CaptchaGateResult(status=STATUS_DISABLED)
        self.checks += 1
        try:
            result = self.detector.detect(page)
        except Exception as e:
            # Distinguishable from "no CAPTCHA": callers must NOT treat
            # this as blocking=False evidence.
            self.errors += 1
            logger.error("[CAPTCHA_GATE] detector failure: %s: %s",
                         type(e).__name__, e)
            return CaptchaGateResult(status=STATUS_ERROR,
                                     error=f"{type(e).__name__}: {e}")
        try:
            blocking = bool(getattr(result, "blocking", False))
        except Exception:
            blocking = False
        if not blocking:
            return CaptchaGateResult(status=STATUS_CLEAR, result=result)
        self.blocks += 1
        try:
            conf = float(getattr(result, "confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        logger.info(
            "[CAPTCHA_GATE] detected=%s blocking=%s type=%s provider=%s "
            "confidence=%.2f",
            getattr(result, "detected", None), blocking,
            getattr(result, "captcha_type", None),
            getattr(result, "provider", None), conf)
        logger.info("[CAPTCHA_GATE] ACTION REQUIRED: existing HITL "
                     "(reason=%s)", CAPTCHA_REASON)
        return CaptchaGateResult(status=STATUS_BLOCKING, result=result)

    def build_ask_dict(self, result: Any) -> Dict[str, Any]:
        """Existing-HITL-compatible ask_user dict for a blocking result."""
        ctype = getattr(result, "captcha_type", None)
        provider = getattr(result, "provider", None)
        try:
            conf = round(float(getattr(result, "confidence", 0.0) or 0.0), 4)
        except (TypeError, ValueError):
            conf = 0.0
        signals = list(getattr(result, "signals", None) or [])
        detail = ", ".join(
            p for p in (f"type={ctype}", f"provider={provider}")
            if not p.endswith("=None"))
        question = CAPTCHA_QUESTION
        if detail:
            question = f"{CAPTCHA_QUESTION} [{detail}]"
        return {
            "action_type": "ask_user",
            "question": question,
            "reason": CAPTCHA_REASON,
            "captcha_type": ctype,
            "captcha_provider": provider,
            "captcha_confidence": conf,
            "captcha_signals": signals,
        }
