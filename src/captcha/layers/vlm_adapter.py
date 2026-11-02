"""Layer 6 — optional VLM fallback (one signal, never absolute truth)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


VLM_PROMPT = """Inspect this browser screenshot.

Determine whether the page currently contains an interactive
CAPTCHA or human-verification challenge that requires a human
to complete before the task can continue.

Return JSON:

{
  "captcha_present": true/false,
  "blocking": true/false,
  "captcha_type": "...",
  "confidence": 0.0-1.0
}

Do not solve the CAPTCHA.
Do not provide instructions for bypassing it."""


@dataclass
class VisualDetectionResult:
    captcha_present: bool = False
    blocking: bool = False
    captcha_type: Optional[str] = None
    confidence: float = 0.0


class VisualCaptchaDetector(Protocol):
    async def detect(self, screenshot: bytes) -> VisualDetectionResult:
        ...

    def detect_sync(self, screenshot: bytes) -> VisualDetectionResult:
        ...


class StubVisualDetector:
    """Deterministic stub for tests / when no VLM is configured."""

    def __init__(self, result: Optional[VisualDetectionResult] = None):
        self._result = result or VisualDetectionResult()
        self.calls = 0

    async def detect(self, screenshot: bytes) -> VisualDetectionResult:
        self.calls += 1
        return self._result

    def detect_sync(self, screenshot: bytes) -> VisualDetectionResult:
        self.calls += 1
        return self._result


def parse_vlm_json(raw: str) -> VisualDetectionResult:
    """Fail-closed parse of the VLM JSON contract."""
    try:
        import re as _re

        m = _re.search(r"\{.*\}", raw or "", _re.DOTALL)
        if not m:
            return VisualDetectionResult()
        data = json.loads(m.group(0))
        present = bool(data.get("captcha_present", False))
        blocking = bool(data.get("blocking", False)) and present
        ctype = data.get("captcha_type")
        try:
            conf = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        if not present:
            return VisualDetectionResult(
                captcha_present=False, blocking=False, captcha_type=None,
                confidence=conf,
            )
        return VisualDetectionResult(
            captcha_present=True, blocking=blocking,
            captcha_type=str(ctype) if ctype else "unknown", confidence=conf,
        )
    except Exception:
        return VisualDetectionResult()


class BaseVLMVisualDetector:
    """Adapter over the repo's BaseVLM (generate(image, prompt) -> str)."""

    def __init__(self, vlm: Any):
        self.vlm = vlm

    def detect_sync(self, screenshot: bytes) -> VisualDetectionResult:
        try:
            from PIL import Image as _Image
            import io as _io

            image = _Image.open(_io.BytesIO(screenshot)).convert("RGB")
            raw = self.vlm.generate(image, VLM_PROMPT)
            return parse_vlm_json(str(raw or ""))
        except Exception:
            return VisualDetectionResult()

    async def detect(self, screenshot: bytes) -> VisualDetectionResult:
        return self.detect_sync(screenshot)

    def to_signal(self, result: VisualDetectionResult) -> Dict[str, Any]:
        if not result.captcha_present:
            return {"signals": [], "result": result}
        return {"signals": ["vlm_visual_detection"], "result": result}
