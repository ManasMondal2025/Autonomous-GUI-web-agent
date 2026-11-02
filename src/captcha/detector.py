"""Standalone CAPTCHA detector: layered pipeline, deterministic-first.

Read-only w.r.t. the page: evaluate/frames/(gated)screenshot only.
Never calls HITL, never clicks, never solves.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from .aggregation import aggregate, resolve_type_provider
from .config import CaptchaConfig
from .layers import dom as dom_layer
from .layers import frames as frames_layer
from .layers import ocr_adapter as ocr_layer
from .layers import structural as structural_layer
from .layers import text as text_layer
from .layers import vlm_adapter as vlm_layer
from .logging_utils import log_confidence, log_signal
from .types import CaptchaDetectionResult


class CaptchaDetector:
    """Clean public API: ``result = detector.detect(page)``."""

    def __init__(
        self,
        config: Optional[CaptchaConfig] = None,
        ocr_backend: Any | None = None,
        visual_detector: Any | None = None,
    ):
        self.config = config or CaptchaConfig()
        self.ocr_backend = ocr_backend
        self.visual_detector = visual_detector

    # -- public -----------------------------------------------------------
    def detect(self, page: Any) -> CaptchaDetectionResult:
        t0 = time.time()
        cfg = self.config
        if not cfg.enabled:
            return CaptchaDetectionResult(
                detected=False, blocking=False, confidence=0.0,
                captcha_type=None, provider=None, signals=[],
                evidence={"disabled": True},
            )

        signals: List[str] = []
        evidence: Dict[str, Any] = {}
        dom_payload: Dict[str, Any] = {}
        dom_info: Dict[str, Any] = {}
        text_info: Dict[str, Any] = {"signals": [], "docs_context": False}
        frame_info: Dict[str, Any] = {"signals": []}

        # L1+L4: single batched evaluate.
        if cfg.dom_detection or cfg.structural_detection:
            dom_payload = dom_layer.run_dom_evaluate(page)
            dom_info = dom_layer.interpret_dom_payload(dom_payload)
            if cfg.dom_detection:
                for s in dom_info.get("signals", []):
                    if s in (
                        "recaptcha_dom", "g_recaptcha_element", "hcaptcha_dom",
                        "turnstile_dom", "cloudflare_challenge_dom",
                        "generic_captcha_attribute",
                        "generic_captcha_attribute_visible",
                        "data_sitekey_present",
                    ):
                        if s not in signals:
                            signals.append(s)
                if dom_info.get("matching_selectors"):
                    evidence["matching_selectors"] = list(
                        dom_info["matching_selectors"]
                    )
            if cfg.structural_detection:
                struct = structural_layer.detect_structural(dom_payload)
                for s in struct.get("signals", []):
                    if s not in signals:
                        signals.append(s)

        # L2: frames.
        if cfg.frame_detection:
            frame_info = frames_layer.detect_frames(page)
            for s in frame_info.get("signals", []):
                # Normalize iframe-flavored names for aggregation compat.
                alias = {
                    "recaptcha_frame": "recaptcha_frame",
                    "hcaptcha_frame": "hcaptcha_frame",
                    "turnstile_frame": "turnstile_frame",
                    "cloudflare_challenge_frame": "cloudflare_challenge_frame",
                    "generic_captcha_frame": "generic_captcha_frame",
                }.get(s, s)
                if alias not in signals:
                    signals.append(alias)
                # Also record iframe-style alias for evidence readability.
                iframe_alias = alias.replace("_frame", "_iframe")
                if iframe_alias not in signals and alias != iframe_alias:
                    pass  # aggregation uses _frame; keep signals canonical
            if frame_info.get("frame_urls"):
                evidence["frame_urls"] = list(frame_info["frame_urls"])

        # L3: visible text.
        if cfg.text_detection:
            try:
                max_chars = int(cfg.max_text_chars)
            except (TypeError, ValueError):
                max_chars = 8000
            # Reuse Observer-captured text when page is an obs dict (see below).
            text_info = text_layer.detect_text(page, max_chars)
            for s in text_info.get("signals", []):
                if s not in signals:
                    signals.append(s)
            if text_info.get("matching_text"):
                evidence["matching_text"] = list(text_info["matching_text"])

        # Visible proof (needed for blocking=True).
        visible_widget = bool(dom_info.get("visible_widget")) or bool(
            dom_info.get("generic_visible")
        )
        structural_visible = any(
            s in signals
            for s in ("challenge_overlay", "image_grid_selection",
                      "checkbox_like_challenge", "verification_button")
        )
        has_visible_proof = bool(
            visible_widget or structural_visible or frame_info.get("signals")
        )
        hidden_only = bool(signals) and not has_visible_proof
        docs_context = bool(text_info.get("docs_context"))

        confidence, notes = aggregate(
            signals,
            cfg.weights,
            docs_context=docs_context,
            hidden_only=hidden_only,
            has_visible_proof=has_visible_proof,
        )
        for s in signals:
            log_signal("DOM" if "dom" in s or "frame" in s else "page", s)

        # L5: gated OCR fallback (only weak/ambiguous, never on clean pages).
        ocr_used = False
        if (
            cfg.ocr_detection
            and self.ocr_backend is not None
            and 0.30 <= confidence < cfg.blocking_threshold
            and (frame_info.get("signals") or visible_widget or structural_visible)
        ):
            try:
                screenshot = page.screenshot() if hasattr(page, "screenshot") else None
            except Exception:
                screenshot = None
            ocr_info = ocr_layer.detect_ocr(screenshot, self.ocr_backend)
            if ocr_info.get("signals"):
                ocr_used = True
                for s in ocr_info["signals"]:
                    if s not in signals:
                        signals.append(s)
                if ocr_info.get("matching_text"):
                    prev = list(evidence.get("matching_text", []))
                    for t in ocr_info["matching_text"]:
                        if t not in prev:
                            prev.append(t)
                    evidence["matching_text"] = prev
                log_signal("OCR", "human-verification text")
                confidence, notes = aggregate(
                    signals, cfg.weights, docs_context=docs_context,
                    hidden_only=hidden_only,
                    has_visible_proof=True,  # OCR text is itself visual proof
                )

        # L6: gated VLM fallback.
        vlm_type = None
        vlm_conf = None
        if cfg.vlm_detection and self.visual_detector is not None and (
            0.30 <= confidence < cfg.blocking_threshold or ocr_used is False
        ) and confidence < cfg.blocking_threshold and confidence >= 0.30:
            try:
                screenshot = page.screenshot() if hasattr(page, "screenshot") else None
            except Exception:
                screenshot = None
            if screenshot:
                try:
                    vres = self.visual_detector.detect_sync(screenshot)
                    vlm_conf = float(getattr(vres, "confidence", 0.0) or 0.0)
                    if getattr(vres, "captcha_present", False):
                        if "vlm_visual_detection" not in signals:
                            signals.append("vlm_visual_detection")
                        vlm_type = getattr(vres, "captcha_type", None)
                        log_signal("VLM", "visual captcha detection")
                        confidence, _ = aggregate(
                            signals, cfg.weights, docs_context=docs_context,
                            hidden_only=False, has_visible_proof=True,
                            vlm_confidence=vlm_conf,
                        )
                except Exception:
                    pass

        ctype, provider = resolve_type_provider(
            signals, dom_info.get("families"), vlm_type
        )
        detected = bool(signals) and confidence >= cfg.confidence_threshold
        # blocking=True needs visible/interactive proof, not just text.
        blocking = bool(
            detected
            and confidence >= cfg.blocking_threshold
            and has_visible_proof_or_ocr(signals, has_visible_proof, ocr_used)
            and not (docs_context and only_weak_text(signals))
        )
        if not detected:
            ctype, provider = None, None
            # Spec: clean pages return signals=[] and evidence={}.
            if confidence < 0.30:
                signals = []
                evidence = {}
                log_confidence(confidence, blocking)
                return CaptchaDetectionResult(
                    detected=False,
                    blocking=False,
                    confidence=float(confidence),
                    captcha_type=None,
                    provider=None,
                    signals=[],
                    evidence={},
                )
        elapsed_ms = (time.time() - t0) * 1000.0
        evidence["latency_ms"] = round(elapsed_ms, 1)
        if notes:
            evidence["notes"] = notes

        log_confidence(confidence, blocking)
        return CaptchaDetectionResult(
            detected=detected,
            blocking=blocking,
            confidence=float(confidence),
            captcha_type=ctype,
            provider=provider,
            signals=sorted(set(signals)) if detected else ([] if confidence < 0.30 else sorted(set(signals))),
            evidence=evidence,
        )

    async def adetect(self, page: Any) -> CaptchaDetectionResult:
        """Async wrapper (spec example ``await detector.detect(page)``)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.detect, page)

    def detect_from_observation(self, observation: Dict[str, Any]) -> CaptchaDetectionResult:
        """Adapter for Observer dicts (reuses visible_text/url, no re-evaluate)."""

        class _ObsPage:
            def __init__(self, obs: Dict[str, Any]):
                self._obs = obs or {}
                self.frames: List[Any] = []

            def evaluate(self, js: str) -> Any:
                if "querySelectorAll" in js:
                    return {}
                return str(self._obs.get("visible_text", "") or "")[:8000]

            def screenshot(self) -> bytes:
                shot = (self._obs or {}).get("screenshot_bytes")
                if shot is None:
                    raise RuntimeError("no screenshot in observation")
                return shot

        return self.detect(_ObsPage(observation or {}))


def has_visible_proof_or_ocr(
    signals: List[str], has_visible_proof: bool, ocr_used: bool
) -> bool:
    return bool(has_visible_proof or ocr_used or "vlm_visual_detection" in (signals or []))


def only_weak_text(signals: List[str]) -> bool:
    strong = {
        "recaptcha_frame", "hcaptcha_frame", "turnstile_frame",
        "cloudflare_challenge_frame", "generic_captcha_frame",
        "recaptcha_dom", "g_recaptcha_element", "hcaptcha_dom",
        "turnstile_dom", "cloudflare_challenge_dom",
        "challenge_overlay", "image_grid_selection",
        "checkbox_like_challenge", "ocr_human_verification_text",
        "vlm_visual_detection",
    }
    return not any(s in strong for s in (signals or []))
