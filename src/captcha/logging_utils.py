"""Structured [CAPTCHA] logging. Never logs screenshots or page HTML."""

from __future__ import annotations

import logging

_logger = logging.getLogger("captcha")


def log_signal(kind: str, detail: str) -> None:
    _logger.info("[CAPTCHA] %s signal: %s", kind, detail[:200])


def log_confidence(confidence: float, blocking: bool) -> None:
    _logger.info(
        "[CAPTCHA] Aggregated confidence: %.2f", max(0.0, min(1.0, float(confidence)))
    )
    _logger.info("[CAPTCHA] Blocking: %s", str(bool(blocking)).lower())
