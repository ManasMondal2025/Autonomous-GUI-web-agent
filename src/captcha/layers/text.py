"""Layer 3 — visible challenge-text detection with FP guards."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List

# (compiled-pattern-source, canonical signal, matched-label)
_PHRASES = [
    (r"i['\u2019]?m not a robot", "robot_checkbox_text", "I'm not a robot"),
    (r"\bi am not a robot\b", "robot_checkbox_text", "I am not a robot"),
    (r"verify you are (a )?human", "human_verification_text", "verify you are human"),
    (r"verify you['\u2019]?re human", "human_verification_text", "verify you're human"),
    (r"verify that you are human", "human_verification_text", "verify that you are human"),
    (r"human verification", "human_verification_text", "human verification"),
    (r"prove you are human", "human_verification_text", "prove you are human"),
    (r"prove you['\u2019]?re human", "human_verification_text", "prove you're human"),
    (r"complete the security check", "security_check_text", "complete the security check"),
    (r"complete the challenge", "challenge_text", "complete the challenge"),
    (r"security check", "security_check_text", "security check"),
    (r"verification required", "verification_required_text", "verification required"),
    (r"select all squares", "image_selection_text", "select all squares"),
    (r"select all images", "image_selection_text", "select all images"),
    (r"click the images", "image_selection_text", "click the images"),
    (r"choose all images", "image_selection_text", "choose all images"),
    (r"\brecaptcha\b", "recaptcha_text", "reCAPTCHA"),
    (r"\bhcaptcha\b", "hcaptcha_text", "hCaptcha"),
    (r"\bcaptcha\b", "captcha_text", "captcha"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), sig, label) for p, sig, label in _PHRASES]

# Documentation/tutorial contexts that contradict a live challenge.
_DOCS_PATTERNS = [
    re.compile(r"captcha (documentation|tutorial|guide|api|docs)", re.IGNORECASE),
    re.compile(r"(documentation|tutorial|guide).{0,40}captcha", re.IGNORECASE),
    re.compile(r"recaptcha api documentation", re.IGNORECASE),
    re.compile(r"human verification documentation", re.IGNORECASE),
    re.compile(r"security challenge article", re.IGNORECASE),
    re.compile(r"coding challenge", re.IGNORECASE),
]


def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = t.lower()
    t = re.sub(r"[\u2018\u2019`´]", "'", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def extract_visible_text(page: Any, max_chars: int = 8000) -> str:
    try:
        text = page.evaluate(
            "(document.body ? document.body.innerText : '').substring(0, %d)"
            % int(max_chars)
        )
        return str(text or "")
    except Exception:
        return ""


def interpret_text(raw_text: str) -> Dict[str, Any]:
    text = normalize_text(raw_text)
    signals: List[str] = []
    matching_text: List[str] = []
    for rx, sig, label in _COMPILED:
        try:
            if rx.search(text):
                if sig not in signals:
                    signals.append(sig)
                if label not in matching_text:
                    matching_text.append(label)
        except Exception:
            continue
    # Docs/tutorial context is contradictory evidence, tracked as a flag
    # (not a detection signal) so clean pages keep signals == [].
    docs_context = any(rx.search(text) for rx in _DOCS_PATTERNS)
    # "challenge" alone (coding challenge etc.) must not signal.
    return {
        "signals": sorted(set(signals)),
        "matching_text": matching_text,
        "docs_context": docs_context,
        "text_length": len(text),
    }


def detect_text(page: Any, max_chars: int = 8000) -> Dict[str, Any]:
    return interpret_text(extract_visible_text(page, max_chars))
