"""Layer 2 — frame/iframe URL detection."""

from __future__ import annotations

from typing import Any, Dict, List


def _url_host(url: str) -> str:
    try:
        from urllib.parse import urlparse

        return (urlparse(url or "").hostname or "").lower()
    except Exception:
        return ""


def _classify_frame_url(url: str) -> str | None:
    u = (url or "").lower()
    if not u or u in ("about:blank", "about:srcdoc"):
        return None
    host = _url_host(url)
    # Host-first attribution: query tokens such as hCaptcha's
    # "recaptchacompat=true" must not misattribute a provider frame.
    if "hcaptcha" in host:
        return "hcaptcha_frame"
    if "recaptcha" in host or ("google" in host and "recaptcha" in u):
        return "recaptcha_frame"
    if "challenges.cloudflare.com" in host or "turnstile" in host:
        if "turnstile" in u:
            return "turnstile_frame"
        return "cloudflare_challenge_frame"
    if "captcha" in host:
        return "generic_captcha_frame"
    # Host gave nothing (e.g. data:/blob: URLs): fall back to substrings,
    # still guarding the "recaptchacompat" compat-token collision.
    if "recaptcha" in u.replace("recaptchacompat", ""):
        return "recaptcha_frame"
    if "hcaptcha" in u:
        return "hcaptcha_frame"
    if "turnstile" in u or "challenges.cloudflare.com" in u:
        if "turnstile" in u:
            return "turnstile_frame"
        return "cloudflare_challenge_frame"
    if "captcha" in u:
        return "generic_captcha_frame"
    return None


def detect_frames(page: Any) -> Dict[str, Any]:
    signals: List[str] = []
    frame_urls: List[str] = []
    try:
        frames = list(getattr(page, "frames", []) or [])
    except Exception:
        return {"signals": [], "frame_urls": []}
    # Skip the main frame: its URL is the page itself (e.g. a local path
    # containing "captcha_fixtures" must never count as a CAPTCHA frame).
    try:
        main = page.main_frame if hasattr(page, "main_frame") else None
    except Exception:
        main = None
    try:
        main_url = str(getattr(page, "url", "") or "")
    except Exception:
        main_url = ""
    for i, f in enumerate(frames):
        try:
            url = str(getattr(f, "url", "") or "")
        except Exception:
            continue
        if not url:
            continue
        if main is not None and f is main:
            continue
        if i == 0 and main is None and main_url and url == main_url:
            continue
        frame_urls.append(url[:500])
        sig = _classify_frame_url(url)
        if sig and sig not in signals:
            signals.append(sig)
    return {"signals": sorted(signals), "frame_urls": frame_urls}


def classify_frame_url(url: str) -> str | None:
    """Exposed for unit tests."""
    return _classify_frame_url(url)
