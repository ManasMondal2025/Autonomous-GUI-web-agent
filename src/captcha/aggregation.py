"""Central evidence aggregator: weighted, capped, contradiction-aware."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# signal -> weight key
_SIGNAL_WEIGHT_KEY = {
    "recaptcha_iframe": "known_iframe",
    "hcaptcha_iframe": "known_iframe",
    "turnstile_iframe": "known_iframe",
    "cloudflare_challenge_iframe": "known_iframe",
    "generic_captcha_iframe": "known_iframe",
    "recaptcha_frame": "provider_frame",
    "hcaptcha_frame": "provider_frame",
    "turnstile_frame": "provider_frame",
    "cloudflare_challenge_frame": "provider_frame",
    "generic_captcha_frame": "provider_frame",
    "recaptcha_dom": "dom_widget",
    "g_recaptcha_element": "dom_widget",
    "hcaptcha_dom": "dom_widget",
    "turnstile_dom": "dom_widget",
    "cloudflare_challenge_dom": "dom_widget",
    "generic_captcha_attribute_visible": "dom_widget",
    "generic_captcha_attribute": "interactive_structure",
    "data_sitekey_present": "interactive_structure",
    "human_verification_text": "challenge_text",
    "robot_checkbox_text": "challenge_text",
    "security_check_text": "challenge_text",
    "challenge_text": "challenge_text",
    "verification_required_text": "challenge_text",
    "image_selection_text": "challenge_text",
    "recaptcha_text": "challenge_text",
    "hcaptcha_text": "challenge_text",
    "captcha_text": "challenge_text",
    "checkbox_like_challenge": "interactive_structure",
    "verification_button": "interactive_structure",
    "challenge_overlay": "interactive_structure",
    "image_grid_selection": "interactive_structure",
    "disabled_form_behind_verification": "interactive_structure",
    "ocr_human_verification_text": "ocr_text",
    "vlm_visual_detection": "vlm",
}


def aggregate(
    signals: List[str],
    weights: Dict[str, float],
    docs_context: bool = False,
    hidden_only: bool = False,
    has_visible_proof: bool = False,
    vlm_confidence: float | None = None,
) -> Tuple[float, List[str]]:
    """Return (confidence 0..1, notes). Pure function for testability."""
    total = 0.0
    for sig in signals or []:
        key = _SIGNAL_WEIGHT_KEY.get(sig)
        if key is None:
            continue
        if sig == "vlm_visual_detection" and vlm_confidence is not None:
            # VLM contributes min(configured weight, its own confidence).
            total += min(float(weights.get(key, 0.7)), float(vlm_confidence))
        else:
            total += float(weights.get(key, 0.0))
    confidence = max(0.0, min(1.0, total))
    notes: List[str] = []

    # Contradiction handling: docs/tutorial context dampens text-only cases.
    text_only = bool(signals) and all(
        _SIGNAL_WEIGHT_KEY.get(s) in ("challenge_text", "interactive_structure", None)
        for s in signals
    )
    if docs_context and text_only:
        confidence = min(confidence, 0.45)
        notes.append("docs_context_capped")
    if hidden_only:
        # A hidden widget alone must never look blocking.
        confidence = min(confidence, 0.55)
        notes.append("hidden_only_capped")
    if not has_visible_proof and confidence >= 0.85 and "vlm_visual_detection" not in (
        signals or []
    ):
        # Strong DOM-only claims without any visible proof stay probable, not high.
        pass  # blocking gate handles this; confidence untouched (auditable)
    return round(confidence, 4), notes


def resolve_type_provider(
    signals: List[str],
    dom_families: List[str] | None = None,
    vlm_type: str | None = None,
) -> Tuple[str | None, str | None]:
    """Controlled-vocabulary classification; None when uncertain."""
    sigs = set(signals or [])
    fams = set(dom_families or [])

    def has(*names: str) -> bool:
        return any(n in sigs for n in names)

    # Provider-specific first (strongest).
    if has("recaptcha_frame", "recaptcha_dom", "g_recaptcha_element") or (
        "recaptcha" in fams and has("recaptcha_iframe")
    ):
        return "recaptcha", "google"
    if has("recaptcha_dom", "g_recaptcha_element", "recaptcha_text") and (
        "recaptcha" in fams
    ):
        return "recaptcha", "google"
    if has("hcaptcha_frame", "hcaptcha_dom", "hcaptcha_text") or "hcaptcha" in fams:
        return "hcaptcha", "hcaptcha"
    if has("turnstile_frame", "turnstile_dom", "turnstile_iframe") or "turnstile" in fams:
        return "turnstile", "cloudflare"
    if has("cloudflare_challenge_frame", "cloudflare_challenge_dom",
           "cloudflare_challenge_iframe") or "cloudflare" in fams:
        return "cloudflare_challenge", "cloudflare"
    if has("generic_captcha_frame", "generic_captcha_iframe"):
        return "unknown", "unknown"
    if has("image_grid_selection", "image_selection_text"):
        return "image_captcha", "unknown"
    if has("robot_checkbox_text", "checkbox_like_challenge"):
        # Generic robot checkbox without provider markers.
        if vlm_type and vlm_type in (
            "recaptcha", "hcaptcha", "turnstile", "cloudflare_challenge",
            "image_captcha", "text_captcha", "generic_human_verification",
        ):
            return vlm_type, "unknown"
        return "generic_human_verification", None
    if has(
        "human_verification_text", "security_check_text", "challenge_text",
        "verification_required_text", "captcha_text", "verification_button",
        "challenge_overlay",
    ):
        if vlm_type and vlm_type in (
            "recaptcha", "hcaptcha", "turnstile", "cloudflare_challenge",
            "image_captcha", "text_captcha", "generic_human_verification",
        ):
            return vlm_type, "unknown"
        return "generic_human_verification", None
    if has("generic_captcha_attribute_visible", "generic_captcha_attribute"):
        return "unknown", "unknown"
    if vlm_type:
        return vlm_type if vlm_type in (
            "recaptcha", "hcaptcha", "turnstile", "cloudflare_challenge",
            "image_captcha", "text_captcha", "generic_human_verification", "unknown",
        ) else "unknown", "unknown"
    return None, None
