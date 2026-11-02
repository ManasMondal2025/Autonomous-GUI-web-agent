"""Layer 4 — structural interpretation (thin wrapper over DOM payload).

Keeps structural heuristics importable/testable without a browser.
"""

from __future__ import annotations

from typing import Any, Dict, List

STRUCTURAL_SIGNALS = (
    "checkbox_like_challenge",
    "verification_button",
    "challenge_overlay",
    "image_grid_selection",
    "disabled_form_behind_verification",
)


def interpret_structural(payload: Dict[str, Any]) -> Dict[str, Any]:
    structural = dict((payload or {}).get("structural") or {})
    signals: List[str] = []
    if structural.get("has_robot_checkbox_text"):
        signals.append("checkbox_like_challenge")
    if structural.get("has_verify_button"):
        signals.append("verification_button")
    if int(structural.get("visible_overlays") or 0) > 0:
        signals.append("challenge_overlay")
    if int(structural.get("grid_images") or 0) >= 4:
        signals.append("image_grid_selection")
    if structural.get("disabled_form"):
        signals.append("disabled_form_behind_verification")
    return {"signals": sorted(set(signals)), "structural": structural}


def detect_structural(dom_payload: Dict[str, Any]) -> Dict[str, Any]:
    return interpret_structural(dom_payload)
