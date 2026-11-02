"""Layer 5 — OCR fallback (gated, opt-in, never solves)."""

from __future__ import annotations

from typing import Any, Dict, List, Protocol

from .text import interpret_text


class OCRBackend(Protocol):
    def detect_text(self, screenshot: Any) -> List[Dict[str, Any]]:
        ...


def ocr_text_from_regions(regions: List[Dict[str, Any]]) -> str:
    parts = []
    for r in regions or []:
        try:
            t = str((r or {}).get("text", "") or "").strip()
            if t:
                parts.append(t)
        except Exception:
            continue
    return "\n".join(parts)


def _as_backend_image(screenshot: Any) -> Any:
    """Normalize to what OCR backends consume (PIL RGB image).

    The repo's PaddleOCRBackend expects a PIL Image / pixel array; raw PNG
    bytes reach it as a 1-D uint8 array and crash inside cv2. Decode bytes
    here so the adapter's ``bytes`` contract actually works end to end.
    """
    if isinstance(screenshot, (bytes, bytearray)):
        try:
            from PIL import Image as _Image
            import io as _io

            return _Image.open(_io.BytesIO(bytes(screenshot))).convert("RGB")
        except Exception:
            return screenshot  # let the backend try raw input
    return screenshot


def detect_ocr(
    screenshot: bytes | None,
    ocr_backend: OCRBackend | None,
) -> Dict[str, Any]:
    """Run OCR backend over a screenshot; return challenge-text signals."""
    if screenshot is None or ocr_backend is None:
        return {"signals": [], "matching_text": [], "ocr_text_excerpt": ""}
    try:
        regions = ocr_backend.detect_text(_as_backend_image(screenshot)) or []
    except Exception:
        return {"signals": [], "matching_text": [], "ocr_text_excerpt": ""}
    joined = ocr_text_from_regions(regions)
    verdict = interpret_text(joined)
    signals = [s for s in verdict["signals"] if s != "docs_context"]
    if signals:
        signals = ["ocr_human_verification_text"] + [
            s for s in signals if s not in ("ocr_human_verification_text",)
        ]
    # Keep raw OCR phrase evidence namespaced.
    return {
        "signals": sorted(set(signals)),
        "matching_text": list(verdict.get("matching_text", [])),
        "ocr_text_excerpt": joined[:500],
    }
