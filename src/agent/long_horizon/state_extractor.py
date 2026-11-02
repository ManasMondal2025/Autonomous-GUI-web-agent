"""Visual State Signature: screenshot -> structured, comparable evidence.

Uses ONLY existing components: OmniParserDetector (icon/UI boxes),
PaddleOCRBackend (text regions), DOMExtractor output (when available).
No new VLM, no new detector. All geometry normalized to 0..1000 space so
signatures compare across resolutions.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

NORM = 1000.0
ROW_BAND = 40          # px (at 1000-scale) for reading-order row grouping
NEAR_PX = 150          # max gap for layout adjacency statements
LARGE_FRACTION = 0.15  # element area fraction flagged as large_region


def _norm_box_xyxy(box_xyxy, w: float, h: float) -> List[int]:
    x1, y1, x2, y2 = box_xyxy
    sx, sy = NORM / max(1.0, w), NORM / max(1.0, h)
    return [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)]


def _to_xyxy(bbox: Any) -> Optional[List[float]]:
    """Accept DOM/OCR dict {x,y,width,height} or [x1,y1,x2,y2] list."""
    try:
        if isinstance(bbox, dict):
            x, y = float(bbox.get("x", 0)), float(bbox.get("y", 0))
            return [x, y, x + float(bbox.get("width", 0)),
                    y + float(bbox.get("height", 0))]
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = (float(v) for v in bbox)
            if x2 > x1 and y2 > y1:
                return [x1, y1, x2, y2]
    except (TypeError, ValueError):
        pass
    return None


def _norm_text(t: Any) -> str:
    return " ".join(str(t or "").lower().split())


def extract_signature(image, dom_elements: Optional[List[Dict[str, Any]]] = None,
                      detector=None, ocr=None,
                      min_confidence: float = 0.0) -> Dict[str, Any]:
    """Build a Visual State Signature for one screenshot.

    detector/ocr are optional live backends (OmniParserDetector /
    PaddleOCRBackend instances); when None those channels are skipped with
    a log line (never fatal). dom_elements uses the observer's already
    extracted list when available.
    """
    from PIL import Image as _Image
    if not isinstance(image, _Image.Image):
        raise TypeError(f"extract_signature needs a PIL Image, got {type(image)}")
    W, H = image.size

    texts: List[str] = []
    text_boxes: List[List[int]] = []
    ui_elements: List[Dict[str, Any]] = []

    for el in dom_elements or []:
        t = _norm_text(el.get("text") or el.get("ariaLabel")
                       or el.get("placeholder") or el.get("value", ""))
        bb = _to_xyxy(el.get("bbox", {}))
        if not bb:
            continue
        if t:
            texts.append(t)
            text_boxes.append(_norm_box_xyxy(bb, W, H))
        ui_elements.append({
            "label": str(el.get("tag", "")) or "element",
            "box": _norm_box_xyxy(bb, W, H),
            "source": "dom"})

    if detector is not None:
        try:
            for el in detector.detect(image) or []:
                bb = _to_xyxy((el.get("bbox") or {}))
                if not bb:
                    # detector may already emit xyxy lists
                    bb = _to_xyxy(el.get("bbox"))
                if not bb:
                    continue
                ui_elements.append({
                    "label": str(el.get("tag", "")) or "icon",
                    "box": _norm_box_xyxy(bb, W, H),
                    "source": "detector"})
        except Exception as e:
            logger.warning(f"State extractor: detector failed: {e}")

    if ocr is not None:
        try:
            for reg in ocr.detect_text(image) or []:
                t = _norm_text(reg.get("text", ""))
                if not t:
                    continue
                try:
                    if float(reg.get("score", 1.0)) < min_confidence:
                        continue
                except (TypeError, ValueError):
                    pass
                bb = _to_xyxy(reg.get("bbox"))
                if not bb:
                    continue
                # DOM text already captured above; keep OCR boxes but avoid
                # double-counting identical strings in `texts`.
                if t not in texts:
                    texts.append(t)
                text_boxes.append(_norm_box_xyxy(bb, W, H))
        except Exception as e:
            logger.warning(f"State extractor: OCR failed: {e}")

    # Reading order + row groups (deterministic layout summary).
    order = sorted(range(len(text_boxes)),
                   key=lambda i: (text_boxes[i][1] // ROW_BAND,
                                  text_boxes[i][0]))
    rows: List[List[str]] = []
    last_band = None
    for i in order:
        band = text_boxes[i][1] // ROW_BAND
        try:
            t = texts[i]
        except IndexError:
            t = ""
        if band != last_band:
            rows.append([])
            last_band = band
        rows[-1].append(t)
    layout_relations = [f"row_{r}: " + " | ".join(t for t in row if t)
                        for r, row in enumerate(rows)]

    # Important regions: 5 largest text boxes + large element boxes.
    areas = sorted((((b[2] - b[0]) * (b[3] - b[1]), b)
                    for b in text_boxes), reverse=True)
    important = [{"kind": "text", "box": b} for _, b in areas[:5]]
    for el in ui_elements:
        b = el["box"]
        if (b[2] - b[0]) * (b[3] - b[1]) >= LARGE_FRACTION * NORM * NORM:
            important.append({"kind": "large_region",
                              "label": el["label"], "box": b})

    return {"texts": texts, "text_boxes": text_boxes,
            "ui_elements": ui_elements,
            "element_boxes": [e["box"] for e in ui_elements],
            "layout_relations": layout_relations,
            "important_regions": important,
            "image_size": [W, H]}


def compare_signatures(before: Dict[str, Any],
                       after: Dict[str, Any]) -> Dict[str, Any]:
    """Diff two signatures: added/removed texts + elements, layout change."""
    tb, ta = set(before.get("texts", [])), set(after.get("texts", []))
    lb = sorted(f"{e.get('label')}:{tuple(e.get('box', []))}"
                for e in before.get("ui_elements", []))
    la = sorted(f"{e.get('label')}:{tuple(e.get('box', []))}"
                for e in after.get("ui_elements", []))
    return {
        "added_texts": sorted(ta - tb),
        "removed_texts": sorted(tb - ta),
        "added_elements": sorted(set(la) - set(lb)),
        "removed_elements": sorted(set(lb) - set(la)),
        "layout_changed": (before.get("layout_relations")
                           != after.get("layout_relations")),
    }
