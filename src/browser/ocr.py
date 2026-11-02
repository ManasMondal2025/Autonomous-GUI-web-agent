"""OCR-based semantic enrichment (PaddleOCR backend, optional dependency).

One OCR pass over the full screenshot produces text regions; regions are
then spatially associated with OmniParser proposals. Detector bboxes are
never altered. No DOM attributes are fabricated.

paddleocr/paddlepaddle are imported lazily so the project runs without OCR.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

def poly_to_xywh(poly) -> Dict[str, float]:
    """Axis-aligned xywh box enclosing a quad point list (list or ndarray)."""
    try:
        import numpy as np
        pts = np.asarray(poly, dtype=float).reshape(-1, 2)
        x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
    except Exception:
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return {"x": x1, "y": y1, "width": max(0.0, x2 - x1),
            "height": max(0.0, y2 - y1)}


def normalize_ocr_output(raw) -> List[Dict[str, Any]]:
    """PaddleOCR 3.x predict() output -> [{text, bbox, score}].

    Accepts the list-of-result-dicts format (rec_texts/rec_scores with
    rec_polys or rec_boxes). Entries are aligned by index; mismatched
    lengths are truncated to the shortest list (deterministic).
    """
    regions: List[Dict[str, Any]] = []
    pages = raw if isinstance(raw, list) else [raw]
    for page in pages:
        if not isinstance(page, dict):
            continue
        texts = page.get("rec_texts") or []
        scores = page.get("rec_scores") or []
        polys = page.get("rec_polys")
        boxes = page.get("rec_boxes")
        geoms = polys if polys is not None else (boxes if boxes is not None else [])
        n = min(len(texts), len(scores), len(geoms))
        for i in range(n):
            try:
                g = geoms[i]
                if hasattr(g, "tolist"):
                    g = g.tolist()
                # rec_boxes are [x1,y1,x2,y2]; rec_polys are quads.
                if len(g) == 4 and all(isinstance(v, (int, float)) for v in g):
                    x1, y1, x2, y2 = (float(v) for v in g)
                    bbox = {"x": x1, "y": y1, "width": max(0.0, x2 - x1),
                            "height": max(0.0, y2 - y1)}
                else:
                    bbox = poly_to_xywh(g)
                regions.append({
                    "text": str(texts[i]),
                    "bbox": bbox,
                    "score": float(scores[i]),
                })
            except Exception as e:
                logger.debug("Skipping malformed OCR entry %d: %s", i, e)
                continue
    return regions


# --------------------------------------------------------------------------
# Backend abstraction
# --------------------------------------------------------------------------

class BaseOCR:
    """OCR backend interface: screenshot in, text regions out."""

    name = "base-ocr"

    def detect_text(self, screenshot) -> List[Dict[str, Any]]:
        raise NotImplementedError


class PaddleOCRBackend(BaseOCR):
    """PaddleOCR backend (lazy import; optional dependency)."""

    name = "paddleocr"

    def __init__(self, lang: str = "en",
                 min_confidence: float = 0.0, **kwargs):
        self.lang = lang
        self.min_confidence = float(min_confidence)
        self.config = kwargs
        self._ocr = None

    def _load(self):
        if self._ocr is not None:
            return
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise ImportError(
                "PaddleOCR backend requires `paddleocr` + `paddlepaddle`. "
                "Install with: pip install paddleocr paddlepaddle"
            ) from e
        # Model dirs (or other PaddleOCR kwargs) pass through self.config,
        # e.g. text_detection_model_dir=... for pinned offline models.
        self._ocr = PaddleOCR(lang=self.lang, **self.config)
        logger.info("[OCR] PaddleOCR loaded (lang=%s)", self.lang)

    def detect_text(self, screenshot) -> List[Dict[str, Any]]:
        import numpy as np
        from PIL import Image as _Image

        self._load()
        if isinstance(screenshot, _Image.Image):
            arr = np.array(screenshot.convert("RGB"))
        else:
            arr = np.array(screenshot)
        regions = normalize_ocr_output(self._ocr.predict(arr))
        kept = [r for r in regions
                if r["text"].strip() and r["score"] >= self.min_confidence]
        logger.info("[OCR] %d/%d text regions kept (min_conf=%.2f)",
                    len(kept), len(regions), self.min_confidence)
        return kept


# --------------------------------------------------------------------------
# Spatial association + enrichment
# --------------------------------------------------------------------------

def _pad_box(bbox: Dict[str, float], pad: float) -> Dict[str, float]:
    return {"x": bbox.get("x", 0) - pad, "y": bbox.get("y", 0) - pad,
            "width": bbox.get("width", 0) + 2 * pad,
            "height": bbox.get("height", 0) + 2 * pad}


def _center_in(box: Dict[str, float], outer: Dict[str, float]) -> bool:
    cx = box.get("x", 0) + box.get("width", 0) / 2
    cy = box.get("y", 0) + box.get("height", 0) / 2
    return (outer.get("x", 0) <= cx <= outer.get("x", 0) + outer.get("width", 0)
            and outer.get("y", 0) <= cy <= outer.get("y", 0) + outer.get("height", 0))


def _inclusion_ratio(inner: Dict[str, float], outer: Dict[str, float]) -> float:
    """Area(inner ∩ outer) / Area(inner); 1.0 = fully contained."""
    try:
        ix1 = max(inner.get("x", 0), outer.get("x", 0))
        iy1 = max(inner.get("y", 0), outer.get("y", 0))
        ix2 = min(inner.get("x", 0) + inner.get("width", 0),
                  outer.get("x", 0) + outer.get("width", 0))
        iy2 = min(inner.get("y", 0) + inner.get("height", 0),
                  outer.get("y", 0) + outer.get("height", 0))
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area = max(1e-9, float(inner.get("width", 0)) * float(inner.get("height", 0)))
        return inter / area
    except (TypeError, ValueError):
        return 0.0


def _axis_overlap(a1: float, a2: float, b1: float, b2: float) -> float:
    """Overlap length of [a1,a2] and [b1,b2] as fraction of the smaller."""
    overlap = max(0.0, min(a2, b2) - max(a1, b1))
    denom = max(1e-9, min(a2 - a1, b2 - b1))
    return overlap / denom


def _gap(a1: float, a2: float, b1: float, b2: float) -> float:
    """Non-negative gap between intervals (0 when overlapping)."""
    return max(0.0, max(a1, b1) - min(a2, b2))


def associate_ocr_to_proposals(
    proposals: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
    pad: float = 4.0,
    min_iou: float = 0.1,
    row_band: float = 10.0,
    img_size=None,
    inclusion_threshold: float = 0.85,
) -> Dict[int, List[Dict[str, Any]]]:
    """Map proposal index -> claimed OCR regions (exclusive, deterministic).

    A region is claimed when ANY of these hold (checked in order):
      1. its center lies inside the padded proposal box, OR
      2. its IoU with the padded box >= min_iou, OR
      3. its inclusion ratio in the padded box >= inclusion_threshold
         (text fully inside a widget boundary), OR
      4. adjacency (only when img_size=(w,h) is given): y-overlap > 40%
         with x-gap < 2% of screen width (horizontal neighbor), OR
         x-overlap > 25% with y-gap < max(2% of screen height, 15px)
         (vertical neighbor). Gaps must be positive (no overlap).
    Rules 1-2 preserve the historical behavior exactly when img_size is
    None. Claimed regions are ordered by reading order (row band, then x)
    and combined downstream. Never mutates inputs.
    """
    from utils.coordinate_utils import bbox_iou

    claimed: Dict[int, List[Dict[str, Any]]] = {}
    taken = set()
    img_w, img_h = img_size if img_size else (None, None)
    for idx, prop in enumerate(proposals):
        padded = _pad_box(prop.get("bbox", {}), pad)
        matched = []
        for r_idx, reg in enumerate(regions):
            if r_idx in taken:
                continue
            rbox = reg.get("bbox", {})
            try:
                hit = _center_in(rbox, padded) or \
                    bbox_iou(rbox, padded) >= min_iou or \
                    _inclusion_ratio(rbox, padded) >= inclusion_threshold
                if not hit and img_w and img_h:
                    px, py, pw, ph = (padded.get("x", 0), padded.get("y", 0),
                                      padded.get("width", 0), padded.get("height", 0))
                    rx, ry, rw, rh = (rbox.get("x", 0), rbox.get("y", 0),
                                      rbox.get("width", 0), rbox.get("height", 0))
                    y_ov = _axis_overlap(py, py + ph, ry, ry + rh)
                    x_gap = _gap(px, px + pw, rx, rx + rw)
                    if y_ov > 0.40 and 0 < x_gap < 0.02 * img_w:
                        hit = True
                    else:
                        x_ov = _axis_overlap(px, px + pw, rx, rx + rw)
                        y_gap = _gap(py, py + ph, ry, ry + rh)
                        if x_ov > 0.25 and \
                                0 < y_gap < max(0.02 * img_h, 15.0):
                            hit = True
            except (TypeError, ValueError):
                hit = False
            if hit:
                matched.append((r_idx, reg))
                taken.add(r_idx)
        matched.sort(key=lambda t: (
            int((t[1].get("bbox", {}).get("y", 0)
                 + t[1].get("bbox", {}).get("height", 0) / 2) // row_band),
            t[1].get("bbox", {}).get("x", 0)))
        claimed[idx] = [reg for _, reg in matched]
    return claimed


def edge_support_score(image, bbox: Dict[str, float]) -> float:
    """Fraction of Canny edge pixels inside bbox (0..1).

    Glyph strokes are edge-dense; flat paint (even dark blue buttons) is
    not. Use as geometry validation, never as text content. Returns 0.0
    for degenerate boxes instead of raising.
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image as _Image

        arr = np.array(image.convert("L")) if isinstance(image, _Image.Image) \
            else np.asarray(image)
        h, w = arr.shape[:2]
        x1 = max(0, int(bbox.get("x", 0)))
        y1 = max(0, int(bbox.get("y", 0)))
        x2 = min(w, int(bbox.get("x", 0) + bbox.get("width", 0)))
        y2 = min(h, int(bbox.get("y", 0) + bbox.get("height", 0)))
        if x2 <= x1 + 1 or y2 <= y1 + 1:
            return 0.0
        edges = cv2.Canny(arr[y1:y2, x1:x2], 100, 200)
        return float((edges > 0).mean())
    except Exception as e:
        logger.debug("edge_support failed: %s", e)
        return 0.0


def match_text(target: str, candidate: str, threshold: float = 0.8) -> bool:
    """Deterministic text match: normalized containment either direction,
    else difflib ratio >= threshold. Empty strings never match."""
    import difflib

    def norm(s):
        return " ".join(str(s).lower().split())

    t, c = norm(target), norm(candidate)
    if not t or not c:
        return False
    if t in c or c in t:
        return True
    return difflib.SequenceMatcher(None, t, c).ratio() >= threshold


def relocalize_text_region(image, ocr_backend, text: str,
                           rough_bbox: Dict[str, float],
                           expand: float = 3.0,
                           match_threshold: float = 0.8,
                           min_edge: float = 0.02):
    """Re-find `text` near `rough_bbox`; return (corrected_bbox, edge, detail).

    Expands the rough box (association-only prior, clamped to the image),
    re-runs OCR detection on the crop, and keeps the best fuzzy-matching
    region mapped back to original coordinates — but ONLY if its own edge
    support passes. Returns (None, edge, detail) when nothing validates.
    Deterministic: highest score wins; ties break by area, then x, y.
    """
    from PIL import Image as _Image

    W, H = image.size if isinstance(image, _Image.Image) else (None, None)
    if W is None:
        return None, 0.0, {"reason": "unknown image size"}
    cx = rough_bbox.get("x", 0) + rough_bbox.get("width", 0) / 2
    cy = rough_bbox.get("y", 0) + rough_bbox.get("height", 0) / 2
    ew, eh = rough_bbox.get("width", 0) * expand, \
        rough_bbox.get("height", 0) * expand
    x1 = max(0, int(cx - ew / 2))
    y1 = max(0, int(cy - eh / 2))
    x2 = min(W, int(cx + ew / 2))
    y2 = min(H, int(cy + eh / 2))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None, 0.0, {"reason": "degenerate search crop"}
    crop = image.crop((x1, y1, x2, y2))
    try:
        cands = ocr_backend.detect_text(crop)
    except Exception as e:
        return None, 0.0, {"reason": f"crop redetect failed: {e}"}
    matches = [c for c in cands if match_text(text, c.get("text", ""),
                                              threshold=match_threshold)]
    if not matches:
        return None, 0.0, {"reason": "no fuzzy match in crop",
                           "n_candidates": len(cands)}

    def rank(c):
        b = c.get("bbox", {})
        return (-float(c.get("score", 0.0)),
                float(b.get("width", 0)) * float(b.get("height", 0)),
                float(b.get("x", 0)), float(b.get("y", 0)))

    best = sorted(matches, key=rank)[0]
    bb = best.get("bbox", {})
    corrected = {"x": bb.get("x", 0) + x1, "y": bb.get("y", 0) + y1,
                 "width": bb.get("width", 0), "height": bb.get("height", 0)}
    edge = edge_support_score(image, corrected)
    if edge < min_edge:
        return None, edge, {"reason": "relocalized box failed edge gate",
                            "box": corrected}
    return corrected, edge, {"box": corrected,
                             "matched_text": best.get("text", "")}


def validate_and_recover_regions(image, ocr_backend,
                                 regions: List[Dict[str, Any]],
                                 expand: float = 3.0,
                                 min_edge: float = 0.02,
                                 match_threshold: float = 0.8):
    """Split regions into (usable, unresolved, log).

    A region is usable if its own box passes the edge gate, or if
    relocalization finds an edge-validated box for the same text. Otherwise
    it is unresolved (kept out of association entirely — never forced).
    Deterministic; inputs never mutated.
    """
    usable, unresolved, log = [], [], []
    for reg in regions:
        entry = dict(reg)
        edge = edge_support_score(image, entry.get("bbox", {}))
        entry["edge_support"] = round(edge, 4)
        entry["recovered"] = False
        if edge >= min_edge:
            usable.append(entry)
            log.append({"text": entry.get("text", "")[:40],
                        "decision": "accept-original", "edge": round(edge, 4)})
            continue
        corrected, new_edge, detail = relocalize_text_region(
            image, ocr_backend, entry.get("text", ""), entry.get("bbox", {}),
            expand=expand, match_threshold=match_threshold, min_edge=min_edge)
        if corrected is not None:
            entry["bbox"] = corrected
            entry["edge_support"] = round(new_edge, 4)
            entry["recovered"] = True
            usable.append(entry)
            log.append({"text": entry.get("text", "")[:40],
                        "decision": "accept-relocalized",
                        "edge": round(new_edge, 4),
                        "orig_bbox": reg.get("bbox"), "new_bbox": corrected})
        else:
            unresolved.append(entry)
            log.append({"text": entry.get("text", "")[:40],
                        "decision": "unresolved",
                        "reason": detail.get("reason")})
    return usable, unresolved, log


def enrich_proposals_with_ocr(
    proposals: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
    pad: float = 4.0,
    min_iou: float = 0.1,
    row_band: float = 10.0,
    verbose: bool = False,
    img_size=None,
    image=None,
    ocr_backend=None,
    recover_text_geometry: bool = False,
    min_edge: float = 0.02,
    expand: float = 3.0,
) -> List[Dict[str, Any]]:
    """Return enriched COPIES: OCR text fills empty proposal text only.

    Precedence (deterministic): matched OCR text wins when non-empty;
    existing visual text/caption is kept when OCR matched nothing.
    Provenance becomes "omniparser+ocr" only where OCR text was assigned.
    Bboxes are never altered. Inputs are never mutated.

    When recover_text_geometry=True (requires image + ocr_backend), regions
    first pass through validate_and_recover_regions: edge-invalid boxes are
    relocalized, unrecoverable ones are excluded from association. Default
    False preserves the exact historical behavior.
    """
    if recover_text_geometry:
        if image is None or ocr_backend is None:
            raise ValueError("recover_text_geometry=True requires image and "
                             "ocr_backend")
        regions, unresolved, recovery_log = validate_and_recover_regions(
            image, ocr_backend, regions, expand=expand, min_edge=min_edge)
        logger.info("[OCR] recovery: %d usable, %d unresolved of %d regions",
                    len(regions), len(unresolved),
                    len(regions) + len(unresolved))
        for item in recovery_log:
            logger.debug("[OCR] recover %s", item)
    claimed = associate_ocr_to_proposals(proposals, regions, pad=pad,
                                         min_iou=min_iou, row_band=row_band,
                                         img_size=img_size)
    enriched = []
    for idx, prop in enumerate(proposals):
        entry = dict(prop)
        matched = claimed.get(idx, [])
        combined = " ".join(r.get("text", "").strip() for r in matched).strip()
        if combined:
            entry["text"] = combined
            entry["source"] = "omniparser+ocr"
        if verbose:
            bb = prop.get("bbox", {})
            logger.info(
                "proposal %s:\n  omni_bbox = (%s)\n  ocr_text = %r\n"
                "  ocr_bbox = (%s)\n  association = %s\n  final_text = %r",
                prop.get("id", idx), bb,
                combined or "(none)",
                matched[0].get("bbox") if matched else "(none)",
                "matched" if matched else "no-match",
                entry.get("text", ""))
        else:
            logger.debug("proposal %s: %d OCR region(s) -> %r",
                         prop.get("id", idx), len(matched),
                         entry.get("text", ""))
        enriched.append(entry)
    return enriched
