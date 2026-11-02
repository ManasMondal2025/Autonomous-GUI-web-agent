#!/usr/bin/env python3
"""OmniParser -> bbox crop -> OCR pipeline for exactly ONE screenshot.

                    Ocr/input.png
                         |
                         v
                    OmniParser          (YOLOv8 icon detect, no captions)
                         |
                         v
               ALL UI ELEMENT BBOXES
                         |
              +----------+----------+
              |                     |
              v                     v
         bboxes.json         bbox_visualization.png
              |
              v
       padded bbox crops
              |
              v
          crops/*.png       (original resolution, saved unchanged)
              |
              v
       OCR ON EACH CROP     (PaddleOCR or EasyOCR -- never the full
                             screenshot, never a VLM)
              |
              v
       ocr_results.json

Run:
    python ocr_pipeline.py
    python ocr_pipeline.py --input Ocr/input.png

Real APIs used (nothing invented):
  * OmniParser detection: src/browser/proposers.py::OmniParserDetector
    (ultralytics YOLO, weights pretrained/omniparser/weights/icon_detect/
    model.pt) with with_captions=False -- Florence-2 captioning is a VLM
    and this pipeline performs OCR separately. Falls back to direct
    ultralytics.YOLO with the identical weights and call pattern when the
    repo module is not importable.
  * PaddleOCR 3.x: PaddleOCR(lang="en").predict(<numpy RGB array>).
  * EasyOCR: easyocr.Reader(['en']).readtext(<numpy array>).
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration (top of file; CLI flags override each of these)
# --------------------------------------------------------------------------

INPUT_IMAGE = "Ocr/input.png"
# Detector backend: "omniparser" (default, YOLOv8 icon detect, captions
# off) or "gpa" (GPA Detector + gpa_postprocess.py stages). The OCR stages
# below are identical on both paths.
DETECTOR = "omniparser"
# NOTE: sections 7/8/12/14 write paths as "Ocr/...", but section 3 fixes the
# full output tree under Ocr/test_2/ (including a copy of input.png), which
# also keeps this run isolated from other files in Ocr/. The output dir is
# therefore Ocr/test_2 by default; --output overrides it.
OUTPUT_DIR = "Ocr/test_2"
PADDING_RATIO = 0.05
OCR_ENGINE = "paddleocr"          # "paddleocr" (preferred) or "easyocr"
OCR_UPSCALE = 4                   # upscale factor for the OCR *copy* only
# Proposal sources: "omni" (OmniParser boxes only, default) or "omni+ocr"
# (adds tight OCR text-region boxes so labels/placeholders/inputs are
# covered, tightens widget boxes to content, drops giant containers).
PROPOSALS = "omni"
HYBRID_CONTAINER_MIN_CHILDREN = 3  # widget dies if it contains >=N texts
HYBRID_TEXT_IN_WIDGET_IOS = 0.5    # text dies if >this frac inside a widget
HYBRID_MIN_TEXT_SIZE = 3           # min text-box width/height in px
OMNIPARSER_MODEL_DIR = None       # None -> repo default (or env var)
BOX_THRESHOLD = 0.05
# NMS: greedy suppression on validated boxes (highest confidence wins).
# A box is suppressed when IoU > NMS_IOU_THRESHOLD vs a kept box, or when
# it is mostly *contained* in a LARGER kept box (intersection over the
# smaller area > NMS_CONTAIN_THRESHOLD). The size guard means small boxes
# can be absorbed into their containers, but a container is never killed
# by its own child. None disables a rule; --no-nms disables both.
NMS_IOU_THRESHOLD = 0.5
NMS_CONTAIN_THRESHOLD = 0.5
LOG_LEVEL = "INFO"

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SCRIPT_DIR = Path(__file__).resolve().parent

# stdlib `logging` is imported BEFORE src/ joins sys.path: this repo ships a
# src/logging/ package that would otherwise shadow it.
log = logging.getLogger("ocr_pipeline")


def _setup_logging(level: str = LOG_LEVEL) -> None:
    logging.basicConfig(level=getattr(logging, str(level).upper(),
                                      logging.INFO),
                        format="%(levelname)s: %(message)s")


def _resolve_input(arg: str) -> Path:
    """Resolve the input path: as-given (cwd) first, then vs script dir."""
    p = Path(arg)
    if p.is_file():
        return p
    alt = SCRIPT_DIR / arg
    if alt.is_file():
        return alt
    return p  # let load_image() raise the not-found error


# --------------------------------------------------------------------------
# 1. Input
# --------------------------------------------------------------------------

def load_image(path: str):
    """Load Ocr/input.png at ORIGINAL resolution (never resized)."""
    from PIL import Image

    p = _resolve_input(path)
    if not p.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format {p.suffix!r}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    img = Image.open(p).convert("RGB")
    print(f"Input image: {path}")
    print(f"Image size: {img.size[0]} x {img.size[1]}")
    log.info("Image size: %d x %d", img.size[0], img.size[1])
    return img, p


# --------------------------------------------------------------------------
# 2. OmniParser (actual installed implementation)
# --------------------------------------------------------------------------

def load_omniparser(model_dir=None, device: str = "auto",
                    box_threshold: float = BOX_THRESHOLD):
    """Load the REAL OmniParser detector.

    Preferred: repo OmniParserDetector (ultralytics YOLOv8 icon detect)
    with with_captions=False. Fallback: direct ultralytics.YOLO with the
    identical weights and predict() call pattern.
    """
    if model_dir is None:
        model_dir = os.getenv("OMNIPARSER_MODEL_DIR", None)
    try:
        sys.path.insert(0, str(SCRIPT_DIR / "src"))
        from browser.proposers import OmniParserDetector
        log.info("Loading OmniParser (repo OmniParserDetector)...")
        detector = OmniParserDetector(
            model_dir=model_dir, device=device,
            box_threshold=box_threshold, with_captions=False)
        detector._load_detect()
        if detector._detect_model is None:
            raise RuntimeError(detector._detect_error
                               or "icon_detect failed to load")
        log.info("OmniParser icon_detect loaded from: %s",
                 detector.model_dir)
        return ("repo", detector)
    except Exception as e:
        log.warning("Repo OmniParserDetector unavailable (%s); using "
                    "direct ultralytics.YOLO with the same weights.", e)
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "OmniParser detection requires `ultralytics` (and torch). "
            "Install with: pip install ultralytics torch") from e
    if model_dir is None:
        model_dir = str(SCRIPT_DIR / "pretrained" / "omniparser")
    pt = Path(str(model_dir)) / "weights" / "icon_detect" / "model.pt"
    if not pt.is_file():
        raise RuntimeError(
            f"[OmniParser] missing weight file: {pt}. Set --model-dir or "
            f"OMNIPARSER_MODEL_DIR.")
    log.info("Loading OmniParser (direct YOLO) from: %s", pt)
    try:
        import torch
        if device == "auto":
            if torch.cuda.is_available():
                yolo_device = "cuda:0"
            elif torch.backends.mps.is_available():
                yolo_device = "mps"
            else:
                yolo_device = "cpu"
        elif device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("device='cuda' requested but CUDA is "
                                   "unavailable. Use --cpu.")
            yolo_device = "cuda:0"
        elif device == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("device='mps' requested but no MPS "
                                   "backend. Use --cpu.")
            yolo_device = "mps"
        elif device == "cpu":
            yolo_device = "cpu"
        else:
            raise ValueError(f"unknown device {device!r}")
    except ImportError:
        yolo_device = "cpu"
    return ("direct", {"model": YOLO(str(pt)), "device": yolo_device,
                       "conf": box_threshold})


def run_omniparser(detector_handle, image):
    """Run detection on the ORIGINAL-resolution image.

    Returns raw dicts with bbox {x, y, width, height} in screenshot pixels
    (ultralytics letterboxes internally and maps boxes back to the input
    image size -- see convert_bbox_to_original).
    """
    kind, det = detector_handle
    if kind == "repo":
        return det.detect(image)
    model, yolo_device, conf = (det["model"], det["device"], det["conf"])
    results = model.predict(image, conf=conf, verbose=False,
                            device=yolo_device)
    elements = []
    if results:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), score in zip(xyxy, scores):
                w, h = float(x2 - x1), float(y2 - y1)
                if w <= 1 or h <= 1:
                    continue
                elements.append({
                    "bbox": {"x": float(x1), "y": float(y1),
                             "width": w, "height": h},
                    "tag": "icon",
                    "text": "",
                    "score": float(score),
                    "source": "omniparser",
                })
    return elements


def run_gpa_detection(image, image_size, device="auto"):
    """GPA detector + gpa_postprocess.py stages (validated in test_14/16).

    Imports the post-processing module from this repo dir and runs its
    exact stage chain (confidence -> exact dup -> parent removal ->
    overlap merge -> final cleanup) with that module's validated
    defaults. Returns (elements, n_raw). OmniParser NMS/hybrid steps are
    skipped on this path. No OCR here -- crops/OCR below run unchanged.
    """
    import sys as _sys

    if str(SCRIPT_DIR) not in _sys.path:
        _sys.path.insert(0, str(SCRIPT_DIR))
    import gpa_postprocess as gp

    raw = gp.run_gpa_detector(
        image, gp.GPA_MODEL, min(gp.DETECTOR_CONF, gp.CONF_THRESHOLD),
        gp.DETECTOR_INTERNAL_NMS_IOU, device)
    recs, s1 = gp.stage_filter(raw, image_size, gp.CONF_THRESHOLD)
    s2 = gp.stage_exact_dup(s1)
    s3 = gp.stage_remove_parents(s2)
    s4 = gp.stage_merge(s3)
    s5 = gp.stage_final_cleanup(s4)
    log.info("GPA post-processing: %d raw -> %d final.",
             len(raw), len(s5))
    elements = []
    for j, r in enumerate(sorted(s5, key=lambda r: (-r["confidence"],
                                                    r["raw_id"])), start=1):
        elements.append({"id": j, "bbox": [int(v) for v in r["bbox"]],
                         "confidence": r["confidence"],
                         "label": r.get("label") or "icon"})
    return elements, len(raw)


def normalize_detections(raw_elements):
    """Raw OmniParser dicts -> [{id, bbox [x1,y1,x2,y2], confidence,
    label}]. The icon model is single-class, so label "icon" is honest."""
    normalized = []
    for i, raw in enumerate(raw_elements or [], start=1):
        raw = dict(raw or {})
        bb = raw.get("bbox") or {}
        try:
            x = float(bb["x"])
            y = float(bb["y"])
            w = float(bb["width"])
            h = float(bb["height"])
        except (KeyError, TypeError, ValueError):
            log.warning("Element %d has malformed bbox %r; skipping.", i,
                        bb)
            continue
        normalized.append({
            "id": i,
            "bbox": [x, y, x + w, y + h],
            "confidence": raw.get("score"),
            "label": raw.get("tag") or "icon",
        })
    return normalized


def convert_bbox_to_original(bbox_xyxy, image_size,
                             coord_space: str = "pixels"):
    """Map a detection bbox to ORIGINAL input.png pixel coordinates.

    "pixels" (default): ultralytics already returns original-image pixels,
    so this is a pass-through. "normalized-1000": scale 0-1000 to pixels.
    Anything else raises -- coordinates are never silently reinterpreted.
    """
    W, H = image_size
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    if coord_space == "pixels":
        return [x1, y1, x2, y2]
    if coord_space == "normalized-1000":
        return [x1 / 1000.0 * W, y1 / 1000.0 * H,
                x2 / 1000.0 * W, y2 / 1000.0 * H]
    raise ValueError(f"unknown coord_space {coord_space!r}")


def validate_bbox(bbox_xyxy, image_size):
    """Enforce x1 < x2, y1 < y2, 0 <= x1 < W, 0 <= y1 < H.

    Clips to the image; rounds to int pixels. Returns (ok, bbox, reason);
    invalid/zero-area boxes are rejected with a reason, never silently.
    """
    import math

    W, H = image_size
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    except (TypeError, ValueError):
        return False, None, "non-numeric coordinates"
    if any(math.isnan(v) or math.isinf(v) for v in (x1, y1, x2, y2)):
        return False, None, "NaN/inf coordinates"
    if not (x1 < x2 and y1 < y2):
        return False, None, "negative or zero width/height"
    ix1, iy1 = max(0, int(round(x1))), max(0, int(round(y1)))
    ix2, iy2 = min(W, int(round(x2))), min(H, int(round(y2)))
    if not (0 <= ix1 < ix2 <= W and 0 <= iy1 < iy2 <= H):
        return False, None, "empty after clipping to image boundaries"
    return True, [ix1, iy1, ix2, iy2], "ok"


def bbox_iou(a, b) -> float:
    """IoU of two [x1, y1, x2, y2] int boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def apply_nms(elements, iou_threshold: float,
              contain_threshold: float | None = None,
              prefer: str = "big"):
    """Greedy NMS on validated boxes: highest confidence first.

    Suppresses a candidate vs a kept box when
      * IoU > iou_threshold (symmetric: first-kept wins), or
      * containment: intersection / min(area) > contain_threshold.
        prefer="big" (omni mode): a kept box absorbs a smaller candidate;
        a container is never killed by its child. prefer="small" (hybrid
        mode): the tighter box wins -- a contained candidate displaces its
        bigger kept box, and a bigger candidate dies to an already-kept
        smaller box. Use "small" only after uninformative containers are
        gone (hybrid merge), otherwise small specks would eat real widgets.
    Returns (kept, suppressed_count). Deterministic. A None threshold
    disables that rule.
    """
    order = sorted(range(len(elements)),
                   key=lambda k: (-(elements[k].get("confidence") or 0.0),
                                  k))
    kept, suppressed = [], 0
    for k in order:
        cand = elements[k]
        cb = cand["bbox"]
        c_area = max(1, (cb[2] - cb[0]) * (cb[3] - cb[1]))
        drop, reason, displace = False, "", None
        for qi, q in enumerate(kept):
            qb = q["bbox"]
            ix1, iy1 = max(cb[0], qb[0]), max(cb[1], qb[1])
            ix2, iy2 = min(cb[2], qb[2]), min(cb[3], qb[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            q_area = max(1, (qb[2] - qb[0]) * (qb[3] - qb[1]))
            union = c_area + q_area - inter
            v = inter / union if union > 0 else 0.0
            if iou_threshold is not None and v > iou_threshold:
                drop, reason = True, f"iou={v:.3f}>{iou_threshold}"
                break
            ios = inter / min(c_area, q_area)
            if (contain_threshold is not None and ios > contain_threshold
                    and inter > 0):
                if prefer == "big" and q_area >= c_area:
                    drop, reason = True, \
                        f"contained ios={ios:.3f}>{contain_threshold}"
                    break
                if prefer == "small" and c_area < q_area:
                    # Tighter candidate displaces its bigger kept box.
                    displace, reason = qi, \
                        f"tighter wins ios={ios:.3f}>{contain_threshold}"
                    break
                if prefer == "small" and c_area >= q_area:
                    drop, reason = True, \
                        f"contained ios={ios:.3f}>{contain_threshold}"
                    break
        if displace is not None:
            old = kept.pop(displace)
            log.info("NMS displacing element %d bbox %r with tighter "
                     "element %d bbox %r (%s).",
                     old["id"], old["bbox"], cand["id"], cb, reason)
            suppressed += 1
            kept.append(cand)
            continue
        if drop:
            log.info("NMS suppressing element %d bbox %r "
                     "(conf=%.3f, %s).",
                     cand["id"], cb, cand.get("confidence") or 0.0,
                     reason)
            suppressed += 1
            continue
        kept.append(cand)
    return kept, suppressed


def save_bboxes(path: Path, elements) -> None:
    """Write bboxes.json in ORIGINAL input.png coordinates."""
    payload = [{"id": e["id"], "bbox": e["bbox"],
                "confidence": e["confidence"], "label": e["label"]}
               for e in elements]
    path.write_text(json.dumps(payload, indent=4))
    log.info("Saving bboxes... %s (%d boxes)", path, len(payload))


def create_bbox_visualization(image, elements, path: Path) -> None:
    """Draw every bbox + element ID on an ORIGINAL-size copy."""
    from PIL import ImageDraw

    vis = image.copy()
    d = ImageDraw.Draw(vis)
    W, H = image.size
    for e in elements:
        x1, y1, x2, y2 = e["bbox"]
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        label = f"[{e['id']}]"
        tx = min(x2 + 3, W - 1)
        ty = max(0, y1 - 12)
        d.rectangle([tx, ty, min(W, tx + 8 + 7 * len(label)), ty + 13],
                    fill=(255, 0, 0))
        d.text((tx + 4, ty + 1), label, fill=(255, 255, 255))
    vis.save(path)
    log.info("Saved visualization: %s", path)


def crop_all_elements(image, elements, crops_dir: Path,
                      padding_ratio: float):
    """Crop EVERY valid bbox from the ORIGINAL image with padding.

    padding_x = int(width * PADDING_RATIO) (same for y), clipped to the
    image. Saved crops are original-resolution and never modified.
    """
    W, H = image.size
    crops_dir.mkdir(parents=True, exist_ok=True)
    for e in elements:
        x1, y1, x2, y2 = e["bbox"]
        w, h = x2 - x1, y2 - y1
        px, py = int(w * padding_ratio), int(h * padding_ratio)
        padded = [max(0, x1 - px), max(0, y1 - py),
                  min(W, x2 + px), min(H, y2 + py)]
        e["padded_bbox"] = padded
        crop_path = crops_dir / f"element_{e['id']:04d}.png"
        image.crop(tuple(padded)).save(crop_path)
        e["crop_path"] = str(crop_path)
    log.info("Creating crops... %d crops in %s", len(elements), crops_dir)
    return elements


# --------------------------------------------------------------------------
# 3. OCR (one crop at a time -- never the full screenshot, never a VLM)
# --------------------------------------------------------------------------

def initialize_ocr(engine: str, cpu: bool = False):
    """Load PaddleOCR (preferred) or EasyOCR.

    Raises an actionable install error when unavailable -- never silently
    substitutes a VLM.
    """
    engine = (engine or "").lower()
    if engine == "paddleocr":
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise ImportError(
                "OCR engine 'paddleocr' requires `paddleocr` + "
                "`paddlepaddle`. Install with: "
                "pip install paddleocr paddlepaddle") from e
        if cpu:
            try:
                log.info("PaddleOCR loaded (lang=en, device=cpu)")
                return ("paddleocr", PaddleOCR(lang="en", device="cpu"))
            except TypeError:
                log.info("PaddleOCR ignores device kwarg; using defaults.")
        log.info("PaddleOCR loaded (lang=en)")
        return ("paddleocr", PaddleOCR(lang="en"))
    if engine == "easyocr":
        try:
            import easyocr
        except ImportError as e:
            raise ImportError(
                "OCR engine 'easyocr' requires `easyocr`. Install with: "
                "pip install easyocr") from e
        import torch
        use_gpu = torch.cuda.is_available() and not cpu
        log.info("Loading EasyOCR (gpu=%s)...", use_gpu)
        reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        try:
            import cv2  # noqa: F401
            has_cv2 = True
        except ImportError:
            has_cv2 = False
        return ("easyocr", {"reader": reader, "has_cv2": has_cv2})
    raise ValueError(f"Unknown OCR engine {engine!r}. Use 'paddleocr' or "
                     f"'easyocr'. (A VLM is never a valid OCR engine.)")


def _upscale_copy(crop, factor: int = OCR_UPSCALE):
    """Upscaled COPY for OCR; the saved crop is never modified."""
    from PIL import Image

    if factor and factor != 1:
        w, h = crop.size
        return crop.resize((w * int(factor), h * int(factor)),
                           resample=Image.BICUBIC)
    return crop.copy()


def normalize_paddle_regions(raw) -> list:
    """PaddleOCR 3.x predict() output -> [{text, bbox xyxy, score}] with
    bboxes in the input-image pixels. Shared by per-crop OCR and hybrid
    text proposals."""
    import numpy as np

    regions = []
    pages = raw if isinstance(raw, list) else [raw]
    for page in pages:
        if not isinstance(page, dict):
            continue
        texts = page.get("rec_texts") or []
        scores = page.get("rec_scores") or []
        polys = page.get("rec_polys")
        boxes = page.get("rec_boxes")
        geoms = polys if polys is not None else (
            boxes if boxes is not None else [])
        for i in range(min(len(texts), len(scores), len(geoms))):
            try:
                g = geoms[i]
                if hasattr(g, "tolist"):
                    g = g.tolist()
                if len(g) == 4 and all(
                        isinstance(v, (int, float)) for v in g):
                    bb = [float(v) for v in g]
                else:
                    pts = np.asarray(g, dtype=float).reshape(-1, 2)
                    bb = [float(pts[:, 0].min()),
                          float(pts[:, 1].min()),
                          float(pts[:, 0].max()),
                          float(pts[:, 1].max())]
            except Exception:
                continue
            if str(texts[i]).strip():
                regions.append({"text": str(texts[i]),
                                "score": float(scores[i]), "bbox": bb})
    return regions


def ocr_crop(ocr_handle, crop, crop_id: str):
    """OCR ONE crop. Returns (ocr_text, ocr_confidence, words).

    Word bboxes are in CROP pixel coordinates (the image OCR actually
    saw). No text -> ("", 0.0, []); the element is always kept.
    """
    import numpy as np

    kind, backend = ocr_handle
    work = _upscale_copy(crop)
    if kind == "paddleocr":
        arr = np.array(work.convert("RGB"))
        # Words are {text, confidence} exactly (no bboxes): PaddleOCR det
        # coordinates are unreliable in this environment, and the OCR copy
        # is upscaled anyway, so crop-space boxes would mislead.
        words = [{"text": r["text"], "confidence": r["score"]}
                 for r in normalize_paddle_regions(backend.predict(arr))]
        text = " ".join(w["text"] for w in words)
        conf = (sum(w["confidence"] for w in words) / len(words)
                if words else 0.0)
        return text, float(conf), words
    arr = np.array(work.convert("RGB"))
    if backend["has_cv2"]:
        import cv2
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    words = []
    for poly, text, score in backend["reader"].readtext(arr) or []:
        try:
            pts = np.asarray(poly, dtype=float).reshape(-1, 2)
            bb = [float(pts[:, 0].min()), float(pts[:, 1].min()),
                  float(pts[:, 0].max()), float(pts[:, 1].max())]
        except Exception:
            continue
        if str(text).strip():
            # Spec-exact words {text, confidence}: per-crop boxes would be
            # in upscaled-crop coordinates and mislead, so they are omitted.
            words.append({"text": str(text), "confidence": float(score)})
    text = " ".join(w["text"] for w in words)
    conf = (sum(w["confidence"] for w in words) / len(words)
            if words else 0.0)
    return text, float(conf), words


_EASY_READER = None


def ensure_easy_reader(cpu: bool = False):
    """Shared EasyOCR reader (lazy). EasyOCR box coordinates are verified
    correct in this environment; PaddleOCR det coordinates are not (see
    get_text_proposals), so proposals always use EasyOCR."""
    global _EASY_READER
    if _EASY_READER is None:
        try:
            import easyocr
        except ImportError as e:
            raise ImportError(
                "Hybrid proposals need `easyocr` for correct text-box "
                "coordinates. Install with: pip install easyocr "
                "opencv-python-headless") from e
        import torch
        use_gpu = torch.cuda.is_available() and not cpu
        log.info("Loading EasyOCR for proposals (gpu=%s)...", use_gpu)
        _EASY_READER = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
    return _EASY_READER


def get_text_proposals(image, cpu: bool = False):
    """Full-image OCR pass FOR PROPOSALS ONLY: tight text-region boxes.

    ALWAYS uses EasyOCR: PaddleOCR 3.7 det coordinates are systematically
    wrong in this environment (verified: a box drawn at [500,300,700,340]
    is reported at [737,410,782,429], at every image size), while EasyOCR
    boxes are verified correct (e.g. Home [65,127,115,145] inside its
    button [58,119,122,153]). Using PaddleOCR boxes here would silently
    corrupt bboxes.json. Recognition still happens later per crop
    (ocr_crop) with the configured OCR engine; these boxes only fix
    localization so labels/placeholders/inputs are covered.
    Returns [{bbox xyxy floats, confidence}].
    """
    import numpy as np

    reader = ensure_easy_reader(cpu)
    try:
        import cv2
        arr = cv2.cvtColor(np.array(image.convert("RGB")),
                           cv2.COLOR_RGB2BGR)
    except ImportError:
        arr = np.array(image.convert("RGB"))
    props = []
    for poly, text, score in reader.readtext(arr) or []:
        if not str(text).strip():
            continue
        try:
            pts = np.asarray(poly, dtype=float).reshape(-1, 2)
            bb = [float(pts[:, 0].min()), float(pts[:, 1].min()),
                  float(pts[:, 0].max()), float(pts[:, 1].max())]
        except Exception:
            continue
        props.append({"bbox": bb, "confidence": float(score)})
    return props


def tighten_box_to_content(image, bbox_xyxy):
    """Shrink a widget box to its non-background content (pixel/edge
    projection only -- no OCR, no learned model). Falls back to the
    original box when degenerate."""
    import numpy as np

    x1, y1, x2, y2 = (int(v) for v in bbox_xyxy)
    w, h = x2 - x1, y2 - y1
    if w < 8 or h < 8:
        return list(bbox_xyxy)
    g = np.array(image.crop((x1, y1, x2, y2)).convert("L"))
    border = np.concatenate([g[:2, :].ravel(), g[-2:, :].ravel(),
                             g[:, :2].ravel(), g[:, -2:].ravel()])
    bg = float(np.median(border))
    m = np.abs(g.astype(float) - bg) > 12
    col_need = max(2, 0.005 * h)
    row_need = max(2, 0.005 * w)
    cols, rows = m.sum(axis=0), m.sum(axis=1)
    kx = [x for x in range(w) if cols[x] >= col_need]
    ky = [y for y in range(h) if rows[y] >= row_need]
    if not kx or not ky:
        return list(bbox_xyxy)
    nx1, nx2, ny1, ny2 = kx[0], kx[-1] + 1, ky[0], ky[-1] + 1
    if nx2 - nx1 < 8 or ny2 - ny1 < 8:
        return list(bbox_xyxy)
    if (nx2 - nx1) * (ny2 - ny1) < 0.10 * w * h:
        return list(bbox_xyxy)
    return [x1 + nx1, y1 + ny1, x1 + nx2, y1 + ny2]


def _inter(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def merge_hybrid_proposals(widgets, texts, image, nms_thr,
                           nms_contain):
    """Union widget + text boxes into a precise set (order matters):

    1. Tighten widget boxes to content (pixel projection only).
    2. Drop widget boxes containing >= HYBRID_CONTAINER_MIN_CHILDREN
       text boxes (giant containers add no information).
    3. Widget-widget dedup with tighter-wins (kills imprecise duplicates
       such as wide price boxes). Runs BEFORE text filtering so a doomed
       duplicate can never eat a text it merely overlaps.
    4. Drop text boxes mostly inside a surviving widget
       (intersection/text-area > HYBRID_TEXT_IN_WIDGET_IOS): the widget
       crop contains that text, so per-crop OCR still recovers it.
    5. Light IoU-only NMS over surviving texts (near-duplicate words).
    Returns (kept_widgets, kept_texts) with original dicts preserved.
    """
    W, H = image.size
    tight = []
    for el in widgets:
        t = tighten_box_to_content(image, el["bbox"])
        ok, clean, _ = validate_bbox(t, (W, H))
        el["bbox"] = clean if ok else el["bbox"]
        tight.append(el)
    kept_w = []
    for el in tight:
        kids = sum(1 for t in texts
                   if _inter(t["bbox"], el["bbox"]) / max(1, _area(t["bbox"]))
                   > 0.9)
        if kids >= HYBRID_CONTAINER_MIN_CHILDREN:
            log.info("Hybrid dropping container bbox %r (%d text children).",
                     el["bbox"], kids)
            continue
        kept_w.append(el)
    if nms_thr is not None or nms_contain is not None:
        kept_w, n = apply_nms(kept_w, nms_thr, nms_contain,
                              prefer="small")
        log.info("Hybrid widget dedup kept %d, suppressed %d.",
                 len(kept_w), n)
    kept_t = []
    for t in texts:
        if any(_inter(t["bbox"], el["bbox"]) / max(1, _area(t["bbox"]))
               > HYBRID_TEXT_IN_WIDGET_IOS for el in kept_w):
            continue
        kept_t.append(t)
    if nms_thr is not None:
        kept_t, n = apply_nms(kept_t, nms_thr, None)
        log.info("Hybrid text dedup kept %d, suppressed %d.",
                 len(kept_t), n)
    log.info("Hybrid: %d widgets + %d texts -> %d + %d.",
             len(widgets), len(texts), len(kept_w), len(kept_t))
    return kept_w, kept_t


def save_ocr_results(path: Path, elements, out_dir: Path) -> None:
    """Write ocr_results.json. crop_path is relative to the output dir
    (crops/element_0001.png) so detection -> crop -> OCR stays traceable."""
    payload = []
    for e in elements:
        try:
            rel_crop = str(Path(e.get("crop_path", "")).relative_to(
                out_dir))
        except ValueError:
            rel_crop = Path(e.get("crop_path", "")).name
        payload.append({
            "id": e["id"],
            "original_bbox": e["bbox"],
            "padded_bbox": e.get("padded_bbox"),
            "confidence": e.get("confidence"),
            "label": e.get("label"),
            "ocr_text": e.get("ocr_text", ""),
            "ocr_confidence": e.get("ocr_confidence", 0.0),
            "crop_path": rel_crop,
            "words": e.get("words", []),
        })
    path.write_text(json.dumps(payload, indent=4))
    log.info("Saved OCR results: %s", path)


def print_summary(elements, raw_count: int, out_dir: Path,
                  detector_name: str = "OmniParser") -> None:
    """Print the final ID / BBOX / OCR TEXT table and output paths."""
    print("=" * 60)
    print(f"{detector_name} + OCR Results")
    print("=" * 60)
    print()
    print(f"{'ID':<6}{'BBOX':<30}OCR TEXT")
    print("-" * 60)
    for e in elements:
        x1, y1, x2, y2 = e["bbox"]
        print(f"{e['id']:<6}[{x1},{y1},{x2},{y2}]"
              f"{'':<{30 - len(f'[{x1},{y1},{x2},{y2}]')}}"
              f"{e.get('ocr_text', '')}")
    print()
    print("=" * 60)
    print(f"Total {detector_name} detections: {raw_count}")
    print(f"Valid bboxes: {len(elements)}")
    print(f"Crops saved: {len(elements)}")
    print(f"OCR results: {len(elements)}")
    print("=" * 60)
    print()
    print("Output:")
    print(f"{out_dir}/bbox_visualization.png")
    print(f"{out_dir}/bboxes.json")
    print(f"{out_dir}/ocr_results.json")
    print(f"{out_dir}/crops/")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="OmniParser UI-element detection + per-crop OCR "
                    "for Ocr/input.png (no VLM, crops only).")
    ap.add_argument("--input", default=None,
                    help="input screenshot (default: Ocr/input.png)")
    ap.add_argument("--output", default=None,
                    help="output directory (default: Ocr/test_2)")
    ap.add_argument("--padding", type=float, default=None,
                    help="bbox padding ratio (default: 0.05)")
    ap.add_argument("--ocr-upscale", type=int, default=None,
                    help="upscale factor for the OCR copy (default: 4)")
    ap.add_argument("--ocr-engine", default=None,
                    choices=["paddleocr", "easyocr"],
                    help="OCR backend (default: paddleocr)")
    ap.add_argument("--proposals", default=None,
                    choices=["omni", "omni+ocr"],
                    help="proposal sources: OmniParser boxes only, or plus "
                         "tight OCR text boxes (default: omni)")
    ap.add_argument("--detector", default=None,
                    choices=["omniparser", "gpa"],
                    help="detection backend: OmniParser (default) or GPA "
                         "Detector with gpa_postprocess.py stages")
    ap.add_argument("--box-threshold", type=float, default=None,
                    help="OmniParser confidence threshold (default: 0.05)")
    ap.add_argument("--nms-iou", type=float, default=None,
                    help="NMS IoU threshold (default: 0.5)")
    ap.add_argument("--nms-contain", type=float, default=None,
                    help="NMS containment threshold on "
                         "intersection/smaller-area (default: 0.5)")
    ap.add_argument("--no-nms", action="store_true",
                    help="disable NMS deduplication")
    ap.add_argument("--model-dir", default=None,
                    help="OmniParser model dir (default: repo pretrained/)")
    ap.add_argument("--cpu", action="store_true",
                    help="force CPU for detection and OCR")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    global OCR_UPSCALE
    _setup_logging()
    args = parse_args(argv)

    input_arg = args.input or INPUT_IMAGE
    padding = PADDING_RATIO if args.padding is None else args.padding
    upscale = OCR_UPSCALE if args.ocr_upscale is None else args.ocr_upscale
    engine = (args.ocr_engine or OCR_ENGINE).lower()
    detector_kind = (args.detector or DETECTOR).lower()
    if detector_kind not in ("omniparser", "gpa"):
        log.error("Unknown detector %r (use omniparser/gpa).",
                  detector_kind)
        return 2
    proposals = args.proposals or PROPOSALS
    box_thr = BOX_THRESHOLD if args.box_threshold is None \
        else args.box_threshold
    if args.no_nms:
        nms_thr, nms_contain = None, None
    else:
        nms_thr = NMS_IOU_THRESHOLD if args.nms_iou is None \
            else args.nms_iou
        nms_contain = NMS_CONTAIN_THRESHOLD if args.nms_contain is None \
            else args.nms_contain
    device = "cpu" if args.cpu else "auto"

    OCR_UPSCALE = upscale

    try:
        image, input_path = load_image(input_arg)
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 2
    W, H = image.size

    out_dir = Path(args.output or OUTPUT_DIR)
    if not out_dir.is_absolute():
        # Resolve a relative output dir against the cwd, unless it is
        # already anchored at the script dir layout (Ocr/...).
        cand = Path.cwd() / out_dir
        alt = SCRIPT_DIR / out_dir
        out_dir = cand if cand == alt else (
            alt if not cand.exists() and alt.exists() else cand)
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "crops"
    log.info("Output directory: %s", out_dir)

    # Self-contained run dir: copy (never move) input.png -> output/.
    try:
        dst_input = out_dir / input_path.name
        if dst_input.resolve() != input_path.resolve():
            shutil.copy2(input_path, dst_input)
            log.info("Copied input -> %s", dst_input)
    except Exception as e:
        log.error("Could not copy input image: %s", e)
        return 2

    if detector_kind == "gpa":
        log.info("Running GPA Detector + post-processing "
                 "(ALL elements, no filter)...")
        try:
            elements, n_raw_dets = run_gpa_detection(image, (W, H),
                                                     device)
        except (ImportError, RuntimeError) as e:
            log.error("%s", e)
            return 2
        log.info("Detected %d UI elements.", len(elements))
        nms_suppressed = 0
    else:
        try:
            detector = load_omniparser(model_dir=args.model_dir or
                                       OMNIPARSER_MODEL_DIR,
                                       device=device, box_threshold=box_thr)
        except (ImportError, RuntimeError) as e:
            log.error("%s", e)
            return 2

        log.info("Running OmniParser on input.png (ALL elements, no filter)...")
        try:
            raw = run_omniparser(detector, image)
        except Exception as e:
            log.error("OmniParser inference failed: %s", e)
            return 2
        log.info("Detected %d UI elements.", len(raw or []))
        n_raw_dets = len(raw or [])

    if detector_kind == "omniparser":
        elements = []
        for cand in normalize_detections(raw):
            native = convert_bbox_to_original(cand["bbox"], (W, H))
            ok, clean, reason = validate_bbox(native, (W, H))
            if not ok:
                log.warning("Rejecting element %d bbox %r: %s",
                            cand["id"], cand["bbox"], reason)
                continue
            cand["bbox"] = clean
            elements.append(cand)
        ocr = None
        if proposals == "omni+ocr":
            # Hybrid: tight OCR text boxes join widget boxes (reference look).
            # Proposals use EasyOCR (verified coordinates); the configured
            # engine is still used later for per-crop recognition.
            try:
                ensure_easy_reader(cpu=args.cpu)
            except ImportError as e:
                log.error("%s", e)
                return 2
            log.info("Running full-image OCR for text proposals...")
            texts = []
            for t in get_text_proposals(image, cpu=args.cpu):
                ok, clean, _ = validate_bbox(t["bbox"], (W, H))
                if not ok:
                    continue
                if (clean[2] - clean[0] < HYBRID_MIN_TEXT_SIZE
                        or clean[3] - clean[1] < HYBRID_MIN_TEXT_SIZE):
                    continue
                texts.append({"id": 0, "bbox": clean,
                              "confidence": t["confidence"], "label": "text"})
            log.info("Got %d text proposals.", len(texts))
            kept_w, kept_t = merge_hybrid_proposals(elements, texts, image,
                                                      nms_thr, nms_contain)
            elements = kept_w + kept_t
        nms_suppressed = 0
        if proposals == "omni" and (
                nms_thr is not None or nms_contain is not None):
            log.info("Running NMS (iou>%s, contain>%s) on %d valid boxes...",
                     nms_thr, nms_contain, len(elements))
            elements, nms_suppressed = apply_nms(elements, nms_thr,
                                                 nms_contain)
            log.info("NMS kept %d boxes, suppressed %d.",
                     len(elements), nms_suppressed)
    else:
        ocr = None
    for j, e in enumerate(elements, start=1):
        e["id"] = j  # IDs match crops/table exactly

    save_bboxes(out_dir / "bboxes.json", elements)
    create_bbox_visualization(image, elements,
                              out_dir / "bbox_visualization.png")
    crop_all_elements(image, elements, crops_dir, padding)

    if ocr is None:
        try:
            ocr = initialize_ocr(engine, cpu=args.cpu)
        except (ImportError, ValueError) as e:
            log.error("%s", e)
            return 2

    log.info("Running OCR (%s) on %d crops...", engine, len(elements))
    for k, e in enumerate(elements, start=1):
        try:
            from PIL import Image as _Image
            crop = _Image.open(e["crop_path"]).convert("RGB")
            text, conf, words = ocr_crop(ocr, crop,
                                         f"element_{e['id']:04d}")
        except Exception as ex:
            log.error("Element %04d OCR failed: %s", e["id"], ex)
            print(f"Element {e['id']:04d} OCR failed: {ex}\nContinuing...")
            text, conf, words = "", 0.0, []
        e["ocr_text"], e["ocr_confidence"], e["words"] = text, conf, words
        log.info("[%d/%d] element_%04d.png -> %r (conf=%.2f)",
                 k, len(elements), e["id"], text, conf)

    save_ocr_results(out_dir / "ocr_results.json", elements, out_dir)
    print_summary(elements, n_raw_dets, out_dir,
                  "GPA" if detector_kind == "gpa" else "OmniParser")
    return 0


if __name__ == "__main__":
    sys.exit(main())
