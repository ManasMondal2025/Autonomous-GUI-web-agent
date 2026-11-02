#!/usr/bin/env python3
"""OCR follow-up step: crop every final bbox from the ORIGINAL image and
OCR each crop independently (PaddleOCR, 4x-upscaled copy).

Input:  screenshot + final_detections.json (e.g. from gpa_postprocess.py)
Output: crops/element_XXXX.png + ocr_results.json
        [{id, original_bbox, padded_bbox, ocr_text, ocr_confidence,
          crop_path, words:[{text, confidence}]}]

Same PaddleOCR call pattern as the earlier ocr_pipeline.py work
(PaddleOCR(lang='en').predict(rgb_array), 4x BICUBIC OCR copy).
No VLM. Never OCRs the full screenshot.

Run:
    python ocr_crops.py --input IMG --detections DET.json --output DIR
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

INPUT_IMAGE = ""
DETECTIONS_JSON = ""
OUTPUT_DIR = ""
PADDING_RATIO = 0.05
OCR_UPSCALE = 4
OCR_ENGINE = "paddleocr"

LOG_LEVEL = "INFO"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SCRIPT_DIR = Path(__file__).resolve().parent

log = logging.getLogger("ocr_crops")


def setup_logging(level: str = LOG_LEVEL) -> None:
    logging.basicConfig(level=getattr(logging, str(level).upper(),
                                      logging.INFO),
                        format="%(levelname)s: %(message)s")


def resolve(p: str) -> Path:
    q = Path(p)
    if q.exists() or q.is_absolute():
        return q
    return SCRIPT_DIR / q


def load_image(path: str):
    from PIL import Image

    p = resolve(path)
    if not p.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format {p.suffix!r}.")
    return Image.open(p).convert("RGB"), p


def load_detections(path: str):
    raw = json.loads(resolve(path).read_text())
    dets = raw.get("detections", raw) if isinstance(raw, dict) else raw
    out = []
    for j, d in enumerate(dets, start=1):
        out.append({"id": d.get("id", j),
                    "bbox": [int(v) for v in d["bbox"]]})
    return out


def load_ocr(engine: str = OCR_ENGINE):
    engine = (engine or "").lower()
    if engine != "paddleocr":
        raise ValueError(f"Only 'paddleocr' is supported, got {engine!r}.")
    try:
        from paddleocr import PaddleOCR
    except ImportError as e:
        raise ImportError(
            "OCR requires `paddleocr` + `paddlepaddle`. Install with: "
            "pip install paddleocr paddlepaddle") from e
    log.info("PaddleOCR loaded (lang=en)")
    return PaddleOCR(lang="en")


def normalize_paddle_output(raw) -> list:
    """PaddleOCR 3.x predict() -> [{text, score}]. Word boxes are omitted
    (PaddleOCR det coordinates are unreliable in this env; OCR text is
    unaffected)."""
    regions = []
    pages = raw if isinstance(raw, list) else [raw]
    for page in pages:
        if not isinstance(page, dict):
            continue
        texts = page.get("rec_texts") or []
        scores = page.get("rec_scores") or []
        for t, s in zip(texts, scores):
            if str(t).strip():
                regions.append({"text": str(t), "score": float(s)})
    return regions


def ocr_crop(ocr, crop):
    """OCR ONE crop (4x-upscaled copy; saved crop untouched)."""
    import numpy as np

    w, h = crop.size
    work = crop.resize((w * OCR_UPSCALE, h * OCR_UPSCALE),
                       __import__("PIL.Image").Image.BICUBIC)
    words = [{"text": r["text"], "confidence": r["score"]}
             for r in normalize_paddle_output(
                 ocr.predict(np.array(work.convert("RGB"))))]
    text = " ".join(w["text"] for w in words)
    conf = (sum(w["confidence"] for w in words) / len(words)
            if words else 0.0)
    return text, float(conf), words


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Per-crop OCR for final detection boxes.")
    ap.add_argument("--input", default=None)
    ap.add_argument("--detections", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--padding", type=float, default=None)
    ap.add_argument("--upscale", type=int, default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    setup_logging()
    args = parse_args(argv)

    input_arg = args.input or INPUT_IMAGE
    det_arg = args.detections or DETECTIONS_JSON
    out_dir = Path(args.output or OUTPUT_DIR)
    if not input_arg or not det_arg or not str(out_dir):
        log.error("Need --input, --detections and --output.")
        return 2
    if not out_dir.is_absolute():
        cand, alt = Path.cwd() / out_dir, SCRIPT_DIR / out_dir
        out_dir = cand if cand == alt else (
            alt if not cand.exists() and alt.exists() else cand)
    out_dir.mkdir(parents=True, exist_ok=True)
    padding = PADDING_RATIO if args.padding is None else args.padding
    global OCR_UPSCALE
    if args.upscale is not None:
        OCR_UPSCALE = args.upscale

    try:
        image, input_path = load_image(input_arg)
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 2
    W, H = image.size
    try:
        dets = load_detections(det_arg)
    except Exception as e:
        log.error("Cannot read detections %s: %s", det_arg, e)
        return 2
    log.info("Loaded %d boxes from %s.", len(dets), det_arg)

    try:
        ocr = load_ocr()
    except (ImportError, ValueError) as e:
        log.error("%s", e)
        return 2

    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(det_arg if Path(det_arg).exists()
                     else resolve(det_arg),
                     out_dir / "final_detections.json")
    except Exception as e:
        log.warning("Could not copy detections json: %s", e)

    results = []
    for k, d in enumerate(dets, start=1):
        x1, y1, x2, y2 = d["bbox"]
        w, h = x2 - x1, y2 - y1
        px, py = int(w * padding), int(h * padding)
        padded = [max(0, x1 - px), max(0, y1 - py),
                  min(W, x2 + px), min(H, y2 + py)]
        cp = crops_dir / f"element_{d['id']:04d}.png"
        if cp.resolve() == input_path.resolve():
            log.error("Refusing to overwrite input: %s", cp)
            return 2
        crop = image.crop(tuple(padded))
        crop.save(cp)
        try:
            text, conf, words = ocr_crop(ocr, crop)
        except Exception as ex:
            log.error("Element %04d OCR failed: %s", d["id"], ex)
            print(f"Element {d['id']:04d} OCR failed: {ex}\nContinuing...")
            text, conf, words = "", 0.0, []
        results.append({"id": d["id"], "original_bbox": d["bbox"],
                        "padded_bbox": padded, "ocr_text": text,
                        "ocr_confidence": round(conf, 4),
                        "crop_path": f"crops/{cp.name}",
                        "words": words})
        log.info("[%d/%d] element_%04d.png -> %r (conf=%.2f)",
                 k, len(dets), d["id"], text[:60], conf)
    (out_dir / "ocr_results.json").write_text(json.dumps(results,
                                                         indent=4))
    print("=" * 60)
    print(f"OCR done: {len(results)} crops, "
          f"{sum(1 for r in results if r['ocr_text'])} with text.")
    print(f"Output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
