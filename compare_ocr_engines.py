#!/usr/bin/env python3
"""PaddleOCR vs EasyOCR inside the hybrid pipeline (comparison only).

For each input image: GPA detector + gpa_postprocess.py stages (the exact
hybrid visual path) -> final boxes -> crop ONCE -> OCR the SAME crop with
BOTH engines (ocr_pipeline.ocr_crop, identical 4x upscale) -> side-by-side
comparison. No GT transcripts exist, so this measures agreement + yield,
not accuracy. No existing code is modified.

Run:
    python compare_ocr_engines.py --input A.png --input B.png --output Ocr/com_test
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PADDING_RATIO = 0.05
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

log = logging.getLogger("compare_ocr")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")


def resolve(p: str) -> Path:
    q = Path(p)
    if q.exists() or q.is_absolute():
        return q
    return SCRIPT_DIR / q


def norm_text(s: str) -> str:
    return " ".join(str(s or "").lower().split())


def token_sim(a: str, b: str) -> float:
    ta, tb = set(norm_text(a).split()), set(norm_text(b).split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, max(len(ta), len(tb)))


def process_image(image_path: Path, out_dir: Path, paddle, easy):
    import sys as _sys
    if str(SCRIPT_DIR) not in _sys.path:
        _sys.path.insert(0, str(SCRIPT_DIR))
    import gpa_postprocess as gp
    import ocr_pipeline as oc
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    raw = gp.run_gpa_detector(img, gp.GPA_MODEL,
                              min(gp.DETECTOR_CONF, gp.CONF_THRESHOLD),
                              gp.DETECTOR_INTERNAL_NMS_IOU, "cpu")
    _, s1 = gp.stage_filter(raw, (W, H), gp.CONF_THRESHOLD)
    s2 = gp.stage_exact_dup(s1)
    s3 = gp.stage_remove_parents(s2)
    s4 = gp.stage_merge(s3)
    s5 = gp.stage_final_cleanup(s4)
    final = sorted(s5, key=lambda r: (-r["confidence"], r["raw_id"]))
    log.info("%s: %d raw -> %d final boxes.",
             image_path.name, len(raw), len(final))

    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final_detections.json").write_text(json.dumps(
        [{"id": j + 1, "bbox": [int(v) for v in r["bbox"]],
          "confidence": r["confidence"], "label": r.get("label") or ""}
         for j, r in enumerate(final)], indent=4))
    pad, pres, eres = [], [], []
    for j, r in enumerate(final, start=1):
        x1, y1, x2, y2 = (int(v) for v in r["bbox"])
        w, h = x2 - x1, y2 - y1
        px, py = int(w * PADDING_RATIO), int(h * PADDING_RATIO)
        pb = [max(0, x1 - px), max(0, y1 - py),
              min(W, x2 + px), min(H, y2 + py)]
        cp = crops_dir / f"element_{j:04d}.png"
        crop = img.crop(tuple(pb))
        crop.save(cp)
        try:
            pt, pc, pw = oc.ocr_crop(paddle, crop, f"element_{j:04d}")
        except Exception as e:
            log.error("paddle failed on %s: %s", cp.name, e)
            pt, pc, pw = "", 0.0, []
        try:
            et, ec, ew = oc.ocr_crop(easy, crop, f"element_{j:04d}")
        except Exception as e:
            log.error("easy failed on %s: %s", cp.name, e)
            et, ec, ew = "", 0.0, []
        pad.append({"id": j, "original_bbox": [x1, y1, x2, y2],
                    "padded_bbox": pb, "ocr_text": pt,
                    "ocr_confidence": round(pc, 4),
                    "crop_path": f"crops/{cp.name}",
                    "words": pw})
        eres.append({"id": j, "original_bbox": [x1, y1, x2, y2],
                     "padded_bbox": pb, "ocr_text": et,
                     "ocr_confidence": round(ec, 4),
                     "crop_path": f"crops/{cp.name}",
                     "words": ew})
        pres.append({"id": j, "bbox": [x1, y1, x2, y2],
                     "paddle_text": pt, "paddle_conf": round(pc, 4),
                     "easy_text": et, "easy_conf": round(ec, 4),
                     "agree_exact": norm_text(pt) == norm_text(et),
                     "token_similarity": round(token_sim(pt, et), 3)})
        log.info("[%d/%d] P=%r (%.2f) | E=%r (%.2f)", j, len(final),
                 pt[:40], pc, et[:40], ec)
    (out_dir / "paddle_ocr.json").write_text(json.dumps(pad, indent=4))
    (out_dir / "easyocr_ocr.json").write_text(json.dumps(eres, indent=4))
    (out_dir / "comparison.json").write_text(json.dumps(pres, indent=4))
    return pres


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="PaddleOCR vs EasyOCR on identical hybrid crops.")
    ap.add_argument("--input", action="append", default=[],
                    help="input image (repeatable)")
    ap.add_argument("--output", default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    setup_logging()
    args = parse_args(argv)
    if not args.input or not args.output:
        log.error("Need --input (repeatable) and --output.")
        return 2
    out_root = Path(args.output)
    if not out_root.is_absolute():
        cand = Path.cwd() / out_root
        out_root = cand if cand == out_root else out_root
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import ocr_pipeline as oc
    log.info("Loading PaddleOCR...")
    paddle = oc.initialize_ocr("paddleocr", cpu=True)
    log.info("Loading EasyOCR (cpu)...")
    easy = oc.initialize_ocr("easyocr", cpu=True)
    summary = {}
    for src in args.input:
        p = resolve(src)
        if not p.is_file():
            log.error("Missing image: %s (skipped)", src)
            continue
        d = out_root / p.stem
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, d / p.name)
        rows = process_image(p, d, paddle, easy)
        n = len(rows)
        agree = sum(1 for r in rows if r["agree_exact"])
        pn = sum(1 for r in rows if r["paddle_text"].strip())
        en = sum(1 for r in rows if r["easy_text"].strip())
        summary[d.name] = {"crops": n, "agree_exact": agree,
                           "paddle_nonempty": pn, "easy_nonempty": en}
        print(f"{d.name}: crops={n} agree={agree} "
              f"paddle_nonempty={pn} easy_nonempty={en}")
    (out_root / "summary.json").write_text(json.dumps(summary, indent=4))
    print(f"Output: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
