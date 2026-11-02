#!/usr/bin/env python3
"""Experimental hybrid SoM mode: live DOM candidates + GPA visual candidates.

Pipeline:
    test_1.html -> browser page -> LIVE DOM + screenshot (same state)
      -> DOM candidates (new, thin wrapper over the repo DOMExtractor)
      -> GPA candidates (EXISTING ocr_pipeline.run_gpa_detection, i.e. the
         test_19 path: GPA detector + gpa_postprocess.py stages, untouched)
      -> cross-source matching (NEW: deterministic geometry + text match)
         matched -> MERGE (union) | DOM-only -> KEEP | GPA-only -> KEEP
      -> hybrid candidates -> EXISTING ocr_pipeline OCR per bbox
         (initialize_ocr + crop_all_elements + ocr_crop, untouched)

The ONLY new logic: DOM extraction wrapper + DOM<->GPA association/union.
No new detector, NMS, containment, merging-for-visuals, OCR, or VLM.
Existing SoM modes, GPA, OmniParser, OCR and grounding are NOT modified
(this file only ADDS code; it edits nothing).

Run:
    python hybrid_som.py
    python hybrid_som.py --html test_1.html --output Ocr/test_20
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

INPUT_HTML = "test_1.html"
OUTPUT_DIR = "Ocr/test_20"

VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 800
WAIT_AFTER_LOAD = 2.0          # dynamic element appears after ~1s; no scroll

# NEW hybrid matching thresholds (existing GPA thresholds untouched).
MATCH_IOU_THRESHOLD = 0.50
MATCH_CONTAINMENT_THRESHOLD = 0.80
MATCH_MAX_CENTER_DIST = 0.50   # normalized by mean box diagonal
MATCH_TEXT_SIM_THRESHOLD = 0.50  # relaxes geometry when texts agree
MATCH_IOU_RELAXED = 0.30
MATCH_CONTAINMENT_RELAXED = 0.60
# Near-identical geometry merges regardless of text (OCR/DOM text is
# noisy; disagreement cannot veto near-coincident boxes).
STRONG_IOU_THRESHOLD = 0.85

OCR_ENGINE = "paddleocr"       # same engine ocr_pipeline.py uses
OCR_UPSCALE = 4
PADDING_RATIO = 0.05

LOG_LEVEL = "INFO"
SCRIPT_DIR = Path(__file__).resolve().parent

# stdlib `logging` FIRST (repo ships src/logging/ which must never shadow it;
# DOMExtractor import adds src/ to sys.path below).
log = logging.getLogger("hybrid_som")


def setup_logging(level: str = LOG_LEVEL) -> None:
    logging.basicConfig(level=getattr(logging, str(level).upper(),
                                      logging.INFO),
                        format="%(levelname)s: %(message)s")


def _ensure_repo_imports():
    src = SCRIPT_DIR / "src"
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


# --------------------------------------------------------------------------
# Geometry + text helpers (pure, deterministic; matching use only)
# --------------------------------------------------------------------------

def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _iou(a, b) -> float:
    inter = _inter(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _containment(a, b) -> float:
    return _inter(a, b) / max(1e-9, min(_area(a), _area(b)))


def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _diag(b) -> float:
    import math
    return math.hypot(b[2] - b[0], b[3] - b[1])


def _tokens(s: str):
    import re
    return [t for t in re.findall(r"[a-z0-9]+", str(s or "").lower())]


def _text_sim(a: str, b: str) -> float:
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


# --------------------------------------------------------------------------
# Browser: same-state screenshot + DOM (NEW wrapper; extractor is reused)
# --------------------------------------------------------------------------

def load_browser_page(html_arg: str, viewport_w: int = VIEWPORT_WIDTH,
                      viewport_h: int = VIEWPORT_HEIGHT,
                      wait_s: float = WAIT_AFTER_LOAD):
    """Open test_1.html headless (system Chrome), wait, screenshot.

    Returns (page, screenshot PIL, metrics, stop_fn). No scrolling, the
    page fits one viewport by construction.
    """
    from io import BytesIO
    from PIL import Image
    from playwright.sync_api import sync_playwright

    html_path = Path(html_arg)
    if not html_path.is_absolute():
        alt = SCRIPT_DIR / html_arg
        html_path = html_path if html_path.is_file() else alt
    if not html_path.is_file():
        raise FileNotFoundError(f"HTML file not found: {html_arg}")

    pw = sync_playwright().start()
    browser = pw.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": viewport_w,
                                      "height": viewport_h})
    page.goto(html_path.resolve().as_uri())
    page.wait_for_timeout(int(wait_s * 1000))
    shot = Image.open(BytesIO(page.screenshot())).convert("RGB")
    try:
        dpr = float(page.evaluate("window.devicePixelRatio || 1"))
    except Exception:
        dpr = 1.0
    metrics = {"screenshot_width": shot.size[0],
               "screenshot_height": shot.size[1],
               "viewport_width": viewport_w, "viewport_height": viewport_h,
               "device_pixel_ratio": dpr}
    print(f"screenshot_width: {metrics['screenshot_width']}")
    print(f"screenshot_height: {metrics['screenshot_height']}")
    print(f"viewport_width: {metrics['viewport_width']}")
    print(f"viewport_height: {metrics['viewport_height']}")
    print(f"device_pixel_ratio: {metrics['device_pixel_ratio']}")

    def stop():
        try:
            browser.close()
        except Exception:
            pass
        try:
            pw.stop()
        except Exception:
            pass

    return page, shot, metrics, stop


def extract_dom_candidates(page, metrics):
    """DOM candidates from the LIVE DOM (getBoundingClientRect, so this is
    the rendered layout -- never parsed HTML text).

    Thin NEW wrapper over the EXISTING repo DOMExtractor (same selector +
    visibility filtering as the agent's som/som_dom modes). Converts to
    screenshot pixels (CSS px * DPR scale, clipped) and the hybrid dict.
    """
    _ensure_repo_imports()
    from browser.dom_extractor import DOMExtractor

    W, H = metrics["screenshot_width"], metrics["screenshot_height"]
    sx = W / max(1, metrics["viewport_width"])
    sy = H / max(1, metrics["viewport_height"])
    out = []
    for el in DOMExtractor().extract(page) or []:
        bb = el.get("bbox") or {}
        try:
            x = float(bb.get("x", 0)) * sx
            y = float(bb.get("y", 0)) * sy
            w = float(bb.get("width", 0)) * sx
            h = float(bb.get("height", 0)) * sy
        except (TypeError, ValueError):
            continue
        box = [max(0, int(round(x))), max(0, int(round(y))),
               min(W, int(round(x + w))), min(H, int(round(y + h)))]
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            continue
        if not (0 <= box[0] < box[2] <= W and 0 <= box[1] < box[3] <= H):
            continue
        text = (el.get("text") or el.get("ariaLabel")
                or el.get("placeholder") or el.get("value")
                or el.get("title") or "")
        out.append({"source": "dom", "bbox": box,
                    "text": str(text)[:200],
                    "role": el.get("role") or "",
                    "tag": el.get("tag") or "",
                    "element_id": f"dom_{el.get('id')}"})
    log.info("DOM candidates: %d.", len(out))
    return out


# --------------------------------------------------------------------------
# GPA candidates (EXISTING test_19 path, untouched)
# --------------------------------------------------------------------------

def run_gpa_candidates(image):
    """Visual candidates via EXISTING ocr_pipeline.run_gpa_detection --
    the exact test_19 logic (GPA detector + gpa_postprocess.py stages,
    same config values, same order). Returns [{source:bbox..., label}]."""
    _ensure_repo_imports()
    import ocr_pipeline as oc

    elements, n_raw = oc.run_gpa_detection(image, image.size, "cpu")
    log.info("GPA candidates: %d (raw %d).", len(elements), n_raw)
    return [{"source": "gpa", "bbox": [int(v) for v in e["bbox"]],
             "confidence": e["confidence"], "label": e.get("label") or "",
             "gpa_id": e["id"]} for e in elements]


# --------------------------------------------------------------------------
# Cross-source matching (the ONLY new bbox operation)
# --------------------------------------------------------------------------

def pair_features(dom, gpa):
    """Deterministic geometry + text features for one DOM/GPA pair."""
    a, b = dom["bbox"], gpa["bbox"]
    iou = _iou(a, b)
    cont = _containment(a, b)
    ca, cb = _center(a), _center(b)
    import math
    dist = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
    norm_center = dist / max(1e-9, (_diag(a) + _diag(b)) / 2.0)
    tsim = _text_sim(dom.get("text", ""), gpa.get("ocr_hint", ""))
    return {"iou": iou, "containment": cont,
            "norm_center": norm_center, "text_sim": tsim,
            # Veto needs words on BOTH sides: production GPA emits no
            # text, so a missing side is "no signal", never disagreement.
            "both_text": bool(_tokens(dom.get("text", ""))
                              and _tokens(gpa.get("ocr_hint", "")))}


def match_reason(feats):
    """Same rule as is_match, but also names the firing path (for MATCH
    debug reporting). Returns (matched: bool, reason: str) with reason in
    {strong_geometry, agree_relax, base_geometry, vetoed_text_disagreement,
    no_match}."""
    if feats["iou"] >= STRONG_IOU_THRESHOLD:
        return True, "strong_geometry"
    relaxed = feats["text_sim"] >= MATCH_TEXT_SIM_THRESHOLD
    iou_thr = MATCH_IOU_RELAXED if relaxed else MATCH_IOU_THRESHOLD
    cont_thr = (MATCH_CONTAINMENT_RELAXED if relaxed
                else MATCH_CONTAINMENT_THRESHOLD)
    if not ((feats["iou"] >= iou_thr
             or feats["containment"] >= cont_thr)
            and feats["norm_center"] <= MATCH_MAX_CENTER_DIST):
        return False, "no_match"
    if feats.get("both_text") and feats["text_sim"] <= 0.0:
        return False, "vetoed_text_disagreement"
    return True, ("agree_relax" if relaxed else "base_geometry")


def is_match(feats):
    """Configurable rule with three text zones (deterministic, no VLM):

    * Near-identical geometry (IoU >= STRONG_IOU_THRESHOLD) merges no
      matter what the texts say -- OCR/DOM strings are noisy.
    * Agreeing text (sim >= .5) relaxes the geometry bars.
    * Otherwise base geometry must hold, AND strong disagreement (words
      on both sides, zero shared tokens) vetoes ambiguous geometry.
    * Weak/empty text is neutral: geometry alone decides.
    """
    matched, _ = match_reason(feats)
    return matched


def match_candidates(dom_list, gpa_list):
    """Greedy 1:1 matching, best score first (score = IoU + containment
    + 0.5 text bonus; ties -> DOM order, then GPA order). Each side used
    at most once. Returns (matches, dom_only, gpa_only)."""
    scored = []
    for di, d in enumerate(dom_list):
        for gi, g in enumerate(gpa_list):
            f = pair_features(d, g)
            if is_match(f):
                scored.append((f["iou"] + f["containment"]
                               + (0.5 if f["text_sim"]
                                  >= MATCH_TEXT_SIM_THRESHOLD else 0.0),
                               di, gi, f))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_d, used_g, matches = set(), set(), []
    for score, di, gi, f in scored:
        if di in used_d or gi in used_g:
            continue
        used_d.add(di)
        used_g.add(gi)
        matches.append((dom_list[di], gpa_list[gi], f))
    dom_only = [d for i, d in enumerate(dom_list) if i not in used_d]
    gpa_only = [g for i, g in enumerate(gpa_list) if i not in used_g]
    return matches, dom_only, gpa_only


def _union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]),
            max(a[2], b[2]), max(a[3], b[3])]


def generate_hybrid_som(dom_list, gpa_list):
    """Associate + union. Matched -> MERGE (geometric union, provenance
    kept); DOM-only -> KEEP; GPA-only -> KEEP. NO new NMS: GPA boxes
    already passed gpa_postprocess.py; DOM boxes need only clip/zero-area
    validity (applied at extraction). Isolated entry point.

    Returns (final_elements, debug) with elements:
      {id, source, bbox, dom_bbox|None, gpa_bbox|None, text, role,
       confidence}.
    """
    matches, dom_only, gpa_only = match_candidates(dom_list, gpa_list)
    final = []
    for d, g, f in matches:
        final.append({"source": "dom+gpa",
                      "bbox": _union(d["bbox"], g["bbox"]),
                      "dom_bbox": d["bbox"], "gpa_bbox": g["bbox"],
                      "text": d.get("text", "") or "",
                      "role": d.get("role", ""),
                      "tag": d.get("tag", ""),
                      "element_id": d.get("element_id", ""),
                      "gpa_id": g.get("gpa_id"),
                      "confidence": g.get("confidence"),
                      "label": g.get("label") or ""})
    for d in dom_only:
        final.append({"source": "dom", "bbox": d["bbox"],
                      "dom_bbox": d["bbox"], "gpa_bbox": None,
                      "text": d.get("text", "") or "",
                      "role": d.get("role", ""), "tag": d.get("tag", ""),
                      "element_id": d.get("element_id", ""),
                      # DOM extraction is deterministic (no sampling), so
                      # DOM-only candidates are recorded at 1.0 by stated
                      # convention, NOT as a measured model confidence.
                      "confidence": 1.0, "label": ""})
    for g in gpa_only:
        final.append({"source": "gpa", "bbox": g["bbox"],
                      "dom_bbox": None, "gpa_bbox": g["bbox"],
                      "text": "", "role": "", "tag": "",
                      "element_id": "", "gpa_id": g.get("gpa_id"),
                      "confidence": g.get("confidence"),
                      "label": g.get("label") or ""})
    # Deterministic order: reading order (top, then left).
    final.sort(key=lambda e: (e["bbox"][1], e["bbox"][0]))
    for j, e in enumerate(final, start=1):
        e["id"] = j
    debug = {"n_dom": len(dom_list), "n_gpa": len(gpa_list),
             "n_matched": len(matches), "n_dom_only": len(dom_only),
             "n_gpa_only": len(gpa_only), "n_final": len(final),
              "matches": [{"dom": m[0]["element_id"],
                           "gpa_id": m[1].get("gpa_id"),
                           "dom_bbox": m[0]["bbox"],
                           "gpa_bbox": m[1]["bbox"],
                           "dom_text": m[0].get("text", "") or "",
                           "gpa_text": m[1].get("ocr_hint", "") or "",
                           "iou": round(m[2]["iou"], 3),
                           "containment": round(m[2]["containment"], 3),
                           "norm_center": round(m[2]["norm_center"], 3),
                           "text_sim": round(m[2]["text_sim"], 3),
                           "reason": match_reason(m[2])[1],
                           "final_bbox": _union(m[0]["bbox"],
                                                m[1]["bbox"])}
                          for m in matches]}
    return final, debug


# --------------------------------------------------------------------------
# OCR (EXISTING ocr_pipeline implementation, per final hybrid bbox)
# --------------------------------------------------------------------------

def ocr_final_elements(image, elements, out_dir, cpu: bool = True):
    """Crop each FINAL hybrid bbox + ocr_pipeline.ocr_crop. Returns the
    hybrid ocr_results entries (spec section 12 shape)."""
    _ensure_repo_imports()
    import ocr_pipeline as oc

    crops_dir = out_dir / "crops"
    ocr = oc.initialize_ocr(OCR_ENGINE, cpu=cpu)
    oc.crop_all_elements(image, elements, crops_dir, PADDING_RATIO)
    from PIL import Image as _Image
    for e in elements:
        crop = _Image.open(e["crop_path"]).convert("RGB")
        try:
            text, conf, words = oc.ocr_crop(ocr, crop,
                                            f"element_{e['id']:04d}")
        except Exception as ex:
            log.error("Element %04d OCR failed: %s", e["id"], ex)
            text, conf, words = "", 0.0, []
        e["ocr_text"], e["ocr_confidence"], e["words"] = text, conf, words
        log.info("[%d/%d] element_%04d.png (%s) -> %r (conf=%.2f)",
                 e["id"], len(elements), e["id"], e["source"], text[:50],
                 conf)
    try:
        rel = str(crops_dir.relative_to(out_dir))
    except ValueError:
        rel = crops_dir.name
    return [{
        "id": e["id"], "bbox": e["bbox"], "source": e["source"],
        "dom_bbox": e["dom_bbox"], "gpa_bbox": e["gpa_bbox"],
        "text": e.get("text", ""), "role": e.get("role", ""),
        "ocr_text": e.get("ocr_text", ""),
        "ocr_confidence": e.get("ocr_confidence", 0.0),
        "crop_path": f"{rel}/element_{e['id']:04d}.png",
        "words": e.get("words", []),
    } for e in elements]


# --------------------------------------------------------------------------
# Debug visualization (NEW image; existing viz untouched)
# --------------------------------------------------------------------------

COLORS = {"dom": (37, 99, 235), "gpa": (220, 38, 38),
          "dom+gpa": (22, 163, 74)}


def draw_hybrid_debug(image, elements, path: Path) -> None:
    """Blue = DOM-only, red = GPA-only, green = DOM+GPA merged."""
    from PIL import ImageDraw

    vis = image.copy()
    d = ImageDraw.Draw(vis)
    W, H = image.size
    for e in elements:
        x1, y1, x2, y2 = e["bbox"]
        color = COLORS.get(e["source"], (255, 0, 0))
        d.rectangle([x1, y1, x2, y2], outline=color, width=2)
        if e["source"] == "dom+gpa":
            tag = (f"[{e['id']}] DOM #{e.get('element_id', '?')}"
                   f"+GPA #{e.get('gpa_id', '?')}")
        elif e["source"] == "dom":
            tag = f"[{e['id']}] DOM #{e.get('element_id', '?')}"
        else:
            tag = f"[{e['id']}] GPA #{e.get('gpa_id', '?')}"
        tx, ty = min(x2 + 3, W - 1), max(0, y1 - 12)
        d.rectangle([tx, ty, min(W, tx + 8 + 7 * len(tag)), ty + 13],
                    fill=color)
        d.text((tx + 4, ty + 1), tag, fill=(255, 255, 255))
    vis.save(path)
    log.info("Saved debug visualization: %s", path)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Experimental hybrid_som: DOM + GPA -> match -> OCR.")
    ap.add_argument("--html", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--viewport", default=None,
                    help="WxH (default 1280x800)")
    ap.add_argument("--wait", type=float, default=None)
    ap.add_argument("--cpu", action="store_true", default=True)
    ap.add_argument("--no-cpu", dest="cpu", action="store_false")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    setup_logging()
    args = parse_args(argv)

    html_arg = args.html or INPUT_HTML
    out_dir = Path(args.output or OUTPUT_DIR)
    if not out_dir.is_absolute():
        cand, alt = Path.cwd() / out_dir, SCRIPT_DIR / out_dir
        out_dir = cand if cand == alt else (
            alt if not cand.exists() and alt.exists() else cand)
    out_dir.mkdir(parents=True, exist_ok=True)
    vw, vh = VIEWPORT_WIDTH, VIEWPORT_HEIGHT
    if args.viewport:
        vw, vh = (int(v) for v in str(args.viewport).lower().split("x"))
    wait_s = WAIT_AFTER_LOAD if args.wait is None else args.wait

    page, shot, metrics, stop = load_browser_page(html_arg, vw, vh,
                                                  wait_s)
    try:
        shot_path = out_dir / "screenshot.png"
        shot.save(shot_path)
        dom_list = extract_dom_candidates(page, metrics)
    finally:
        stop()
    gpa_list = run_gpa_candidates(shot)

    final, debug = generate_hybrid_som(dom_list, gpa_list)

    (out_dir / "hybrid_bboxes.json").write_text(json.dumps(
        [{"id": e["id"], "source": e["source"], "bbox": e["bbox"],
          "dom_bbox": e["dom_bbox"], "gpa_bbox": e["gpa_bbox"],
          "text": e["text"], "role": e["role"], "tag": e["tag"],
          "element_id": e["element_id"], "gpa_id": e.get("gpa_id"),
          "confidence": e["confidence"]} for e in final], indent=4))
    draw_hybrid_debug(shot, final, out_dir / "hybrid_som_debug.png")
    entries = ocr_final_elements(shot, final, out_dir, cpu=args.cpu)
    (out_dir / "hybrid_ocr_results.json").write_text(json.dumps(
        entries, indent=4))

    print("=" * 40)
    print("HYBRID SOM TEST")
    print("===============")
    print()
    print(f"DOM candidates:       {debug['n_dom']}")
    print(f"GPA candidates:       {debug['n_gpa']}")
    print(f"Matched:              {debug['n_matched']}")
    print(f"DOM-only:             {debug['n_dom_only']}")
    print(f"GPA-only:             {debug['n_gpa_only']}")
    print(f"Final hybrid:         {debug['n_final']}")
    print()
    print("========================================")
    print()
    for e in entries:
        x1, y1, x2, y2 = e["bbox"]
        print(f"{e['id']} | {e['source']} | [{x1},{y1},{x2},{y2}] | "
              f"{e['ocr_text']}")
    print()
    print("---------- MATCHED PAIRS ----------")
    for m in debug["matches"]:
        x1, y1, x2, y2 = m["gpa_bbox"]
        dx1, dy1, dx2, dy2 = m["dom_bbox"]
        import math
        dcx, dcy = (dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0
        gcx, gcy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        print(f"MATCH\n"
              f"  DOM #{m['dom']} <-> GPA #{m['gpa_id']}\n"
              f"  DOM bbox {m['dom_bbox']} GPA bbox {m['gpa_bbox']}\n"
              f"  IoU={m['iou']} containment={m['containment']} "
              f"center distance={math.hypot(dcx - gcx, dcy - gcy):.1f}px "
              f"(norm {m['norm_center']})\n"
              f"  DOM text={m['dom_text']!r} GPA text={m['gpa_text']!r} "
              f"text similarity={m['text_sim']}\n"
              f"  reason={m['reason']}\n"
              f"  final_bbox={m.get('final_bbox')}")
    print()
    print(f"Output: {out_dir}")
    return 0


if __name__ == "__main__":
    main()
