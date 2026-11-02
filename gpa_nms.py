#!/usr/bin/env python3
"""GPA-only GUI-element bbox cleanup: raw detections -> staged filtering.

Pipeline (each stage separate and inspectable):
    Stage A: confidence filtering      (MIN_CONFIDENCE)
    Stage B: class-agnostic IoU NMS    (hard default, soft optional)
    Stage C: containment analysis      (no removals; small boxes never cut)
    Stage D: parent-box filtering      (parents with >=2 children removed)
    Stage E: final duplicate check     (near-identical coordinates)

NO OCR. NO VLM. NO other detector. Only:
    GPA GUI Detector (Salesforce/GPA-GUI-Detector, ultralytics YOLO).

Run:
    python gpa_nms.py
    python gpa_nms.py --input Ocr/input.png --output Ocr/test_8
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration (top of file -- all thresholds live here)
# --------------------------------------------------------------------------

INPUT_IMAGE = "Ocr/input.png"
OUTPUT_DIR = "Ocr/test_8"

GPA_MODEL = "pretrained/detectors/gpa/model.pt"
# Detector-internal NMS IoU (ultralytics always applies NMS inside
# predict()). Kept loose at the model-card value 0.7 so Stage B sees the
# full proposal set; the REAL dedup threshold is NMS_IOU_THRESHOLD.
GPA_DETECTOR_IOU = 0.7

MIN_CONFIDENCE = 0.05
NMS_IOU_THRESHOLD = 0.50
CONTAINMENT_THRESHOLD = 0.90
COORDINATE_SIMILARITY_THRESHOLD = 0.95
MIN_CHILDREN_FOR_PARENT_REMOVAL = 2
MIN_BOX_AREA = 20
NMS_METHOD = "hard"               # "hard" (default baseline) or "soft"

LOG_LEVEL = "INFO"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SCRIPT_DIR = Path(__file__).resolve().parent

# stdlib `logging` first (repo ships src/logging/ which must never shadow it;
# this script never adds src/ to sys.path anyway).
log = logging.getLogger("gpa_nms")


def setup_logging(level: str = LOG_LEVEL) -> None:
    logging.basicConfig(level=getattr(logging, str(level).upper(),
                                      logging.INFO),
                        format="%(levelname)s: %(message)s")


def resolve(p: str) -> Path:
    q = Path(p)
    if q.exists() or q.is_absolute():
        return q
    return SCRIPT_DIR / q


# --------------------------------------------------------------------------
# Geometry (pure, deterministic)
# --------------------------------------------------------------------------

def box_area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter_area(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def box_iou(a, b) -> float:
    inter = inter_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(a_in_b) -> float:
    """intersection_area / area(A): is A contained in B?"""
    a, b = a_in_b
    return inter_area(a, b) / max(1e-9, box_area(a))


# --------------------------------------------------------------------------
# Input + normalization
# --------------------------------------------------------------------------

def load_image(path: str):
    from PIL import Image

    p = resolve(path)
    if not p.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format {p.suffix!r}.")
    return Image.open(p).convert("RGB"), p  # original resolution, untouched


def normalize_box(raw_xyxy, image_size):
    """Clip to ORIGINAL pixels; reject zero/negative-area, outside-image,
    NaN/Inf, and sub-MIN_BOX_AREA noise boxes with explicit reasons."""
    import math

    W, H = image_size
    try:
        x1, y1, x2, y2 = (float(v) for v in raw_xyxy)
    except (TypeError, ValueError):
        return False, None, "non-numeric coordinates"
    if any(math.isnan(v) or math.isinf(v) for v in (x1, y1, x2, y2)):
        return False, None, "NaN/Inf coordinates"
    if not (x1 < x2 and y1 < y2):
        return False, None, "negative or zero width/height"
    ix1, iy1 = max(0, int(round(x1))), max(0, int(round(y1)))
    ix2, iy2 = min(W, int(round(x2))), min(H, int(round(y2)))
    if ix2 - ix1 < 2 or iy2 - iy1 < 2:
        return False, None, "empty after clipping to image boundaries"
    if (ix2 - ix1) * (iy2 - iy1) < MIN_BOX_AREA:
        return False, None, f"area below MIN_BOX_AREA={MIN_BOX_AREA}"
    if not (0 <= ix1 < ix2 <= W and 0 <= iy1 < iy2 <= H):
        return False, None, "outside image boundaries"
    return True, [ix1, iy1, ix2, iy2], "ok"


def renumber(elements):
    for j, e in enumerate(elements, start=1):
        e["id"] = j
    return elements


# --------------------------------------------------------------------------
# GPA detector (official per-model-card usage)
# --------------------------------------------------------------------------

def run_gpa(image, image_size, model_path, conf, nms_iou, device):
    from ultralytics import YOLO

    pt = resolve(model_path)
    if not pt.is_file():
        raise RuntimeError(f"[GPA] missing weight file: {pt}")
    log.info("Loading GPA detector from %s ...", pt)
    model = YOLO(str(pt))
    try:
        names = dict(model.names)
    except Exception:
        names = {}
    log.info("GPA classes: %s", names or "unknown")
    log.info("Running GPA (conf=%.2f, iou=%.2f, device=%s)...",
             conf, nms_iou, device)
    res = model.predict(image, conf=conf, iou=nms_iou, verbose=False,
                        device=device)[0]
    out, n_raw = [], 0
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        sc = res.boxes.conf.cpu().numpy()
        try:
            cl = res.boxes.cls.cpu().numpy().astype(int)
        except Exception:
            cl = [None] * len(xyxy)
        n_raw = len(xyxy)
        for b, s, c in zip(xyxy, sc, cl):
            if (b[2] - b[0]) <= 1 or (b[3] - b[1]) <= 1:
                continue
            ok, clean, reason = normalize_box([float(v) for v in b],
                                              image_size)
            if not ok:
                log.warning("[gpa] rejecting box %r: %s",
                            [round(float(v), 1) for v in b], reason)
                continue
            label = str(names.get(int(c), "element")) if c is not None \
                else "element"
            out.append({"id": 0, "bbox": clean, "confidence": float(s),
                        "label": label})
    log.info("GPA raw detections: %d (kept %d after normalization).",
             n_raw, len(out))
    return out, n_raw


# --------------------------------------------------------------------------
# Stage A -- confidence filtering
# --------------------------------------------------------------------------

def confidence_filter(dets, min_conf=MIN_CONFIDENCE):
    kept = [dict(d) for d in dets if (d["confidence"] or 0.0) >= min_conf]
    log.info("Stage A: %d -> %d (MIN_CONFIDENCE=%.2f).",
             len(dets), len(kept), min_conf)
    return renumber(kept)


# --------------------------------------------------------------------------
# Stage B -- class-agnostic IoU NMS (hard default, soft optional)
# --------------------------------------------------------------------------

def iou_nms(dets, iou_thr=NMS_IOU_THRESHOLD, method=NMS_METHOD,
            min_conf=MIN_CONFIDENCE):
    """Greedy, confidence-descending, CLASS-AGNOSTIC (labels ignored).

    hard: drop boxes with IoU > thr vs a kept box.
    soft (linear): decay score *= (1 - IoU) instead of deleting; drop
      boxes whose decayed score falls below min_conf. Default is hard.
    """
    if method not in ("hard", "soft"):
        raise ValueError(f"NMS_METHOD must be 'hard' or 'soft', "
                         f"got {method!r}.")
    scores = [float(d["confidence"] or 0.0) for d in dets]
    if method == "hard":
        order = sorted(range(len(dets)),
                       key=lambda k: (-scores[k], k))
        kept_idx, suppressed = [], 0
        for k in order:
            if any(box_iou(dets[k]["bbox"], dets[q]["bbox"]) > iou_thr
                   for q in kept_idx):
                suppressed += 1
                continue
            kept_idx.append(k)
        kept = [dict(dets[k]) for k in kept_idx]
        log.info("Stage B (hard, IoU>%.2f): %d -> %d.", iou_thr,
                 len(dets), len(kept))
        return renumber(kept)
    # soft (linear): each round selects the current max, decays the rest
    # by (1 - IoU) where IoU > thr, and drops decayed scores < min_conf.
    remaining = list(range(len(dets)))
    kept_idx, suppressed = [], 0
    while remaining:
        m = max(remaining, key=lambda k: (scores[k], -k))
        remaining.remove(m)
        if scores[m] < min_conf:
            suppressed += 1
            continue
        kept_idx.append(m)
        for k in list(remaining):
            v = box_iou(dets[k]["bbox"], dets[m]["bbox"])
            if v > iou_thr:
                scores[k] *= (1.0 - v)
    kept = []
    for k in kept_idx:
        d = dict(dets[k])
        d["confidence"] = round(scores[k], 4)
        kept.append(d)
    log.info("Stage B (soft, IoU>%.2f): %d -> %d.", iou_thr,
             len(dets), len(kept))
    return renumber(kept)


# --------------------------------------------------------------------------
# Stage C -- containment analysis (NO removals here by design)
# --------------------------------------------------------------------------

def containment_analysis(dets, contain_thr=CONTAINMENT_THRESHOLD):
    """For every pair, containment ratios both ways. Small boxes are NEVER
    removed for being inside a big box -- this stage only measures and
    reports; removals happen in Stage D."""
    pairs = 0
    for i in range(len(dets)):
        for j in range(len(dets)):
            if i == j:
                continue
            if containment((dets[i]["bbox"], dets[j]["bbox"])) > contain_thr:
                pairs += 1
    log.info("Stage C: %d contained-pairs (ratio>%.2f); no removals.",
             pairs, contain_thr)
    return list(dets), pairs


# --------------------------------------------------------------------------
# Stage D -- parent-box filtering (small boxes ALWAYS survive)
# --------------------------------------------------------------------------

def parent_filter(dets, contain_thr=CONTAINMENT_THRESHOLD,
                  min_children=MIN_CHILDREN_FOR_PARENT_REMOVAL):
    """Remove a large box containing >= min_children smaller boxes
    (containment > thr each). A parent with 0-1 children is kept. A small
    box is NEVER removed for being contained (buttons, inputs, text,
    prices, nav, search boxes all survive)."""
    drop = set()
    for i, big in enumerate(dets):
        kids = [j for j, sm in enumerate(dets) if j != i and
                containment((sm["bbox"], big["bbox"])) > contain_thr]
        if len(kids) >= min_children:
            drop.add(i)
            log.info("Parent-filter removing box %r (conf=%.2f, %d "
                     "children).", big["bbox"], big["confidence"] or 0.0,
                     len(kids))
    kept = [d for i, d in enumerate(dets) if i not in drop]
    log.info("Stage D: removed %d parent boxes (min_children=%d).",
             len(drop), min_children)
    return renumber(kept), len(drop)


# --------------------------------------------------------------------------
# Stage E -- final near-duplicate check
# --------------------------------------------------------------------------

def final_duplicate_check(dets, sim_thr=COORDINATE_SIMILARITY_THRESHOLD):
    """Near-identical coordinates (IoU > sim_thr -- mathematically
    equivalent to normalized coordinate agreement within a few px).
    Keeps higher confidence. With hard NMS@0.50 upstream this normally
    finds 0 (reported honestly)."""
    order = sorted(range(len(dets)),
                   key=lambda k: (-(dets[k]["confidence"] or 0.0), k))
    kept, removed = [], 0
    for k in order:
        if any(box_iou(dets[k]["bbox"], q["bbox"]) > sim_thr for q in kept):
            removed += 1
            continue
        kept.append(dets[k])
    log.info("Stage E: removed %d near-duplicates (IoU>%.2f).", removed,
             sim_thr)
    return renumber(kept), removed


# --------------------------------------------------------------------------
# Per-stage saves + visualization + stats + main
# --------------------------------------------------------------------------

def save_list(path: Path, elements) -> None:
    path.write_text(json.dumps(elements, indent=4))
    log.info("Saved %s (%d).", path, len(elements))


def draw_stage(image, elements, path: Path) -> None:
    """ORIGINAL-size visualization: red box + [id] + confidence."""
    from PIL import ImageDraw

    vis = image.copy()
    d = ImageDraw.Draw(vis)
    W, H = image.size
    for e in elements:
        x1, y1, x2, y2 = e["bbox"]
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        tag = f"[{e['id']}] {e['confidence']:.2f}"
        tx, ty = min(x2 + 3, W - 1), max(0, y1 - 12)
        d.rectangle([tx, ty, min(W, tx + 8 + 7 * len(tag)), ty + 13],
                    fill=(255, 0, 0))
        d.text((tx + 4, ty + 1), tag, fill=(255, 255, 255))
    vis.save(path)
    log.info("Saved visualization: %s", path)


def print_stats(n_raw, n_conf, n_nms, n_cont, n_parents_removed,
                n_dup_removed, final) -> None:
    print("=" * 40)
    print("GPA DETECTION")
    print("=" * 40)
    print()
    print(f"{'Raw detections:':<27}{n_raw}")
    print()
    print(f"{'After confidence filtering:':<27}{n_conf}")
    print()
    print(f"{'After IoU NMS:':<27}{n_nms}")
    print()
    print(f"{'Contained-box analysis:':<27}{n_cont}")
    print()
    print(f"{'Parent boxes removed:':<27}{n_parents_removed}")
    print()
    print(f"{'Final duplicate check:':<27}{n_dup_removed}")
    print()
    print(f"{'FINAL BOXES:':<27}{len(final)}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="GPA-only GUI bbox cleanup (no OCR, staged).")
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--gpa-model", default=None)
    ap.add_argument("--min-confidence", type=float, default=None)
    ap.add_argument("--nms-iou", type=float, default=None)
    ap.add_argument("--nms-method", default=None, choices=["hard", "soft"])
    ap.add_argument("--contain", type=float, default=None)
    ap.add_argument("--coord-sim", type=float, default=None)
    ap.add_argument("--min-children", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    setup_logging()
    args = parse_args(argv)

    input_arg = args.input or INPUT_IMAGE
    out_dir = Path(args.output or OUTPUT_DIR)
    if not out_dir.is_absolute():
        cand, alt = Path.cwd() / out_dir, SCRIPT_DIR / out_dir
        out_dir = cand if cand == alt else (
            alt if not cand.exists() and alt.exists() else cand)
    out_dir.mkdir(parents=True, exist_ok=True)

    min_conf = MIN_CONFIDENCE if args.min_confidence is None \
        else args.min_confidence
    iou_thr = NMS_IOU_THRESHOLD if args.nms_iou is None else args.nms_iou
    method = (args.nms_method or NMS_METHOD).lower()
    contain = CONTAINMENT_THRESHOLD if args.contain is None else args.contain
    sim_thr = COORDINATE_SIMILARITY_THRESHOLD if args.coord_sim is None \
        else args.coord_sim
    min_children = MIN_CHILDREN_FOR_PARENT_REMOVAL \
        if args.min_children is None else args.min_children

    try:
        image, input_path = load_image(input_arg)
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 2
    W, H = image.size
    try:
        shutil.copy2(input_path, out_dir / input_path.name)
    except Exception as e:
        log.error("Could not copy input image: %s", e)
        return 2

    # Raw GPA detections (detector conf floor = MIN_CONFIDENCE: cleanup
    # must come from NMS/containment, not aggressive pre-filtering).
    try:
        raw, n_raw = run_gpa(image, (W, H),
                             args.gpa_model or GPA_MODEL,
                             min_conf, GPA_DETECTOR_IOU, args.device)
    except Exception as e:
        log.error("GPA detection failed: %s", e)
        return 2
    save_list(out_dir / "gpa_raw_detections.json", renumber(raw))
    draw_stage(image, renumber(raw), out_dir / "gpa_raw_visualization.png")

    # Stage A.
    after_conf = confidence_filter(raw, min_conf)
    save_list(out_dir / "gpa_after_confidence.json", after_conf)

    # Stage B.
    after_nms = iou_nms(after_conf, iou_thr, method, min_conf)
    save_list(out_dir / "gpa_after_nms.json", after_nms)
    draw_stage(image, after_nms, out_dir / "gpa_nms_visualization.png")

    # Stage C (analysis only) + Stage D.
    after_cont, _pairs = containment_analysis(after_nms, contain)
    save_list(out_dir / "gpa_after_containment.json", after_cont)
    final, n_parents = parent_filter(after_cont, contain, min_children)

    # Stage E.
    final, n_dup = final_duplicate_check(final, sim_thr)
    save_list(out_dir / "gpa_final_bboxes.json", {
        "image": input_path.name, "image_width": W, "image_height": H,
        "detections": final})
    draw_stage(image, final, out_dir / "gpa_final_visualization.png")

    print_stats(n_raw, len(after_conf), len(after_nms), len(after_cont),
                n_parents, n_dup, final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
