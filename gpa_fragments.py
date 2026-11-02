#!/usr/bin/env python3
"""GPA-only geometry post-processing: containment -> fragment merging ->
duplicates -> final NMS. Produces ONE bbox per logical UI element.

Order (never standard-NMS-first):
    GPA detections -> basic filtering -> containment filtering ->
    horizontal fragments -> vertical fragments -> duplicate merging ->
    final NMS -> clean boxes.

NO OCR. NO VLM. GPA detector itself is NOT changed (official model-card
usage only).

Run:
    python gpa_fragments.py
    python gpa_fragments.py --input Ocr/input.png --output Ocr/test_13
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration (all parameters at the top)
# --------------------------------------------------------------------------

INPUT_IMAGE = "Ocr/input.png"
OUTPUT_DIR = "Ocr/test_13"

GPA_MODEL = "pretrained/detectors/gpa/model.pt"
DETECTOR_CONF = 0.05            # loose detector floor; REAL cut is below
DETECTOR_INTERNAL_NMS_IOU = 0.70

CONF_THRESHOLD = 0.20

DUPLICATE_IOU_THRESHOLD = 0.80
CONTAINMENT_THRESHOLD = 0.95

HORIZONTAL_OVERLAP_THRESHOLD = 0.50
VERTICAL_OVERLAP_THRESHOLD = 0.50

FRAGMENT_GAP_FACTOR = 0.20

CENTER_DISTANCE_FACTOR = 0.50

HEIGHT_SIMILARITY_THRESHOLD = 0.50
WIDTH_SIMILARITY_THRESHOLD = 0.50

MIN_BOX_WIDTH = 5
MIN_BOX_HEIGHT = 5

NMS_IOU_THRESHOLD = 0.50
NMS_MODE = "hybrid"               # "hard" | "soft" | "hybrid" (default)

LOG_LEVEL = "INFO"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SCRIPT_DIR = Path(__file__).resolve().parent

# stdlib `logging` first (never add repo src/ to sys.path in this script).
log = logging.getLogger("gpa_frag")


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


def inter_box(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return [ix1, iy1, ix2, iy2]


def inter_area(a, b) -> float:
    ix1, iy1, ix2, iy2 = inter_box(a, b)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def box_iou(a, b) -> float:
    inter = inter_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def containment_ratio(inner, outer) -> float:
    """intersection_area / area(inner): is inner contained in outer?"""
    return inter_area(inner, outer) / max(1e-9, box_area(inner))


def union_box(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]),
            max(a[2], b[2]), max(a[3], b[3])]


# --------------------------------------------------------------------------
# Records (raw_id retained for source_ids debugging)
# --------------------------------------------------------------------------

def new_record(raw_id, bbox, confidence, label):
    return {"raw_id": raw_id, "bbox": list(bbox),
            "confidence": float(confidence), "label": label,
            "source_ids": [raw_id], "merged": False}


def load_image(path: str):
    from PIL import Image

    p = resolve(path)
    if not p.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format {p.suffix!r}.")
    return Image.open(p).convert("RGB"), p


def run_gpa_detector(image, model_path, det_conf, det_iou, device):
    """Official model-card usage. Returns raw [(bbox floats, conf, cls)]."""
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
    log.info("Running GPA (conf=%.2f, internal_nms_iou=%.2f, device=%s)...",
             det_conf, det_iou, device)
    res = model.predict(image, conf=det_conf, iou=det_iou, verbose=False,
                        device=device)[0]
    out = []
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        sc = res.boxes.conf.cpu().numpy()
        try:
            cl = res.boxes.cls.cpu().numpy().astype(int)
        except Exception:
            cl = [None] * len(xyxy)
        for b, s, c in zip(xyxy, sc, cl):
            label = str(names.get(int(c), "element")) \
                if c is not None else "element"
            out.append(([round(float(v), 1) for v in b], float(s), label))
    log.info("GPA raw detections: %d.", len(out))
    return out


# --------------------------------------------------------------------------
# STAGE 1 -- basic filtering (ORIGINAL coordinates, clipped, never resized)
# --------------------------------------------------------------------------

def stage_basic(raw, image_size, conf_thr=CONF_THRESHOLD):
    import math

    W, H = image_size
    kept = []
    for i, (b, s, label) in enumerate(raw, start=1):
        try:
            x1, y1, x2, y2 = (float(v) for v in b)
        except (TypeError, ValueError):
            log.warning("Stage 1 dropping id=%d: non-numeric %r.", i, b)
            continue
        if any(math.isnan(v) or math.isinf(v) for v in (x1, y1, x2, y2)):
            log.warning("Stage 1 dropping id=%d: NaN/Inf.", i)
            continue
        c = [max(0, x1), max(0, y1), min(float(W), x2),
             min(float(H), y2)]
        if (c[2] - c[0]) < MIN_BOX_WIDTH or \
                (c[3] - c[1]) < MIN_BOX_HEIGHT or c[2] <= c[0] or \
                c[3] <= c[1]:
            log.warning("Stage 1 dropping id=%d: invalid size %r.", i, b)
            continue
        if s < conf_thr:
            continue
        kept.append(new_record(i, [int(round(v)) for v in c], s, label))
    log.info("Stage 1: %d -> %d (CONF_THRESHOLD=%.2f).",
             len(raw), len(kept), conf_thr)
    return kept


# --------------------------------------------------------------------------
# STAGE 2 -- containment filtering (remove grouping containers only)
# --------------------------------------------------------------------------

def stage_containment(boxes, contain_thr=CONTAINMENT_THRESHOLD):
    """Remove an outer box holding >=1 meaningful child (containment >=
    thr). Inner detections are ALWAYS preserved. Simultaneous marking, so
    chains (card > wide price > tight price) collapse correctly."""
    children_of = {}
    for i, outer in enumerate(boxes):
        kids = [j for j, inner in enumerate(boxes) if j != i and
                containment_ratio(inner["bbox"], outer["bbox"])
                >= contain_thr]
        if kids:
            children_of[i] = kids
    drop = set(children_of)
    kept = []
    for i, b in enumerate(boxes):
        if i in drop:
            log.info("REMOVE:\n  box %d\n  reason: contains boxes %s\n"
                     "  containment_ratio: %s",
                     b["raw_id"],
                     [boxes[j]["raw_id"] for j in children_of[i]],
                     round(max(containment_ratio(
                         boxes[j]["bbox"], b["bbox"])
                         for j in children_of[i]), 3))
            continue
        kept.append(b)
    log.info("Stage 2: %d -> %d containers removed.", len(boxes),
             len(drop))
    return kept


# --------------------------------------------------------------------------
# STAGES 3+4 -- fragment merging (union-find, iterative by construction)
# --------------------------------------------------------------------------

class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _h_compatible(a, b):
    """Horizontal-fragment test (spec STAGE 3, all conditions)."""
    ax1, ay1, ax2, ay2 = a["bbox"]
    bx1, by1, bx2, by2 = b["bbox"]
    ha, hb = ay2 - ay1, by2 - by1
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    if inter_h / max(1e-9, min(ha, hb)) < HORIZONTAL_OVERLAP_THRESHOLD:
        return False
    avg_h = (ha + hb) / 2.0
    cya, cyb = (ay1 + ay2) / 2.0, (by1 + by2) / 2.0
    if abs(cya - cyb) > CENTER_DISTANCE_FACTOR * avg_h:
        return False
    gap = max(ax1, bx1) - min(ax2, bx2)  # negative == overlap (allowed)
    if gap > FRAGMENT_GAP_FACTOR * avg_h:
        return False
    if min(ha, hb) / max(ha, hb) < HEIGHT_SIMILARITY_THRESHOLD:
        return False
    return True


def _v_compatible(a, b):
    """Vertical-fragment test (spec STAGE 4, mirrored)."""
    ax1, ay1, ax2, ay2 = a["bbox"]
    bx1, by1, bx2, by2 = b["bbox"]
    wa, wb = ax2 - ax1, bx2 - bx1
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    if inter_w / max(1e-9, min(wa, wb)) < VERTICAL_OVERLAP_THRESHOLD:
        return False
    avg_w = (wa + wb) / 2.0
    cxa, cxb = (ax1 + ax2) / 2.0, (bx1 + bx2) / 2.0
    if abs(cxa - cxb) > CENTER_DISTANCE_FACTOR * avg_w:
        return False
    gap = max(ay1, by1) - min(ay2, by2)
    if gap > FRAGMENT_GAP_FACTOR * avg_w:
        return False
    if min(wa, wb) / max(wa, wb) < WIDTH_SIMILARITY_THRESHOLD:
        return False
    return True


def _merge_components(boxes, compatible, reason):
    """Union-find over compatible pairs; each component -> union bbox,
    max confidence, union of source_ids. Transitive chains (B1+B2->M1,
    M1+B3->M2) collapse in ONE pass by construction."""
    dsu = _DSU(len(boxes))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if compatible(boxes[i], boxes[j]):
                dsu.union(i, j)
    comps = {}
    for i in range(len(boxes)):
        comps.setdefault(dsu.find(i), []).append(i)
    out = []
    for members in comps.values():
        if len(members) == 1:
            out.append(boxes[members[0]])
            continue
        ub = boxes[members[0]]["bbox"]
        for m in members[1:]:
            ub = union_box(ub, boxes[m]["bbox"])
        best = max(members, key=lambda m: (boxes[m]["confidence"],
                                           -boxes[m]["raw_id"]))
        sids = []
        for m in members:
            sids += [s for s in boxes[m]["source_ids"] if s not in sids]
        print(f"MERGE:\n  boxes {sorted(boxes[m]['raw_id'] for m in members)}\n"
              f"  reason: {reason}\n  final_bbox: {[int(v) for v in ub]}")
        log.info("Merged %d boxes (%s) -> %r.",
                 len(members), reason, [int(v) for v in ub])
        out.append({"raw_id": boxes[best]["raw_id"], "bbox": ub,
                    "confidence": boxes[best]["confidence"],
                    "label": boxes[best]["label"], "source_ids": sids,
                    "merged": True})
    # Preserve input order (first-member order) for determinism.
    return out


def stage_horizontal_fragments(boxes):
    out = _merge_components(boxes, _h_compatible, "horizontal_fragment")
    log.info("Stage 3: %d -> %d.", len(boxes), len(out))
    return out


def stage_vertical_fragments(boxes):
    out = _merge_components(boxes, _v_compatible, "vertical_fragment")
    log.info("Stage 4: %d -> %d.", len(boxes), len(out))
    return out


# --------------------------------------------------------------------------
# STAGE 5 -- duplicate / near-duplicate merging (keep higher confidence)
# --------------------------------------------------------------------------

def stage_duplicates(boxes, iou_thr=DUPLICATE_IOU_THRESHOLD):
    order = sorted(range(len(boxes)),
                   key=lambda k: (-boxes[k]["confidence"],
                                  boxes[k]["raw_id"]))
    kept = []
    for k in order:
        r = boxes[k]
        dup = None
        for q in kept:
            if box_iou(r["bbox"], q["bbox"]) >= iou_thr:
                dup = q
                break
        if dup is None:
            kept.append(r)
            continue
        for s in r["source_ids"]:
            if s not in dup["source_ids"]:
                dup["source_ids"].append(s)
        dup["merged"] = True
        log.info("Duplicate: box %d merged into %d (IoU=%.3f).",
                 r["raw_id"], dup["raw_id"],
                 box_iou(r["bbox"], dup["bbox"]))
    log.info("Stage 5: %d -> %d (IoU>=%.2f).", len(boxes), len(kept),
             iou_thr)
    return kept


# --------------------------------------------------------------------------
# STAGE 6 -- final NMS (duplicates/conflicts only; NEVER merges fragments)
# --------------------------------------------------------------------------

def stage_final_nms(boxes, iou_thr=NMS_IOU_THRESHOLD, mode=NMS_MODE):
    if mode not in ("hard", "soft", "hybrid"):
        raise ValueError(f"NMS_MODE must be hard/soft/hybrid, got {mode!r}.")
    if mode == "soft":
        scores = [b["confidence"] for b in boxes]
        order = sorted(range(len(boxes)),
                       key=lambda k: (-scores[k], k))
        kept_idx = []
        remaining = list(order)
        while remaining:
            m = max(remaining, key=lambda k: (scores[k], -k))
            remaining.remove(m)
            if scores[m] < CONF_THRESHOLD:
                continue
            kept_idx.append(m)
            for k in list(remaining):
                v = box_iou(boxes[k]["bbox"], boxes[m]["bbox"])
                if v > iou_thr:
                    scores[k] *= (1.0 - v)
        kept = [dict(boxes[k], confidence=round(scores[k], 4))
                for k in kept_idx]
        log.info("Stage 6 (soft, IoU>%.2f): %d -> %d.", iou_thr,
                 len(boxes), len(kept))
        return kept
    # hard + hybrid: greedy delete on IoU (hybrid already merged fragments,
    # removed containers and fused duplicates in dedicated stages).
    order = sorted(range(len(boxes)),
                   key=lambda k: (-boxes[k]["confidence"],
                                  boxes[k]["raw_id"]))
    kept = []
    for k in order:
        if any(box_iou(boxes[k]["bbox"], q["bbox"]) > iou_thr
               for q in kept):
            continue
        kept.append(boxes[k])
    log.info("Stage 6 (%s, IoU>%.2f): %d -> %d.", mode, iou_thr,
             len(boxes), len(kept))
    return kept


# --------------------------------------------------------------------------
# Output: saves, visualization, console, main
# --------------------------------------------------------------------------

def save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=4))
    log.info("Saved %s.", path)


def draw(image, elements, path: Path, color) -> None:
    from PIL import ImageDraw

    vis = image.copy()
    d = ImageDraw.Draw(vis)
    W, H = image.size
    for j, e in enumerate(elements, start=1):
        x1, y1, x2, y2 = e["bbox"]
        d.rectangle([x1, y1, x2, y2], outline=color, width=2)
        tag = f"[{j}]"
        tx, ty = min(x2 + 3, W - 1), max(0, y1 - 12)
        d.rectangle([tx, ty, min(W, tx + 8 + 7 * len(tag)), ty + 13],
                    fill=color)
        d.text((tx + 4, ty + 1), tag, fill=(255, 255, 255))
    vis.save(path)
    log.info("Saved visualization: %s", path)


def envelope(input_name, image_size, elements):
    W, H = image_size
    return {"image": input_name, "image_width": W, "image_height": H,
            "detections": [
                {"id": j, "bbox": [int(v) for v in e["bbox"]],
                 "confidence": round(e["confidence"], 4),
                 "label": e["label"], "source_ids": e["source_ids"],
                 "merged": e["merged"]} for j, e in enumerate(elements,
                                                              start=1)]}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="GPA geometry post-processing (no OCR).")
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--gpa-model", default=None)
    ap.add_argument("--conf-threshold", type=float, default=None)
    ap.add_argument("--dup-iou", type=float, default=None)
    ap.add_argument("--contain", type=float, default=None)
    ap.add_argument("--h-overlap", type=float, default=None)
    ap.add_argument("--v-overlap", type=float, default=None)
    ap.add_argument("--gap-factor", type=float, default=None)
    ap.add_argument("--center-factor", type=float, default=None)
    ap.add_argument("--h-sim", type=float, default=None)
    ap.add_argument("--w-sim", type=float, default=None)
    ap.add_argument("--nms-iou", type=float, default=None)
    ap.add_argument("--nms-mode", default=None,
                    choices=["hard", "soft", "hybrid"])
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
    mode = (args.nms_mode or NMS_MODE).lower()
    if mode not in ("hard", "soft", "hybrid"):
        log.error("NMS_MODE must be hard/soft/hybrid, got %r.", mode)
        return 2
    # Per-run overrides (globals stay untouched).
    global HORIZONTAL_OVERLAP_THRESHOLD, VERTICAL_OVERLAP_THRESHOLD
    global FRAGMENT_GAP_FACTOR, CENTER_DISTANCE_FACTOR
    global HEIGHT_SIMILARITY_THRESHOLD, WIDTH_SIMILARITY_THRESHOLD
    if args.h_overlap is not None:
        HORIZONTAL_OVERLAP_THRESHOLD = args.h_overlap
    if args.v_overlap is not None:
        VERTICAL_OVERLAP_THRESHOLD = args.v_overlap
    if args.gap_factor is not None:
        FRAGMENT_GAP_FACTOR = args.gap_factor
    if args.center_factor is not None:
        CENTER_DISTANCE_FACTOR = args.center_factor
    if args.h_sim is not None:
        HEIGHT_SIMILARITY_THRESHOLD = args.h_sim
    if args.w_sim is not None:
        WIDTH_SIMILARITY_THRESHOLD = args.w_sim
    conf_thr = CONF_THRESHOLD if args.conf_threshold is None \
        else args.conf_threshold
    dup_thr = DUPLICATE_IOU_THRESHOLD if args.dup_iou is None \
        else args.dup_iou
    cont_thr = CONTAINMENT_THRESHOLD if args.contain is None \
        else args.contain
    nms_thr = NMS_IOU_THRESHOLD if args.nms_iou is None else args.nms_iou

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

    det_conf = min(DETECTOR_CONF, CONF_THRESHOLD)
    raw = run_gpa_detector(image, args.gpa_model or GPA_MODEL, det_conf,
                           DETECTOR_INTERNAL_NMS_IOU, args.device)
    n_raw = len(raw)
    save_json(out_dir / "raw_detections.json",
              [{"id": i + 1, "bbox": b,
                "confidence": round(s, 4), "label": lab}
               for i, (b, s, lab) in enumerate(raw)])
    draw(image, [{"bbox": b} for b, _, _ in raw],
         out_dir / "gpa_raw.png", (255, 0, 0))

    s1 = stage_basic(raw, (W, H), conf_thr)
    s2 = stage_containment(s1, cont_thr)
    save_json(out_dir / "after_containment.json",
              envelope(input_path.name, (W, H), s2))
    draw(image, s2, out_dir / "gpa_containment_filtered.png",
         (255, 165, 0))
    s3 = stage_horizontal_fragments(s2)
    s4 = stage_vertical_fragments(s3)
    save_json(out_dir / "after_fragments.json",
              envelope(input_path.name, (W, H), s4))
    draw(image, s4, out_dir / "gpa_fragment_merged.png", (0, 0, 255))
    s5 = stage_duplicates(s4, dup_thr)
    s6 = stage_final_nms(s5, nms_thr, mode=mode)
    save_json(out_dir / "final_bboxes.json",
              envelope(input_path.name, (W, H), s6))
    draw(image, s6, out_dir / "gpa_final.png", (0, 200, 0))
    save_json(out_dir / "nms_statistics.json",
              {"raw_detections": n_raw,
               "after_basic_filtering": len(s1),
               "after_containment_filtering": len(s2),
               "after_fragment_merging": len(s4),
               "after_duplicate_merging": len(s5),
               "final_detections": len(s6)})

    print(f"Raw detections: {n_raw}")
    print(f"After containment filtering: {len(s2)}")
    print(f"After fragment merging: {len(s4)}")
    print(f"Final detections: {len(s6)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
