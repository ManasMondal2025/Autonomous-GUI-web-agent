#!/usr/bin/env python3
"""GPA-only containment-aware + overlap-aware post-processing.

Order (never aggressive-NMS-first):
    GPA raw -> confidence -> sanity/invalid -> exact duplicates ->
    containment analysis -> REMOVE parents -> overlap clustering ->
    union merge -> final duplicate/containment cleanup -> final boxes.

Rules:
  * Small boxes are NEVER removed for being contained.
  * A large box containing smaller box(es) is a container -> REMOVED
    (children survive), subject to explicit meaningful-child checks.
  * Strongly overlapping pairs merge to their geometric UNION
    (IoU >= MERGE_IOU_THRESHOLD OR min directional overlap >=
    MERGE_OVERLAP_THRESHOLD), guarded by MIN_MERGE_AREA_RATIO so tiny
    slivers can't fuse into giants. Mere adjacency never merges.
  * Duplicates keep higher confidence; merged box confidence = max.

NO OCR. NO VLM. GPA detector itself is NOT changed (official model-card
usage only).

Run:
    python gpa_postprocess.py
    python gpa_postprocess.py --input Ocr/input.png --output Ocr/test_14
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# ALL parameters live here (single variables, never buried in functions)
# --------------------------------------------------------------------------

INPUT_IMAGE = "Ocr/input.png"
OUTPUT_DIR = "Ocr/test_14"

GPA_MODEL = "pretrained/detectors/gpa/model.pt"
# Detector floor (loose; the REAL cut is CONF_THRESHOLD in stage 1).
DETECTOR_CONF = 0.05
# Detector-internal NMS IoU (ultralytics always applies NMS inside
# predict()). Kept loose so stages see the full proposal set.
DETECTOR_INTERNAL_NMS_IOU = 0.70

CONF_THRESHOLD = 0.20

CONTAINMENT_THRESHOLD = 0.90

MERGE_IOU_THRESHOLD = 0.50
MERGE_OVERLAP_THRESHOLD = 0.70

MIN_BOX_WIDTH = 5
MIN_BOX_HEIGHT = 5

# A contained child justifies parent removal when ANY of these hold
# (each is explicit + logged; check 1 subsumes on post-filter data while
# the rest govern if the confidence floor is ever lowered).
MIN_CHILD_AREA_RATIO = 0.01

# Guard: merge only similarly-sized boxes (min_area/max_area >= this),
# so accidental touches between tiny slivers and giants never fuse.
MIN_MERGE_AREA_RATIO = 0.50

LOG_LEVEL = "INFO"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SCRIPT_DIR = Path(__file__).resolve().parent

# stdlib `logging` first (never add repo src/ to sys.path in this script).
log = logging.getLogger("gpa_post")


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


def directional_overlaps(a, b):
    """(IoU, overlap_A, overlap_B) with overlap_X = inter/area(X)."""
    inter = inter_area(a, b)
    union = box_area(a) + box_area(b) - inter
    iou = inter / union if union > 0 else 0.0
    return (iou, inter / max(1e-9, box_area(a)),
            inter / max(1e-9, box_area(b)))


def containment_ratio(inner, outer) -> float:
    """intersection_area / area(inner): is inner contained in outer?"""
    return inter_area(inner, outer) / max(1e-9, box_area(inner))


def union_box(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]),
            max(a[2], b[2]), max(a[3], b[3])]


# --------------------------------------------------------------------------
# Records: one dict per raw detection, carried through all stages
# --------------------------------------------------------------------------

def new_record(raw_id, bbox, confidence, label):
    return {"raw_id": raw_id, "bbox": list(bbox),
            "confidence": float(confidence), "label": label,
            "state": "active", "reason": None, "iou": None,
            "merged_into": None, "source_ids": [raw_id], "merged": False}


def load_image(path: str):
    from PIL import Image

    p = resolve(path)
    if not p.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format {p.suffix!r}.")
    return Image.open(p).convert("RGB"), p


def run_gpa_detector(image, model_path, det_conf, det_iou, device):
    """Official model-card usage. Returns raw [(bbox floats, conf, label)]."""
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
# STAGE 1 -- confidence + sanity/invalid-box filtering
# --------------------------------------------------------------------------

def stage_filter(raw, image_size, conf_thr=CONF_THRESHOLD):
    import math

    W, H = image_size
    recs = []
    for i, (b, s, label) in enumerate(raw, start=1):
        r = new_record(i, b, s, label)
        try:
            x1, y1, x2, y2 = (float(v) for v in b)
        except (TypeError, ValueError):
            r["state"], r["reason"] = "removed", "invalid_bbox"
            recs.append(r)
            continue
        if any(math.isnan(v) or math.isinf(v) for v in (x1, y1, x2, y2)):
            r["state"], r["reason"] = "removed", "invalid_bbox"
            recs.append(r)
            continue
        c = [max(0, x1), max(0, y1), min(float(W), x2),
             min(float(H), y2)]
        if (c[2] - c[0]) < MIN_BOX_WIDTH or \
                (c[3] - c[1]) < MIN_BOX_HEIGHT or c[2] <= c[0] or \
                c[3] <= c[1]:
            r["state"], r["reason"] = "removed", "invalid_bbox"
            recs.append(r)
            continue
        r["bbox"] = [int(round(v)) for v in c]
        if s < conf_thr:
            r["state"], r["reason"] = "removed", "below_confidence"
            recs.append(r)
            continue
        recs.append(r)
    kept = [r for r in recs if r["state"] == "active"]
    log.info("Stage 1: %d -> %d (CONF_THRESHOLD=%.2f).",
             len(raw), len(kept), conf_thr)
    return recs, kept


# --------------------------------------------------------------------------
# Exact duplicates (identical coordinates -> keep higher confidence)
# --------------------------------------------------------------------------

def stage_exact_dup(active):
    groups = {}
    for r in active:
        groups.setdefault(tuple(r["bbox"]), []).append(r)
    removed = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda r: (-r["confidence"], r["raw_id"]))
        win = members[0]
        for m in members[1:]:
            m["state"], m["reason"] = "removed", "exact_duplicate"
            m["merged_into"], m["iou"] = win["raw_id"], 1.0
            win["source_ids"] = sorted(set(win["source_ids"])
                                       | set(m["source_ids"]))
            win["merged"] = True
            removed += 1
    kept = [r for r in active if r["state"] == "active"]
    log.info("Exact duplicates: removed %d.", removed)
    return kept


# --------------------------------------------------------------------------
# Containment analysis + REMOVE parents (small boxes NEVER removed here)
# --------------------------------------------------------------------------

def _parent_checks(big, kids, all_recs):
    """Explicit meaningful-child checks (each logged by the caller).

    Removal fires when ANY holds: (1) a child has reasonable confidence,
    (2) multiple meaningful children, (3) a child occupies a meaningful
    fraction of the container, (4) the container is substantially larger.
    Check 1 subsumes on post-filter data (every child passed the floor);
    the rest are the operative guards if the floor is ever lowered.
    """
    conf_ok = any(all_recs[j]["confidence"] >= CONF_THRESHOLD
                  for j, _ in kids)
    multi = len(kids) >= 2
    fracs = [box_area(all_recs[j]["bbox"]) / max(1e-9, box_area(big["bbox"]))
             for j, _ in kids]
    frac_ok = max(fracs) >= MIN_CHILD_AREA_RATIO
    size_ok = any(box_area(big["bbox"]) / max(
        1e-9, box_area(all_recs[j]["bbox"])) >= 1.0 / MIN_CHILD_AREA_RATIO
        for j, _ in kids)
    return {"conf_ok": conf_ok, "multi": multi, "frac_ok": frac_ok,
            "size_ok": size_ok,
            "fire": conf_ok or multi or frac_ok or size_ok}


def stage_remove_parents(active, contain_thr=CONTAINMENT_THRESHOLD):
    """Drop outer boxes holding >=1 meaningful child. Simultaneous marking
    (chains collapse correctly). Inner detections always survive."""
    children_of = {}
    for i, outer in enumerate(active):
        kids = [(j, containment_ratio(inner["bbox"], outer["bbox"]))
                for j, inner in enumerate(active) if j != i
                and containment_ratio(inner["bbox"], outer["bbox"])
                >= contain_thr]
        if kids:
            children_of[i] = kids
    drop = set()
    for i, kids in children_of.items():
        checks = _parent_checks(active[i], kids, active)
        if checks["fire"]:
            drop.add(i)
            print(f"REMOVED CONTAINER:\n"
                  f"  box_id={active[i]['raw_id']}\n"
                  f"  confidence={active[i]['confidence']:.2f}\n"
                  f"  reason=contains {len(kids)} boxes\n"
                  f"  contained_ids={[active[j]['raw_id'] for j, _ in kids]}\n"
                  f"  containment={round(max(r for _, r in kids), 3)}\n"
                  f"  area_ratio={round(max(box_area(active[j]['bbox']) / max(1e-9, box_area(active[i]['bbox'])) for j, _ in kids), 4)}\n"
                  f"  checks={checks}")
            log.info("Parent-filter removing box id=%d (%d children).",
                     active[i]["raw_id"], len(kids))
    for i in drop:
        active[i]["state"] = "removed"
        active[i]["reason"] = "parent_container"
        active[i]["duplicate_of"] = None
    kept = [r for r in active if r["state"] == "active"]
    log.info("Parent removal: %d -> %d.", len(active), len(kept))
    return kept


# --------------------------------------------------------------------------
# Overlap clustering + union merge (union-find; transitive chains collapse
# in ONE pass). Mere adjacency (IoU ~ 0, touching edges) NEVER merges.
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


def _mergeable(a, b, iou_thr=MERGE_IOU_THRESHOLD,
               ov_thr=MERGE_OVERLAP_THRESHOLD,
               area_ratio=MIN_MERGE_AREA_RATIO):
    iou, oa, ob = directional_overlaps(a["bbox"], b["bbox"])
    if area_ratio is not None:
        ar = min(box_area(a["bbox"]), box_area(b["bbox"])) / max(
            1e-9, max(box_area(a["bbox"]), box_area(b["bbox"])))
        if ar < area_ratio:
            return False, iou, oa, ob
    if iou >= iou_thr or min(oa, ob) >= ov_thr:
        return True, iou, oa, ob
    return False, iou, oa, ob


def stage_merge(active):
    dsu = _DSU(len(active))
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            ok, _, _, _ = _mergeable(active[i], active[j])
            if ok:
                dsu.union(i, j)
    comps = {}
    for i in range(len(active)):
        comps.setdefault(dsu.find(i), []).append(i)
    out = []
    for members in comps.values():
        if len(members) == 1:
            out.append(active[members[0]])
            continue
        ub = active[members[0]]["bbox"]
        for m in members[1:]:
            ub = union_box(ub, active[m]["bbox"])
        ub = [int(round(v)) for v in ub]
        best = max(members, key=lambda m: (active[m]["confidence"],
                                           -active[m]["raw_id"]))
        sids = []
        for m in members:
            sids += [s for s in active[m]["source_ids"] if s not in sids]
        win = active[best]
        iou0, oa0, ob0 = directional_overlaps(active[members[0]]["bbox"],
                                              active[members[1]]["bbox"])
        print(f"MERGED:\n"
              f"  box_ids={sorted(active[m]['raw_id'] for m in members)}\n"
              f"  IoU={round(iou0, 3)}\n"
              f"  overlap_A={round(oa0, 3)}\n"
              f"  overlap_B={round(ob0, 3)}\n"
              f"  merged_bbox={ub}")
        log.info("Merged %d boxes -> %r.", len(members), ub)
        for m in members:
            if m != best:
                active[m]["state"] = "removed"
                active[m]["reason"] = "merged"
                active[m]["merged_into"] = win["raw_id"]
                _, oa, ob = directional_overlaps(active[m]["bbox"],
                                                 win["bbox"])
                active[m]["iou"] = round(
                    box_iou(active[m]["bbox"], win["bbox"]), 3)
        win["bbox"] = ub
        win["confidence"] = max(active[m]["confidence"] for m in members)
        win["source_ids"] = sorted(sids)
        win["merged"] = True
        out.append(win)
    log.info("Merge: %d -> %d.", len(active), len(out))
    return out


# --------------------------------------------------------------------------
# Final duplicate/containment cleanup (exact + parent re-check on unions)
# --------------------------------------------------------------------------

def stage_final_cleanup(active):
    n0 = len(active)
    active = stage_exact_dup(active)
    active = stage_remove_parents(active)
    log.info("Final cleanup: %d -> %d.", n0, len(active))
    return active


# --------------------------------------------------------------------------
# Output: saves, debug visualization, stats, console, main
# --------------------------------------------------------------------------

def save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=4))
    log.info("Saved %s.", path)


def draw_labeled(image, elements, path: Path, color) -> None:
    from PIL import ImageDraw

    vis = image.copy()
    d = ImageDraw.Draw(vis)
    W, H = image.size
    for e in elements:
        x1, y1, x2, y2 = e["bbox"]
        d.rectangle([x1, y1, x2, y2], outline=color, width=2)
        tag = f"[{e['id']}] {e['confidence']:.2f}"
        tx, ty = min(x2 + 3, W - 1), max(0, y1 - 12)
        d.rectangle([tx, ty, min(W, tx + 8 + 7 * len(tag)), ty + 13],
                    fill=color)
        d.text((tx + 4, ty + 1), tag, fill=(255, 255, 255))
    vis.save(path)
    log.info("Saved visualization: %s", path)


def draw_debug(image, all_recs, final_ids, path: Path) -> None:
    """4 colors: raw context gray, removed red, merged-winner blue,
    untouched final green."""
    from PIL import ImageDraw

    vis = image.copy()
    d = ImageDraw.Draw(vis)
    finals = {r["raw_id"] for r in final_ids}
    for r in all_recs:
        x1, y1, x2, y2 = r["bbox"]
        d.rectangle([x1, y1, x2, y2], outline=(150, 150, 150), width=1)
    for r in all_recs:
        if r["state"] != "removed":
            continue
        x1, y1, x2, y2 = r["bbox"]
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
    for r in final_ids:
        x1, y1, x2, y2 = r["bbox"]
        color = (0, 0, 255) if r.get("merged") else (0, 200, 0)
        d.rectangle([x1, y1, x2, y2], outline=color, width=2)
        tag = f"[{r['raw_id']}]"
        tx, ty = min(x2 + 3, image.size[0] - 1), max(0, y1 - 12)
        d.rectangle([tx, ty, min(image.size[0], tx + 8 + 7 * len(tag)),
                     ty + 13], fill=color)
        d.text((tx + 4, ty + 1), tag, fill=(255, 255, 255))
    vis.save(path)
    log.info("Saved debug visualization: %s", path)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="GPA containment+overlap post-processing (no OCR).")
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--gpa-model", default=None)
    ap.add_argument("--conf-threshold", type=float, default=None)
    ap.add_argument("--contain", type=float, default=None)
    ap.add_argument("--merge-iou", type=float, default=None)
    ap.add_argument("--merge-overlap", type=float, default=None)
    ap.add_argument("--min-child-area", type=float, default=None)
    ap.add_argument("--min-merge-area", type=float, default=None)
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
    global MIN_CHILD_AREA_RATIO, MIN_MERGE_AREA_RATIO
    global MERGE_IOU_THRESHOLD, MERGE_OVERLAP_THRESHOLD
    global CONTAINMENT_THRESHOLD, CONF_THRESHOLD
    if args.min_child_area is not None:
        MIN_CHILD_AREA_RATIO = args.min_child_area
    if args.min_merge_area is not None:
        MIN_MERGE_AREA_RATIO = args.min_merge_area
    if args.merge_iou is not None:
        MERGE_IOU_THRESHOLD = args.merge_iou
    if args.merge_overlap is not None:
        MERGE_OVERLAP_THRESHOLD = args.merge_overlap
    if args.contain is not None:
        CONTAINMENT_THRESHOLD = args.contain
    conf_thr = CONF_THRESHOLD if args.conf_threshold is None \
        else args.conf_threshold
    CONF_THRESHOLD = conf_thr

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

    det_conf = min(DETECTOR_CONF, conf_thr)
    raw = run_gpa_detector(image, args.gpa_model or GPA_MODEL, det_conf,
                           DETECTOR_INTERNAL_NMS_IOU, args.device)
    n_raw = len(raw)
    save_json(out_dir / "raw_detections.json",
              [{"id": i + 1, "bbox": b,
                "confidence": round(s, 4), "label": lab}
               for i, (b, s, lab) in enumerate(raw)])
    draw_labeled(image,
                 [{"id": i + 1, "bbox": b, "confidence": s}
                  for i, (b, s, _) in enumerate(raw)],
                 out_dir / "raw_gpa.png", (255, 0, 0))

    recs, s1 = stage_filter(raw, (W, H), conf_thr)
    n_conf = len(s1)
    s2 = stage_exact_dup(s1)
    s3 = stage_remove_parents(s2)
    s4 = stage_merge(s3)
    s5 = stage_final_cleanup(s4)

    s5.sort(key=lambda r: (-r["confidence"], r["raw_id"]))
    final = [{"id": j, "bbox": [int(v) for v in r["bbox"]],
              "confidence": round(r["confidence"], 4),
              "label": r["label"]} for j, r in enumerate(s5, start=1)]
    for e, r in zip(final, sorted(s5, key=lambda r: (-r["confidence"],
                                                    r["raw_id"]))):
        e["source_ids"] = sorted(r["source_ids"])
        e["merged"] = r["merged"]
    save_json(out_dir / "final_detections.json",
              {"image": input_path.name, "image_width": W,
               "image_height": H, "detections": final})
    draw_labeled(image, final, out_dir / "final_boxes.png", (0, 200, 0))
    draw_debug(image, [{"raw_id": r["raw_id"], "bbox": r["bbox"],
                        "state": r["state"], "merged": r["merged"]}
                       for r in recs],
                [{"raw_id": r["raw_id"], "bbox": r["bbox"],
                  "merged": r["merged"]} for r in s5],
                out_dir / "postprocess_debug.png")

    removed = []
    for r in recs:
        if r["state"] == "removed":
            entry = {"id": r["raw_id"], "bbox": r["bbox"],
                     "confidence": round(r["confidence"], 4),
                     "removed": True, "reason": r["reason"]}
            if r["iou"] is not None:
                entry["iou"] = r["iou"]
            if r["merged_into"] is not None:
                entry["duplicate_of"] = r["merged_into"]
            removed.append(entry)
    save_json(out_dir / "nms_statistics.json",
              {"raw_detections": n_raw,
               "after_confidence_filter": n_conf,
               "after_exact_duplicate_removal": len(s2),
               "after_containment_filter": len(s3),
               "after_overlap_merge": len(s4),
               "final_detections": len(s5),
               "removed": removed})

    print("=" * 50)
    print("GPA GUI DETECTION POST-PROCESSING")
    print("=" * 50)
    print()
    print(f"Raw detections:{n_raw:>22}")
    print(f"After confidence filtering:{n_conf:>17}")
    print(f"After duplicate removal:{len(s2):>21}")
    print(f"After sibling suppression:{len(s4):>19}")
    print(f"Final detections:{len(s5):>25}")
    print()
    print("=" * 50)
    print("FINAL DETECTIONS")
    print("=" * 50)
    print()
    print(f"{'ID':<6}{'Confidence':<14}BBox")
    print("-" * 50)
    for e in final:
        x1, y1, x2, y2 = e["bbox"]
        print(f"{e['id']:<6}{e['confidence']:<14.2f}"
              f"[{x1}, {y1}, {x2}, {y2}]")
    print()
    print("Saved:")
    print("raw_detections.json")
    print("final_detections.json")
    print("raw_gpa.png")
    print("postprocess_debug.png")
    print("final_boxes.png")
    print("nms_statistics.json")
    print()
    print("KEPT:")
    for e in final:
        x1, y1, x2, y2 = e["bbox"]
        reason = ("merged_from %s" % e["source_ids"]
                  if e["merged"] else "preserved")
        print(f"  box_id={e['id']} bbox=[{x1}, {y1}, {x2}, {y2}] "
              f"confidence={e['confidence']:.2f} reason={reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
