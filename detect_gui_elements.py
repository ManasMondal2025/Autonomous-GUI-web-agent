#!/usr/bin/env python3
"""Screenshot-only GUI element detection: GPA + OmniParser V2, union +
intelligent filtering. NO OCR. NO VLM.

Pipeline:
    input.png -> GPA + icon_detect_v3 -> UNION -> normalize -> duplicates
    (IoU>0.85) -> containment (keep small) -> parent filtering ->
    final boxes + visualization + JSON (+ optional crops for inspection).

Run:
    python detect_gui_elements.py
    python detect_gui_elements.py --input Ocr/input.png --output Ocr

Real detector APIs (nothing invented):
  * GPA GUI Detector (Salesforce/GPA-GUI-Detector): ultralytics YOLO,
    `YOLO(model.pt).predict(img, conf, iou)` per its model card.
  * OmniParser V2 icon_detect_v3: YOLOv9-E TorchScript run through the
    OFFICIAL `YOLOv9Detector` from microsoft/OmniParser `util/yolov9.py`
    (inlined below with attribution; only the HF auto-download default
    is removed -- an explicit local weight path is required).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration (top of file)
# --------------------------------------------------------------------------

INPUT_IMAGE = "Ocr/input.png"
OUTPUT_DIR = "Ocr"

GPA_MODEL = "pretrained/detectors/gpa/model.pt"
OMNI_MODEL = "pretrained/detectors/omniparser_v3/icon_detect_v3/model.pt"
GPA_CONF = 0.05
OMNI_CONF = 0.05
NMS_IOU = 0.7

IOU_DUPLICATE_THRESHOLD = 0.85
CONTAINMENT_THRESHOLD = 0.90
PADDING_RATIO = 0.05
MIN_BOX_AREA = 20

# Parent-box filtering: drop a box containing >= this many smaller boxes
# (containment > CONTAINMENT_THRESHOLD each). Single-child nested pairs
# (one box nearly containing exactly one other + substantial overlap) keep
# the TIGHTER box when PARENT_SINGLE_CHILD_TIGHTER_WINS is True.
PARENT_MIN_CHILDREN = 2
PARENT_SINGLE_CHILD_TIGHTER_WINS = True
PARENT_SINGLE_CHILD_MIN_IOU = 0.25

LOG_LEVEL = "INFO"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SCRIPT_DIR = Path(__file__).resolve().parent

# stdlib `logging` first: this repo ships src/logging/, which would shadow
# it if src/ ever joins sys.path (it never does in this script).
log = logging.getLogger("detect_gui")


def setup_logging(level: str = LOG_LEVEL) -> None:
    logging.basicConfig(level=getattr(logging, str(level).upper(),
                                      logging.INFO),
                        format="%(levelname)s: %(message)s")


def resolve(p: str) -> Path:
    """Resolve a configured path: as-given (cwd) first, then vs script dir."""
    q = Path(p)
    if q.exists() or q.is_absolute():
        return q
    return SCRIPT_DIR / p


# --------------------------------------------------------------------------
# Geometry helpers (pure, deterministic)
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


def containment_ratio(small, large) -> float:
    """intersection(small, large) / area(small): is small inside large?"""
    return inter_area(small, large) / max(1e-9, box_area(small))


# --------------------------------------------------------------------------
# Input + coordinate normalization (Stage 3)
# --------------------------------------------------------------------------

def load_image(path: str):
    from PIL import Image

    p = resolve(path)
    if not p.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format {p.suffix!r}. "
                         f"Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    img = Image.open(p).convert("RGB")  # original resolution, never resized
    return img, p


def normalize_box(raw_xyxy, image_size):
    """Clip to the ORIGINAL image; reject invalid/zero-area/tiny boxes.

    Returns (ok, int_box_or_None, reason). Both detectors already emit
    original-scale pixels (ultralytics maps back internally; the official
    v3 code clamps to image size) -- this is a defensive second pass, and
    the coordinate system is never silently changed.
    """
    import math

    W, H = image_size
    try:
        x1, y1, x2, y2 = (float(v) for v in raw_xyxy)
    except (TypeError, ValueError):
        return False, None, "non-numeric coordinates"
    if any(math.isnan(v) or math.isinf(v) for v in (x1, y1, x2, y2)):
        return False, None, "NaN/inf coordinates"
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


# --------------------------------------------------------------------------
# Detectors (Stage 1 + Stage 2). Real APIs only.
# --------------------------------------------------------------------------

# --- Official microsoft/OmniParser util/yolov9.py (verbatim inference logic;
# attribution: https://github.com/microsoft/OmniParser). Only change: the
# HF auto-download default is removed; an explicit model_path is required.
class _V3Boxes:
    def __init__(self, xyxy, confidence):
        self.xyxy = xyxy
        self.conf = confidence


class _V3Result:
    def __init__(self, boxes):
        self.boxes = boxes


class YOLOv9Detector:
    """TorchScript YOLOv9-E detector with an Ultralytics-compatible predict
    API. (Official OmniParser code, see module docstring.)"""

    strides = (8, 16, 32)

    def __init__(self, model_path, device=None):
        from contextlib import nullcontext  # noqa: F401  (kept for parity)
        import torch

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise RuntimeError(
                f"[OmniParser/v3] missing weight file: {self.model_path}")
        import torch as _t
        self.model = _t.jit.load(str(self.model_path),
                                 map_location=self.device).eval()

    @staticmethod
    def _normalize_image_size(image_size):
        if isinstance(image_size, int):
            width = height = image_size
        elif len(image_size) == 2:
            height, width = image_size
        else:
            raise ValueError(
                f"Expected one or two image dimensions, got {image_size}")
        width = ((int(width) + 31) // 32) * 32
        height = ((int(height) + 31) // 32) * 32
        return width, height

    @staticmethod
    def _load_image(source):
        from PIL import Image
        import numpy as np

        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, np.ndarray):
            return Image.fromarray(source).convert("RGB")
        with Image.open(source) as image:
            return image.convert("RGB")

    def _preprocess(self, image, image_size):
        import numpy as np
        import torch
        from PIL import Image

        target_width, target_height = self._normalize_image_size(image_size)
        image_width, image_height = image.size
        scale = min(target_width / image_width,
                    target_height / image_height)
        resized_width = int(image_width * scale)
        resized_height = int(image_height * scale)
        pad_left = (target_width - resized_width) // 2
        pad_top = (target_height - resized_height) // 2

        resized = image.resize((resized_width, resized_height),
                               Image.LANCZOS)
        padded = Image.new("RGB", (target_width, target_height),
                           (114, 114, 114))
        padded.paste(resized, (pad_left, pad_top))
        image_array = np.asarray(padded, dtype=np.float32
                                 ).transpose(2, 0, 1) / 255.0
        image_tensor = torch.from_numpy(image_array
                                        ).unsqueeze(0).to(self.device)
        return image_tensor, scale, pad_left, pad_top

    def _decode(self, outputs):
        import torch

        class_logits = []
        decoded_boxes = []
        for output_index, stride in zip(range(0, len(outputs), 2),
                                        self.strides):
            layer_logits, layer_distances = outputs[output_index:
                                                    output_index + 2]
            batch_size, _, height, width = layer_logits.shape
            layer_logits = layer_logits.permute(0, 2, 3, 1).reshape(
                batch_size, -1, layer_logits.shape[1])
            layer_distances = layer_distances.permute(0, 2, 3, 1).reshape(
                batch_size, -1, 4) * stride

            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, device=self.device),
                torch.arange(width, device=self.device),
                indexing="ij",
            )
            anchors = (torch.stack((grid_x, grid_y), dim=-1
                                   ).reshape(-1, 2) + 0.5) * stride
            left_top, right_bottom = layer_distances.chunk(2, dim=-1)
            decoded_boxes.append(torch.cat((anchors - left_top,
                                            anchors + right_bottom), dim=-1))
            class_logits.append(layer_logits)

        return (torch.cat(class_logits, dim=1).sigmoid(),
                torch.cat(decoded_boxes, dim=1))

    @__import__("torch").inference_mode()
    def predict(self, source, conf=0.25, imgsz=640, iou=0.7, max_det=300):
        import torch
        from torchvision.ops import batched_nms

        image = self._load_image(source)
        image_tensor, scale, pad_left, pad_top = self._preprocess(
            image, imgsz)

        if self.device.type == "cuda":
            ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            from contextlib import nullcontext
            ctx = nullcontext()
        with ctx, torch.jit.optimized_execution(False):
            class_scores, boxes = self._decode(self.model(image_tensor))

        scores, class_ids = class_scores[0].max(dim=-1)
        valid = scores > conf
        scores = scores[valid]
        class_ids = class_ids[valid]
        boxes = boxes[0][valid]
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale

        keep = batched_nms(boxes, scores, class_ids, iou)[:max_det]
        boxes = boxes[keep]
        scores = scores[keep]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, image.width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, image.height)
        return [_V3Result(_V3Boxes(boxes, scores))]


def run_gpa(image, image_size, model_path, conf, nms_iou, device):
    """Stage 1: GPA GUI Detector (official per-model-card usage)."""
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

    def label_of(box, cls_id=None):
        if cls_id is not None and cls_id in names:
            return str(names[cls_id])
        return "element"

    log.info("Running GPA (conf=%.2f, iou=%.2f, device=%s)...",
             conf, nms_iou, device)
    res = model.predict(image, conf=conf, iou=nms_iou, verbose=False,
                        device=device)[0]
    recs = []
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        sc = res.boxes.conf.cpu().numpy()
        try:
            cl = res.boxes.cls.cpu().numpy().astype(int)
        except Exception:
            cl = [None] * len(xyxy)
        for b, s, c in zip(xyxy, sc, cl):
            if (b[2] - b[0]) <= 1 or (b[3] - b[1]) <= 1:
                continue
            recs.append((b, s, c))
    out = []
    for b, s, c in recs:
        ok, clean, reason = normalize_box([float(v) for v in b], image_size)
        if not ok:
            log.warning("[gpa] rejecting box %r: %s",
                        [round(float(v), 1) for v in b], reason)
            continue
        out.append({"id": 0, "bbox": clean, "confidence": float(s),
                    "label": label_of(b, int(c) if c is not None else None),
                    "source": "gpa",
                    "area": (clean[2] - clean[0]) * (clean[3] - clean[1]),
                    "width": clean[2] - clean[0],
                    "height": clean[3] - clean[1]})
    log.info("GPA raw detections: %d (kept %d after normalization).",
             len(recs), len(out))
    return out


def run_omniparser_v3(image, image_size, model_path, conf, nms_iou, device):
    """Stage 2: OmniParser V2 icon_detect_v3 (official YOLOv9Detector)."""
    log.info("Loading OmniParser V2 icon_detect_v3 from %s ...",
             resolve(model_path))
    det = YOLOv9Detector(model_path=str(resolve(model_path)),
                         device=device)
    log.info("Running icon_detect_v3 (conf=%.2f, iou=%.2f, device=%s)...",
             conf, nms_iou, device)
    res = det.predict(image, conf=conf, imgsz=640, iou=nms_iou)[0]
    raw = []
    if res.boxes is not None and len(res.boxes.xyxy) > 0:
        for b, s in zip(res.boxes.xyxy.cpu().numpy(),
                        res.boxes.conf.cpu().numpy()):
            if (b[2] - b[0]) <= 1 or (b[3] - b[1]) <= 1:
                continue
            raw.append((b, s))
    out = []
    for b, s in raw:
        ok, clean, reason = normalize_box([float(v) for v in b], image_size)
        if not ok:
            log.warning("[omniparser] rejecting box %r: %s",
                        [round(float(v), 1) for v in b], reason)
            continue
        out.append({"id": 0, "bbox": clean, "confidence": float(s),
                    "label": "icon", "source": "omniparser",
                    "area": (clean[2] - clean[0]) * (clean[3] - clean[1]),
                    "width": clean[2] - clean[0],
                    "height": clean[3] - clean[1]})
    log.info("OmniParser raw detections: %d (kept %d after normalization).",
             len(raw), len(out))
    return out


# --------------------------------------------------------------------------
# Filter stages (Stages 6-9)
# --------------------------------------------------------------------------

def _merge_pair(keep, other):
    """Merge `other` into `keep`: better box wins, sources preserved."""
    a, b = keep["confidence"] or 0.0, other["confidence"] or 0.0
    if b > a + 1e-9 or (abs(b - a) <= 1e-9
                        and box_area(other["bbox"]) < box_area(keep["bbox"])):
        keep["bbox"] = other["bbox"]
        keep["confidence"] = other["confidence"]
        keep["label"] = other.get("label") or keep.get("label")
        keep["area"] = box_area(other["bbox"])
        keep["width"] = other["bbox"][2] - other["bbox"][0]
        keep["height"] = other["bbox"][3] - other["bbox"][1]
    for s in ([other["source"]] if isinstance(other["source"], str)
              else list(other["source"])):
        if s not in keep["source"]:
            keep["source"].append(s)
    keep.setdefault("merged_from", []).append(
        {"source": other["source"], "bbox": other["bbox"],
         "confidence": other["confidence"]})


def duplicate_filter(proposals, iou_thr=IOU_DUPLICATE_THRESHOLD):
    """Stage 6: IoU > thr -> duplicates. Higher confidence wins (tie:
    smaller/more precise wins). Sources merged, never dropped silently."""
    for p in proposals:
        p["source"] = ([p["source"]] if isinstance(p["source"], str)
                       else list(p["source"]))
        p.setdefault("merged_from", [])
    order = sorted(range(len(proposals)),
                   key=lambda k: (-(proposals[k]["confidence"] or 0.0),
                                  box_area(proposals[k]["bbox"]), k))
    kept = []
    for k in order:
        cand = proposals[k]
        dup = None
        for q in kept:
            if box_iou(cand["bbox"], q["bbox"]) > iou_thr:
                dup = q
                break
        if dup is None:
            kept.append(cand)
        else:
            _merge_pair(dup, cand)
    return kept


def containment_parent_filter(boxes,
                              contain_thr=CONTAINMENT_THRESHOLD,
                              min_children=PARENT_MIN_CHILDREN,
                              single_tighter_wins=
                              PARENT_SINGLE_CHILD_TIGHTER_WINS,
                              single_min_iou=PARENT_SINGLE_CHILD_MIN_IOU):
    """Stages 7-9: containment analysis + parent filtering + small-box
    preservation.

    * Small boxes are NEVER removed for being inside a big box.
    * A big box containing >= min_children smaller boxes (containment >
      thr each) is a structural parent -> removed (sources of children
      already preserved on the children themselves).
    * A big box containing exactly ONE smaller box with substantial
      overlap (IoU >= single_min_iou) is a loose duplicate -> keep the
      TIGHTER (smaller) box, merging sources (only when the flag is on).
    Returns (kept, n_parents_removed, n_tightened).
    """
    drop_parent, tighten = set(), {}
    for i, big in enumerate(boxes):
        kids = [j for j, sm in enumerate(boxes) if j != i and
                containment_ratio(sm["bbox"], big["bbox"]) > contain_thr]
        if len(kids) >= min_children:
            drop_parent.add(i)
        elif single_tighter_wins and len(kids) == 1:
            j = kids[0]
            if box_iou(big["bbox"], boxes[j]["bbox"]) >= single_min_iou:
                drop_parent.add(i)
                tighten[j] = tighten.get(j, []) + [i]
    kept = []
    for i, b in enumerate(boxes):
        if i in drop_parent:
            log.info("Parent-filter removing box %r (conf=%.2f, src=%s).",
                     b["bbox"], b["confidence"] or 0.0, b["source"])
            continue
        for pi in tighten.get(i, []):
            # Tighter-wins: child geometry/label stay; confidence takes
            # the max; parent source is preserved, never dropped silently.
            parent = boxes[pi]
            b["confidence"] = max(b["confidence"] or 0.0,
                                  parent["confidence"] or 0.0)
            for s in parent["source"]:
                if s not in b["source"]:
                    b["source"].append(s)
            b.setdefault("merged_from", []).append(
                {"source": parent["source"], "bbox": parent["bbox"],
                 "confidence": parent["confidence"]})
        kept.append(b)
    return kept, len(drop_parent), sum(len(v) for v in tighten.values())


# --------------------------------------------------------------------------
# Output: JSON, visualization, crops, console
# --------------------------------------------------------------------------

def save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=4))
    log.info("Saved %s (%d entries).", path,
             len(payload) if isinstance(payload, list) else 1)


def draw_boxes(image, elements, path: Path, with_source: bool = False
               ) -> None:
    """Draw red boxes + [id] (+ optional source) on an ORIGINAL-size copy."""
    from PIL import ImageDraw

    vis = image.copy()
    d = ImageDraw.Draw(vis)
    W, H = image.size
    for e in elements:
        x1, y1, x2, y2 = e["bbox"]
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        tag = f"[{e['id']}]"
        if with_source:
            src = e.get("source")
            s = "+".join(src) if isinstance(src, list) else str(src)
            tag += f" {s}"
        tx, ty = min(x2 + 3, W - 1), max(0, y1 - 12)
        d.rectangle([tx, ty, min(W, tx + 8 + 7 * len(tag)), ty + 13],
                    fill=(255, 0, 0))
        d.text((tx + 4, ty + 1), tag, fill=(255, 255, 255))
    vis.save(path)
    log.info("Saved visualization: %s", path)


def save_crops(image, elements, crops_dir: Path, padding: float,
               input_path: Path) -> None:
    """Debug crops from the ORIGINAL screenshot (no OCR)."""
    crops_dir.mkdir(parents=True, exist_ok=True)
    W, H = image.size
    for e in elements:
        x1, y1, x2, y2 = e["bbox"]
        px, py = int((x2 - x1) * padding), int((y2 - y1) * padding)
        box = [max(0, x1 - px), max(0, y1 - py),
               min(W, x2 + px), min(H, y2 + py)]
        cp = crops_dir / f"element_{e['id']:04d}.png"
        if cp.resolve() == input_path.resolve():
            raise RuntimeError(f"Refusing to overwrite input: {cp}")
        image.crop(tuple(box)).save(cp)
    log.info("Saved %d debug crops to %s.", len(elements), crops_dir)


def print_report(input_name: str, image_size, n_gpa: int, n_omni: int,
                 n_prop: int, n_dup: int, n_cont: int, final) -> None:
    W, H = image_size
    print("=" * 40)
    print("GUI BBOX DETECTION")
    print("=" * 40)
    print()
    print("Image:")
    print(input_name)
    print()
    print("Image size:")
    print(f"{W} x {H}")
    print()
    print("GPA detections:")
    print(n_gpa)
    print()
    print("OmniParser detections:")
    print(n_omni)
    print()
    print("Total proposals:")
    print(n_prop)
    print()
    print("After duplicate filtering:")
    print(n_dup)
    print()
    print("After containment filtering:")
    print(n_cont)
    print()
    print("Final bounding boxes:")
    print(len(final))
    print()
    print("=" * 40)
    print("FINAL ELEMENTS")
    print("=" * 40)
    print()
    print(f"{'ID':<6}{'SOURCE':<18}{'BBOX':<30}LABEL")
    print("-" * 64)
    for e in final:
        x1, y1, x2, y2 = e["bbox"]
        src = e.get("source")
        s = "+".join(src) if isinstance(src, list) else str(src)
        print(f"{e['id']:<6}{s:<18}[{x1},{y1},{x2},{y2}]"
              f"{'':<{30 - len(f'[{x1},{y1},{x2},{y2}]')}}"
              f"{e.get('label', '')}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="GPA + OmniParser V2 union bbox detection (no OCR).")
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--gpa-model", default=None)
    ap.add_argument("--omni-model", default=None)
    ap.add_argument("--gpa-conf", type=float, default=None)
    ap.add_argument("--omni-conf", type=float, default=None)
    ap.add_argument("--iou-dup", type=float, default=None)
    ap.add_argument("--contain", type=float, default=None)
    ap.add_argument("--min-children", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-crops", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    setup_logging()
    args = parse_args(argv)

    global MIN_BOX_AREA
    input_arg = args.input or INPUT_IMAGE
    out_dir = Path(args.output or OUTPUT_DIR)
    if not out_dir.is_absolute():
        cand, alt = Path.cwd() / out_dir, SCRIPT_DIR / out_dir
        out_dir = cand if cand == alt else (
            alt if not cand.exists() and alt.exists() else cand)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        image, input_path = load_image(input_arg)
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 2
    W, H = image.size
    log.info("Image: %s (%d x %d)", input_path, W, H)

    gpa_conf = GPA_CONF if args.gpa_conf is None else args.gpa_conf
    omni_conf = OMNI_CONF if args.omni_conf is None else args.omni_conf
    iou_dup = IOU_DUPLICATE_THRESHOLD if args.iou_dup is None \
        else args.iou_dup
    contain = CONTAINMENT_THRESHOLD if args.contain is None else args.contain
    min_children = PARENT_MIN_CHILDREN if args.min_children is None \
        else args.min_children

    # -- Stage 1 + Stage 2 (warn-and-continue; both fail -> terminate) --
    gpa, omni = None, None
    try:
        gpa = run_gpa(image, (W, H), args.gpa_model or GPA_MODEL,
                      gpa_conf, NMS_IOU, args.device)
    except Exception as e:
        print("WARNING: GPA detection failed")
        log.warning("GPA detection failed: %s", e)
    try:
        omni = run_omniparser_v3(image, (W, H), args.omni_model or OMNI_MODEL,
                                 omni_conf, NMS_IOU, args.device)
    except Exception as e:
        print("WARNING: OmniParser detection failed")
        log.warning("OmniParser detection failed: %s", e)
    if gpa is None and omni is None:
        log.error("Both detectors failed; terminating.")
        return 2
    gpa, omni = gpa or [], omni or []

    for j, e in enumerate(gpa, start=1):
        e["id"] = j
    for j, e in enumerate(omni, start=1):
        e["id"] = j
    save_json(out_dir / "gpa_detections.json", gpa)
    save_json(out_dir / "omniparser_detections.json", omni)
    draw_boxes(image, gpa, out_dir / "gpa_visualization.png")
    draw_boxes(image, omni, out_dir / "omniparser_visualization.png")

    # -- Stage 5: union (high recall; nothing discarded here) --
    proposals = [dict(e) for e in gpa] + [dict(e) for e in omni]
    save_json(out_dir / "all_proposals.json", proposals)
    log.info("Union proposals: %d.", len(proposals))

    # -- Stage 6: duplicates --
    deduped = duplicate_filter(proposals, iou_dup)
    log.info("After duplicate filtering (IoU>%.2f): %d.", iou_dup,
             len(deduped))

    # -- Stages 7-9: containment + parents (small boxes always survive) --
    final, n_par, n_tight = containment_parent_filter(
        deduped, contain, min_children)
    log.info("Parent filtering removed %d boxes (%d single-child "
             "tightened).", n_par, n_tight)

    # -- Stage 11: final representation (confidence order, renumbered) --
    final.sort(key=lambda e: (-(e["confidence"] or 0.0),
                              e["bbox"][1], e["bbox"][0]))
    for j, e in enumerate(final, start=1):
        e["id"] = j
        e["original_sources"] = list(e["source"])
        e.setdefault("merged_from", [])
        e["filter_reason"] = None
    save_json(out_dir / "final_bboxes.json", final)
    draw_boxes(image, final, out_dir / "final_bbox_visualization.png",
               with_source=True)
    if not args.no_crops:
        save_crops(image, final, out_dir / "final_crops", PADDING_RATIO,
                   input_path)

    # -- Stage 14: console --
    print_report(input_path.name, (W, H), len(gpa), len(omni),
                 len(proposals), len(deduped), len(final), final)
    return 0


if __name__ == "__main__":
    sys.exit(main())

