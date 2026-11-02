#!/usr/bin/env python3
"""Tiny-text detector experiment: CRAFT vs DBNet++ inside OmniParser boxes.

Standalone diagnostic — modifies nothing in the agent pipeline. Uses the
ORIGINAL screenshot throughout; crops are taken at original resolution and
all predictions are mapped back to original screenshot coordinates.

- CRAFT lineage: EasyOCR's maintained CRAFT detector (craft-text-detector is
  incompatible with modern torchvision; EasyOCR Reader(detect_network='craft')
  is the same architecture). Exp A constrains it to original geometry
  (canvas_size = crop long side, mag_ratio=1.0); Exp B uses its defaults
  (canvas_size=2560), i.e. its normal internal preprocessing.
- DBNet++: paddlex PP-OCRv5_mobile_det, detection-only (no recognition).
  Its internal preprocessing is fixed; the same call serves both experiments
  and this is reported as such.

Usage:
  python3 scripts/evaluate_tiny_text_detectors.py \
      --screenspot-index 8 --out-dir reports/tiny_text_detection
  python3 scripts/evaluate_tiny_text_detectors.py \
      --image test_pages/fixtures/product.png --gt <gt.json> \
      --out-dir reports/tiny_text_detection_product
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.ERROR)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from PIL import Image, ImageDraw
import numpy as np

IOU_TP = 0.5
MIN_DIM_FLAG = 16  # below this, crops are flagged below-detector-resolution


def to_xyxy(bbox):
    if isinstance(bbox, dict):
        x, y = float(bbox["x"]), float(bbox["y"])
        return [x, y, x + float(bbox["width"]), y + float(bbox["height"])]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return [x1, y1, x2, y2]


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def pad_box(b, frac, img_w, img_h):
    w, h = b[2] - b[0], b[3] - b[1]
    px, py = w * frac, h * frac
    return [max(0.0, b[0] - px), max(0.0, b[1] - py),
            min(float(img_w), b[2] + px), min(float(img_h), b[3] + py)]


def load_gt(args):
    if args.screenspot_index is not None:
        samples = [json.loads(l) for l in
                   open(REPO / "data/screenspot/v2_mini.jsonl")]
        s = samples[args.screenspot_index]
        img_path = Path(s["img_path"])
        x1, y1, x2, y2 = s["bbox"]
        gt = [{"text": s["instruction"], "bbox": [x1, y1, x2, y2]}]
        return img_path, gt, f"screenspot_{args.screenspot_index}"
    img_path = Path(args.image)
    gt = []
    for i, g in enumerate(json.load(open(args.gt))):
        gt.append({"text": g.get("text", f"gt-{i}"), "bbox": to_xyxy(g["bbox"])})
    return img_path, gt, Path(args.image).stem


def get_omni_boxes(image, reuse_path=None):
    if reuse_path and Path(reuse_path).exists():
        return json.load(open(reuse_path))
    from browser.proposers import OmniParserDetector
    det = OmniParserDetector(with_captions=False)
    els = det.detect(image)
    det.release()
    import gc
    gc.collect()
    return [{"bbox": {"x": e["bbox"]["x"], "y": e["bbox"]["y"],
                      "width": e["bbox"]["width"],
                      "height": e["bbox"]["height"]},
             "score": e.get("score", 0.0)} for e in els]


def poly_to_box_origin(poly, ox, oy):
    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
    return [float(pts[:, 0].min()) + ox, float(pts[:, 1].min()) + oy,
            float(pts[:, 0].max()) + ox, float(pts[:, 1].max()) + oy]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=None)
    p.add_argument("--gt", default=None)
    p.add_argument("--screenspot-index", type=int, default=None)
    p.add_argument("--omniparser-boxes", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--iou-threshold", type=float, default=IOU_TP)
    p.add_argument("--paddings", default="0,10,20")
    args = p.parse_args(argv)

    img_path, gt, name = load_gt(args)
    out_dir = Path(args.out_dir) if args.out_dir else \
        REPO / "reports" / "tiny_text_detection"
    out_dir.mkdir(parents=True, exist_ok=True)
    paddings = [float(v) / 100.0 for v in args.paddings.split(",")]
    img_name = img_path.stem

    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size
    img_np = np.array(img)

    omni = get_omni_boxes(img, args.omniparser_boxes)
    omni_boxes = [to_xyxy(e["bbox"]) for e in omni]
    print(f"Image: {img_path.name} ({img_w}x{img_h}), "
          f"OmniParser boxes: {len(omni_boxes)}, GT regions: {len(gt)}",
          flush=True)

    # Experiment matrix: CRAFT with original-geometry vs internal
    # preprocessing, DBNet++ (fixed internal preprocessing for both).
    settings = [
        ("craft_original", "craft", "original"),
        ("craft_internal", "craft", "internal"),
        ("dbnet", "dbnet", "fixed-internal"),
    ]
    results = {"image": str(img_path), "img_size": [img_w, img_h],
               "target": gt[0]["text"] if len(gt) == 1 else f"{len(gt)} regions",
               "omniparser_boxes": len(omni_boxes), "experiments": {}}
    viz_first = {}

    for exp_name, det_name, mode in settings:
        results["experiments"][exp_name] = {}
        for pad in paddings:
            key = f"padding_{int(pad * 100)}"
            preds = []
            below_res = 0
            for oi, ob in enumerate(omni_boxes):
                pb = pad_box(ob, pad, img_w, img_h)
                cw, ch = pb[2] - pb[0], pb[3] - pb[1]
                below_res += int(min(cw, ch) < MIN_DIM_FLAG)
                crop = np.ascontiguousarray(
                    img_np[int(pb[1]):int(pb[3]), int(pb[0]):int(pb[2])])
                if crop.size == 0:
                    continue
                if det_name == "craft":
                    # Original: canvas locked to crop long side (no upscale).
                    # Internal: library default 2560 (normal preprocessing).
                    canvas = max(cw, ch) if mode == "original" else None
                    boxes = detect_craft_longsize(crop, canvas)
                else:
                    boxes = detect_dbnet(crop)
                for poly, score in boxes:
                    if np.asarray(poly).size < 6:
                        continue  # degenerate/empty polygon: no box to map
                    preds.append({
                        "text_bbox_original": poly_to_box_origin(poly, pb[0], pb[1]),
                        "omniparser_bbox": ob,
                        "crop_bbox": pb,
                        "detector_bbox_in_crop": poly,
                        "confidence": score,
                        "detector": det_name,
                    })
                if oi == 0 and pad == paddings[0] and exp_name not in viz_first:
                    viz_first[exp_name] = (ob, pb, crop.copy())
            results["experiments"][exp_name][key] = {
                "n_predictions": len(preds),
                "below_resolution_crops": below_res,
                "n_crops": len(omni_boxes),
                "predictions": preds,
            }
            print(f"  {exp_name} {key}: {len(preds)} preds, "
                  f"{below_res} below-res crops", flush=True)
    # crop viz for the first omni box (blue box + yellow padding)
    ob0 = omni_boxes[0]
    pb0 = pad_box(ob0, paddings[0], img_w, img_h)
    crop_viz = Image.fromarray(img_np).copy()
    dv = ImageDraw.Draw(crop_viz)
    dv.rectangle(ob0, outline=(0, 0, 255), width=3)
    dv.rectangle(pb0, outline=(255, 255, 0), width=2)
    crop_viz.save(out_dir / f"{img_name}_omni_crop.png")

    summary = evaluate(results, gt, args.iou_threshold)
    results["summary"] = summary
    results["target"] = gt[0]["text"] if len(gt) == 1 else results["target"]
    visualize(img_np, gt, results, out_dir, args.iou_threshold)
    with open(out_dir / "tiny_text_detection_results.json", "w") as f:
        json.dump(results, f, indent=1)

    print("\nTiny Text Detection Experiment")
    print("==============================================")
    print(f"\nImage: {img_path.name}\nTarget: {results['target']}\n")
    print("Detector       Padding    Detected    IoU    Support")
    print("------------------------------------------------------")
    for exp_name in ["craft_original", "craft_internal", "dbnet"]:
        for key in sorted(results["experiments"][exp_name].keys()):
            run = results["experiments"][exp_name][key]
            s = summary[exp_name][key]
            det = "YES" if s["recall"] > 0 else "NO"
            best = max([p.get("best_iou", 0) for p in run["predictions"]],
                       default=0.0)
            print(f"{exp_name:14s} {key:10s} {det:8s}  "
                  f"{best:<6.2f} {s['mean_support']:<6.2f}")
    print("\nSummary:")
    allious = [s for exp in summary.values() for s in exp.values()]
    print(f"Best detector: {max(summary, key=lambda e: summary[e]['padding_0']['recall'])}")
    print(f"Blog detected: {'YES' if any(summary[e][k]['recall'] > 0 for e in summary for k in summary[e]) else 'NO'}")
    print(f"Mean GT recall: {sum(s['recall'] for s in allious) / max(1, len(allious)):.3f}")
    print(f"Mean IoU: {sum(s['mean_iou'] for s in allious) / max(1, len(allious)):.3f}")
    print(f"Mean pixel support: {sum(s['mean_support'] for s in allious) / max(1, len(allious)):.3f}")
    print(f"Number of below-resolution crops: "
          f"{sum(r['below_resolution_crops'] if isinstance(r, dict) else 0 for exp in results['experiments'].values() for r in exp.values())}")
    print(f"\nArtifacts: {out_dir}")
    return results


def _ensure_certs():
    # Standard python.org-Mac fix: route TLS through certifi when present.
    try:
        import certifi, os
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except Exception:
        pass


def detect_craft_longsize(crop_np, canvas_override=None):
    """CRAFT detector via EasyOCR (detect stage only, no recognition).

    craft-text-detector is incompatible with modern torchvision, so the
    maintained EasyOCR CRAFT detector is used instead (same architecture
    family). canvas_size is the honest A/B lever: original-geometry runs
    pass canvas_size=crop long side (no upscaling); internal runs pass the
    2560 default. Returns [(poly_xy, score)] with score 1.0 (EasyOCR detect
    exposes no per-box confidence; thresholding happens inside).
    """
    _ensure_certs()
    if not hasattr(detect_craft_longsize, "_reader"):
        import easyocr
        detect_craft_longsize._reader = easyocr.Reader(
            ["en"], gpu=False, detector=True, recognizer=False,
            verbose=False, quantize=False)
    reader = detect_craft_longsize._reader
    h, v = reader.detect(
        crop_np,
        canvas_size=int(canvas_override) if canvas_override else 2560,
        mag_ratio=1.0)
    out = []
    for b in (h or []) + (v or []):
        pts = np.asarray(b, dtype=float).reshape(-1, 2)
        if pts.shape == (2, 2):  # [xmin,xmax],[ymin,ymax] form
            (x0, x1), (y0, y1) = pts[0], pts[1]
            pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        out.append((pts.tolist(), 1.0))
    return out


def detect_dbnet(crop_np):
    # NOTE: bare paddlex create_model() returns zero boxes even on rich
    # images (misconfigured defaults); the maintained TextDetection wrapper
    # from the paddleocr package is used instead (same PP-OCRv5_mobile_det
    # weights, detection-only, no recognition).
    if not hasattr(detect_dbnet, "_model"):
        from paddleocr._models.text_detection import TextDetection
        detect_dbnet._model = TextDetection(model_name="PP-OCRv5_mobile_det")
    out = []
    for r in detect_dbnet._model.predict(crop_np):
        # NOTE: TextDetResult is dict-like; attribute access misses keys.
        d = dict(r) if not isinstance(r, dict) else r
        polys = d.get("dt_polys", None)
        scores = d.get("dt_scores", None)
        if polys is None:
            continue
        for j, poly in enumerate(polys):
            pts = np.asarray(poly, dtype=float).reshape(-1, 2)
            s = float(np.asarray(scores[j]).reshape(-1)[0]) \
                if scores is not None else 1.0
            out.append((pts.tolist(), s))
    return out


def evaluate(results, gt, iou_threshold):
    """Recall, per-pred classification, pixel support (spec definitions)."""
    summary = {}
    for exp_name, exp in results["experiments"].items():
        summary[exp_name] = {}
        for key, run in exp.items():
            tp = misloc = fp = 0
            ious, sups = [], []
            for pr in run["predictions"]:
                best = max([box_iou(pr["text_bbox_original"], g["bbox"])
                            for g in gt] or [0.0])
                ious.append(best)
                gt_best = max(gt, key=lambda g: box_iou(
                    pr["text_bbox_original"], g["bbox"]))
                gb = gt_best["bbox"]
                pb = pr["text_bbox_original"]
                ix1, iy1 = max(pb[0], gb[0]), max(pb[1], gb[1])
                ix2, iy2 = min(pb[2], gb[2]), min(pb[3], gb[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                support = inter / max(1e-9, box_area(pb))
                sups.append(support)
                if best >= iou_threshold:
                    tp += 1
                    pr["status"] = "TRUE_POSITIVE"
                elif best > 0:
                    misloc += 1
                    pr["status"] = "MISLOCALIZED"
                else:
                    fp += 1
                    pr["status"] = "FALSE_POSITIVE"
                pr["best_iou"] = round(best, 3)
                pr["pixel_support"] = round(support, 3)
            matched = sum(1 for g in gt if any(
                box_iou(pr["text_bbox_original"], g["bbox"]) >= iou_threshold
                for pr in run["predictions"]))
            summary[exp_name][key] = {
                "recall": round(matched / max(1, len(gt)), 3),
                "tp": tp, "mislocalized": misloc, "fp": fp,
                "missed_gt": len(gt) - matched,
                "mean_iou": round(sum(ious) / max(1, len(ious)), 3),
                "mean_support": round(sum(sups) / max(1, len(sups)), 3),
            }
    return summary


def visualize(img_np, gt, results, out_dir, iou_threshold):
    """Crop viz + per-detector result + GT comparison (blue/yellow/red/green)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(img_np)
    first = results["experiments"]["craft_original"]["padding_0"]
    # 1. omni crop viz (first omni box, first padding) is per-run; use GT viz:
    for exp_name, exp in results["experiments"].items():
        for key, run in exp.items():
            base = img.copy()
            d = ImageDraw.Draw(base)
            for pr in run["predictions"]:
                b = pr["text_bbox_original"]
                d.rectangle(b, outline=(255, 0, 0), width=2)
            base.save(out_dir / f"{exp_name}_{key}.png")
    # 4. GT comparison (first padding run of each detector family)
    for exp_name in ["craft_original", "dbnet"]:
        run = results["experiments"][exp_name]["padding_0"]
        base = img.copy()
        d = ImageDraw.Draw(base)
        for g in gt:
            b = g["bbox"]
            d.rectangle(b, outline=(0, 200, 0), width=4)
        for pr in run["predictions"]:
            d.rectangle(pr["text_bbox_original"], outline=(255, 0, 0), width=2)
        for ob in set(tuple(r["omniparser_bbox"]) for r in run["predictions"]):
            d.rectangle(list(ob), outline=(0, 0, 255), width=2)
        leg = Image.new("RGB", (base.width, 30), "white")
        base.paste(leg, (0, 0))
        dl = ImageDraw.Draw(base)
        dl.text((10, 5), "GREEN=GT  RED=detector  BLUE=OmniParser",
                fill=(0, 0, 0))
        base.save(out_dir / f"{exp_name}_vs_gt.png")


if __name__ == "__main__":
    main()
