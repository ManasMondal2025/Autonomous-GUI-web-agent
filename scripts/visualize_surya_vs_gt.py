#!/usr/bin/env python3
"""GT vs Surya-0.4.2-classic OCR localization evaluation (standalone diagnostic).

- Loads ONE original screenshot, never resizes/crops it.
- GT bboxes: from a DOM-derived *.gt.json fixture, or --screenspot-index N
  (single GT bbox from data/screenspot/v2_mini.jsonl).
- Predictions: Surya OCR 0.4.2 classic ONLY (run with the classic venv, e.g.
  /tmp/surya_old/bin/python). No VLM anywhere in this script.
- Outputs: gt_bbox_visualization.png, surya_bbox_visualization.png,
  gt_vs_surya_overlay.png + console report + surya_vs_gt_results.json.

Usage:
  /tmp/surya_old/bin/python scripts/visualize_surya_vs_gt.py \\
      --image test_pages/fixtures/product.png \\
      --gt test_pages/fixtures/product.gt.json
  /tmp/surya_old/bin/python scripts/visualize_surya_vs_gt.py \\
      --screenspot-index 8
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.ERROR)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from utils.coordinate_utils import bbox_iou  # reuse, read-only
from PIL import Image, ImageDraw
import numpy as np

IOU_TP = 0.5


def norm(s):
    return " ".join(str(s or "").lower().split())


def text_match(a, b):
    a, b = norm(a), norm(b)
    return bool(a) and bool(b) and (a in b or b in a)


def to_xywh(bbox):
    if isinstance(bbox, dict):
        return {"x": float(bbox["x"]), "y": float(bbox["y"]),
                "width": float(bbox["width"]), "height": float(bbox["height"])}
    x1, y1, x2, y2 = (float(v) for v in bbox)  # [x1,y1,x2,y2] form
    return {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}


def pixel_support(gray, bbox):
    x1 = max(0, int(bbox["x"])); y1 = max(0, int(bbox["y"]))
    x2 = min(gray.shape[1], int(bbox["x"] + bbox["width"]))
    y2 = min(gray.shape[0], int(bbox["y"] + bbox["height"]))
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    return round(float((crop < 150).mean()), 3)


def run_surya_classic(image):
    try:
        from surya.model.detection.segformer import load_model as load_det
        from surya.model.detection.segformer import load_processor as load_det_proc
        from surya.model.recognition.model import load_model as load_rec
        from surya.model.recognition.processor import load_processor as load_rec_proc
        from surya.ocr import run_ocr
    except ImportError as e:
        raise SystemExit(
            "Surya 0.4.2 classic API not found. Run this script with the "
            f"classic venv (e.g. /tmp/surya_old/bin/python). Details: {e}")
    det_m, det_p = load_det(device="cpu"), load_det_proc()
    rec_m, rec_p = load_rec(device="cpu"), load_rec_proc()
    res = run_ocr([image], [["en"]], det_m, det_p, rec_m, rec_p)[0]
    preds = []
    for line in res.text_lines:
        poly = np.array(line.polygon, dtype=float)
        x1, y1 = float(poly[:, 0].min()), float(poly[:, 1].min())
        x2, y2 = float(poly[:, 0].max()), float(poly[:, 1].max())
        preds.append({"text": line.text,
                      "bbox": {"x": x1, "y": y1, "width": x2 - x1,
                               "height": y2 - y1},
                      "score": float(line.confidence)})
    return preds


def draw_boxes(base, boxes, color, width, label_fn):
    img = base.copy()
    d = ImageDraw.Draw(img)
    for i, b in enumerate(boxes):
        bb = b["bbox"]
        d.rectangle([bb["x"], bb["y"], bb["x"] + bb["width"],
                     bb["y"] + bb["height"]], outline=color, width=width)
        d.text((bb["x"] + 2, max(0, bb["y"] - 12)),
               label_fn(i, b), fill=color)
    return img


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=None)
    p.add_argument("--gt", default=None)
    p.add_argument("--screenspot-index", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--iou-threshold", type=float, default=IOU_TP)
    args = p.parse_args(argv)

    if args.screenspot_index is not None:
        samples = [json.loads(l) for l in
                   open(REPO / "data/screenspot/v2_mini.jsonl")]
        s = samples[args.screenspot_index]
        img_path = Path(s["img_path"])
        gt = [{"id": 0, "tag": "?", "text": s["instruction"],
               "bbox": to_xywh(s["bbox"])}]
        name = f"screenspot_{args.screenspot_index}"
    else:
        if not args.image or not args.gt:
            p.error("provide --image + --gt, or --screenspot-index")
        img_path = Path(args.image)
        gt = [{"id": g.get("id", i), "tag": g.get("tag", "?"),
               "text": g.get("text", ""),
               "bbox": to_xywh(g["bbox"])}
              for i, g in enumerate(json.load(open(args.gt)))]
        name = Path(args.image).stem

    out_dir = Path(args.out_dir) if args.out_dir else REPO / "reports" / "surya_vs_gt"
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(img_path).convert("RGB")  # original pixels, untouched
    gray = np.array(img.convert("L")).astype(float)
    preds = run_surya_classic(img)

    # --- match predictions to GT ---
    for pr in preds:
        best, best_iou = None, -1.0
        for g in gt:
            v = bbox_iou(pr["bbox"], g["bbox"])
            if v > best_iou:
                best, best_iou = g, v
        pr["best_gt"] = best["id"] if best else None
        pr["best_iou"] = round(best_iou, 3)
        pr["pixel_support"] = pixel_support(gray, pr["bbox"])
    gt_best = {}
    for g in gt:
        vals = [(pr, bbox_iou(pr["bbox"], g["bbox"])) for pr in preds]
        gt_best[g["id"]] = max(vals, key=lambda t: t[1]) if vals else (None, 0.0)

    # --- classify (spatial correctness first, text second) ---
    for pr in preds:
        g = next(x for x in gt if x["id"] == pr["best_gt"]) \
            if pr["best_gt"] is not None else None
        if pr["best_iou"] >= args.iou_threshold:
            pr["status"] = "TRUE_POSITIVE"
        elif g is not None and text_match(pr["text"], g["text"]):
            pr["status"] = "MISLOCALIZED"  # right words, wrong place
        else:
            pr["status"] = "FALSE_POSITIVE"
    missed = [g for g in gt
              if not any(pr["best_gt"] == g["id"] and
                         pr["best_iou"] >= args.iou_threshold for pr in preds)]

    # --- images (same pixels, annotations only) ---
    gt_items = [{"bbox": g["bbox"], "label": g["text"][:30]} for g in gt]
    draw_boxes(Image.open(img_path).convert("RGB"), gt_items, (0, 200, 0), 3,
               lambda i, b: gt_items[i]["label"]).save(
        out_dir / "gt_bbox_visualization.png")
    pred_items = [{"bbox": pr["bbox"],
                   "label": f"{i}:{pr['text'][:24]}:{pr.get('score', 0):.2f}"}
                  for i, pr in enumerate(preds)]
    draw_boxes(Image.open(img_path).convert("RGB"), pred_items, (255, 0, 0), 2,
               lambda i, b: pred_items[i]["label"]).save(
        out_dir / "surya_bbox_visualization.png")
    ov = draw_boxes(Image.open(img_path).convert("RGB"), gt_items, (0, 200, 0),
                    4, lambda i, b: "")
    ov = draw_boxes(ov, pred_items, (255, 0, 0), 2,
                    lambda i, b: str(i))
    ov.save(out_dir / "gt_vs_surya_overlay.png")
    # legend
    leg = Image.open(out_dir / "gt_vs_surya_overlay.png")
    dl = ImageDraw.Draw(leg)
    dl.rectangle([10, 10, 220, 44], fill=(255, 255, 255))
    dl.text((16, 14), "GREEN = Ground Truth", fill=(0, 180, 0))
    dl.text((16, 28), "RED = Surya OCR 0.4.2 Classic", fill=(255, 0, 0))
    leg.save(out_dir / "gt_vs_surya_overlay.png")

    # --- report ---
    tp = sum(1 for pr in preds if pr["status"] == "TRUE_POSITIVE")
    rec = {"image": str(img_path), "n_gt": len(gt), "n_pred": len(preds),
           "predictions": [
               {"text": pr["text"][:60], "bbox": pr["bbox"],
                "score": round(pr.get("score", 0), 3),
                "best_gt": pr["best_gt"], "best_iou": pr["best_iou"],
                "pixel_support": pr["pixel_support"], "status": pr["status"]}
               for pr in preds],
           "missed_gt": [{"id": g["id"], "text": g["text"]} for g in missed]}
    (out_dir / "surya_vs_gt_results.json").write_text(json.dumps(rec, indent=1))

    print(f"Image: {img_path.name}\n\nGT regions: {len(gt)}\n"
          f"Surya predictions: {len(preds)}\n")
    for pr in rec["predictions"]:
        g = next((x for x in gt if x["id"] == pr["best_gt"]), {"text": "?"})
        print(f"Surya: {pr['text'][:40]!r}\n  best GT: {g['text'][:40]!r} "
              f"(id={pr['best_gt']}) IoU={pr['best_iou']} "
              f"pixel_support={pr['pixel_support']} -> {pr['status']}")
    print(f"\nMissed GT: {[g['text'][:30] for g in missed] or 'none'}")
    ious = [pr["best_iou"] for pr in rec["predictions"]]
    sup = [pr["pixel_support"] for pr in rec["predictions"]]
    print(f"\nOverall:\nGT recall: {len(gt) - len(missed)}/{len(gt)}="
          f"{(len(gt) - len(missed)) / max(1, len(gt)):.1%}")
    print(f"TP rate: {tp}/{len(preds)}={tp / max(1, len(preds)):.1%}")
    print(f"Mean IoU: {sum(ious) / max(1, len(ious)):.3f} | "
          f"Mean pixel support: {sum(sup) / max(1, len(sup)):.3f}")
    print(f"\nArtifacts: {out_dir}")


if __name__ == "__main__":
    main()
