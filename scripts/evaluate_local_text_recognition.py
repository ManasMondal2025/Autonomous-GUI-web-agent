#!/usr/bin/env python3
"""Local text-recognition experiment: can CRAFT/DBNet++ + tight-crop Surya
recover tiny GUI text that direct Surya misses?

Standalone diagnostic. Modifies NOTHING in the agent pipeline (no changes to
OmniParser, OCR backend, association, grounding, prompts, or harness).

Phases (split across venvs by necessity; one file, --phase flag):
  detect    (main env) : YOLO Omni boxes + padded crops + CRAFT/DBNet++
                         tight boxes (original coords). No recognition, no VLM.
  recognize (surya 0.4.2 venv, e.g. /tmp/surya_old/bin/python):
                         Surya direct on crops + on tight boxes. No VLM
                         (classic Donut-based recognition only).
  report    (main env) : GT metrics, 4 visualizations, console table, JSON.

Usage:
  python3 scripts/evaluate_local_text_recognition.py --phase detect
  /tmp/surya_old/bin/python scripts/evaluate_local_text_recognition.py --phase recognize
  python3 scripts/evaluate_local_text_recognition.py --phase report
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

OUT_DIR = REPO / "reports" / "local_text_recognition"
CROP_DIR = OUT_DIR / "crops"
PADDINGS = [0, 10, 20]

IMAGES = {
    "web_cmyhc5": {
        "img": ("GT-DATASET", "screenspotv2_image/web_cmyhc5.png"),
        "gt": [{"text": "Blog", "bbox": [869, 9, 964, 60]}],
    },
    "product": {
        "img": ("FIXTURE", "test_pages/fixtures/product.png"),
        "gt": None,  # DOM button boxes from product.gt.json
    },
}

GT_DATASET_ROOT = Path("/Users/manasmondal/Desktop/GUI-Web-Agent/dataset/ScreenSpot-v2")


def resolve_image(spec):
    kind, rel = spec
    if kind == "GT-DATASET":
        return GT_DATASET_ROOT / rel
    return REPO / rel


def load_product_gt():
    gt = []
    for e in json.load(open(REPO / "test_pages/fixtures/product.gt.json")):
        b = e["bbox"]
        t = (e.get("text") or "").strip()
        if t:
            gt.append({"text": t,
                       "bbox": [b["x"], b["y"],
                                b["x"] + b["width"], b["y"] + b["height"]]})
    return gt


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


def norm_text(s):
    return " ".join(str(s or "").lower().split())


# ---------------------------------------------------------------- detect
def run_detect():
    import numpy as np
    from PIL import Image
    from browser.proposers import OmniParserDetector
    import easyocr
    from paddleocr._models.text_detection import TextDetection

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    reader = easyocr.Reader(["en"], gpu=False, detector=True,
                            recognizer=False, verbose=False, quantize=False)
    dbnet = TextDetection(model_name="PP-OCRv5_mobile_det")
    det = OmniParserDetector(with_captions=False)
    manifest = {"images": {}}

    for iname, spec in IMAGES.items():
        img_path = resolve_image(spec["img"])
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        img_np = np.array(img)
        els = det.detect(img)
        omni = []
        for e in els:
            b = e["bbox"]
            omni.append([float(b["x"]), float(b["y"]),
                         float(b["x"] + b["width"]),
                         float(b["y"] + b["height"])])
        entry = {"img_path": str(img_path), "img_size": [img_w, img_h],
                 "gt": spec["gt"] or load_product_gt(),
                 "omni_boxes": omni, "crops": []}
        for oi, ob in enumerate(omni):
            for pad in PADDINGS:
                pb = pad_box(ob, pad / 100.0, img_w, img_h)
                x1, y1, x2, y2 = (int(round(v)) for v in pb)
                crop = np.ascontiguousarray(img_np[y1:y2, x1:x2])
                if crop.size == 0:
                    continue
                cf = f"{iname}_omni{oi:02d}_pad{pad}.png"
                Image.fromarray(crop).save(CROP_DIR / cf)
                cw, ch = x2 - x1, y2 - y1
                # CRAFT, original geometry (canvas locked to crop long side)
                h, v = reader.detect(
                    crop, canvas_size=max(cw, ch), mag_ratio=1.0)
                craft = []
                for b in (h or []) + (v or []):
                    pts = np.asarray(b, dtype=float).reshape(-1, 2)
                    if pts.shape == (2, 2):
                        (a, c), (d, e) = pts[0], pts[1]
                        pts = np.array([[a, d], [c, d], [c, e], [a, e]])
                    if pts.size < 6:
                        continue
                    craft.append([
                        float(pts[:, 0].min()) + x1,
                        float(pts[:, 1].min()) + y1,
                        float(pts[:, 0].max()) + x1,
                        float(pts[:, 1].max()) + y1])
                # DBNet++, fixed internal preprocessing
                db = []
                for r in dbnet.predict(crop):
                    d = dict(r)
                    _polys = d.get("dt_polys")
                    _polys = [] if _polys is None else list(_polys)
                    for poly in _polys:
                        pts = np.asarray(poly, dtype=float).reshape(-1, 2)
                        db.append([
                            float(pts[:, 0].min()) + x1,
                            float(pts[:, 1].min()) + y1,
                            float(pts[:, 0].max()) + x1,
                            float(pts[:, 1].max()) + y1])
                entry["crops"].append({
                    "crop_file": cf, "omni_idx": oi,
                    "omniparser_bbox": ob, "padding": pad,
                    "crop_bbox": [x1, y1, x2, y2],
                    "crop_size": [cw, ch],
                    "below_resolution": min(cw, ch) < 16,
                    "craft_boxes": craft, "dbnet_boxes": db,
                })
        manifest["images"][iname] = entry
        print(f"{iname}: {len(omni)} omni boxes -> "
              f"{len(entry['crops'])} crops", flush=True)
    det.release()
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"manifest: {OUT_DIR / 'manifest.json'}")


# --------------------------------------------------------------- recognize
def run_recognize():
    from PIL import Image
    from surya.model.recognition.model import load_model
    from surya.model.recognition.processor import load_processor
    from surya.ocr import run_recognition

    manifest = json.load(open(OUT_DIR / "manifest.json"))
    rec_model = load_model(device="cpu")
    rec_proc = load_processor()

    def rec_batch(items):
        """items: [(crop_file, [x1,y1,x2,y2]-in-crop-boxes-or-None]."""
        if not items:
            return []
        imgs = [Image.open(CROP_DIR / cf).convert("RGB") for cf, _ in items]
        box_lists = []
        for (_, boxes), img in zip(items, imgs):
            if boxes is None:  # whole crop
                w, h = img.size
                box_lists.append([[0, 0, w, h]])
            else:
                box_lists.append([list(map(float, b)) for b in boxes])
        t0 = time.time()
        res = run_recognition(imgs, [["en"]] * len(imgs),
                              rec_model, rec_proc, bboxes=box_lists)
        out = []
        for r in res:
            lines = []
            for tl in r.text_lines:
                poly = getattr(tl, "polygon", None)
                lines.append({
                    "text_raw": tl.text,
                    "confidence": tl.confidence,
                    "poly": poly.tolist() if poly is not None and
                    hasattr(poly, "tolist") else poly,
                })
            out.append(lines)
        return out, round(time.time() - t0, 2)

    for iname, entry in manifest["images"].items():
        entry["timings"] = {}
        # A. direct: whole crop
        t_direct = time.time()
        direct_items = [(c["crop_file"], None) for c in entry["crops"]]
        direct_out, direct_s = rec_batch(direct_items)
        t_tight = time.time()
        # B/C. tight boxes, mapped back to original screenshot coords
        for src, rkey in (("craft_boxes", "craft_rec"),
                          ("dbnet_boxes", "dbnet_rec")):
            items, backmap = [], []
            for c in entry["crops"]:
                ox, oy = c["crop_bbox"][0], c["crop_bbox"][1]
                for b in c[src]:
                    # to crop coords
                    items.append((c["crop_file"],
                                  [[b[0] - ox, b[1] - oy, b[2] - ox, b[3] - oy]]))
                    backmap.append((c["crop_file"], b))
            outs, secs = rec_batch(items) if items else ([], 0.0)
            entry[rkey] = []
            for (cf, ob), lines in zip(backmap, outs):
                for tl in lines:
                    entry[rkey].append({
                        "crop_file": cf,
                        "detector_bbox_original": ob,
                        "text_raw": tl["text_raw"],
                        "confidence": tl["confidence"],
                    })
            entry[rkey + "_s"] = secs
        # attach direct outputs to crops (polys mapped to original coords);
        # timings recorded per stage for latency reporting
        entry["timings"]["direct_s"] = round(time.time() - t_direct, 2)
        import numpy as np
        for c, lines in zip(entry["crops"], direct_out):
            ox, oy = c["crop_bbox"][0], c["crop_bbox"][1]
            for tl in lines:
                if tl.get("poly") is not None:
                    pts = np.asarray(tl["poly"], dtype=float).reshape(-1, 2)
                    tl["bbox_original"] = [
                        float(pts[:, 0].min()) + ox,
                        float(pts[:, 1].min()) + oy,
                        float(pts[:, 0].max()) + ox,
                        float(pts[:, 1].max()) + oy]
                else:
                    tl["bbox_original"] = c["crop_bbox"]
            c["direct_rec"] = lines
        entry["direct_rec_s"] = direct_s
        print(f"{iname}: direct + tight recognition done", flush=True)
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("recognition done")


# ------------------------------------------------------------------ report
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["detect", "recognize", "report"],
                   required=True)
    args = p.parse_args(argv)
    if args.phase == "detect":
        run_detect()
    elif args.phase == "recognize":
        run_recognize()
    else:
        run_report()


def run_report():
    import numpy as np
    from PIL import Image, ImageDraw

    manifest = json.load(open(OUT_DIR / "manifest.json"))
    all_rows = []
    for iname, entry in manifest["images"].items():
        gt = entry["gt"]
        img = Image.open(resolve_image(IMAGES[iname]["img"])).convert("RGB")
        # pipelines: direct (per-crop text), craft, dbnet (per-tight-box text)
        for pipe, key in (("direct", None), ("craft", "craft_rec"),
                          ("dbnet", "dbnet_rec")):
            preds = []  # (text, conf, orig_box)
            if pipe == "direct":
                for c in entry["crops"]:
                    for tl in c.get("direct_rec", []):
                        preds.append((tl["text_raw"], tl["confidence"],
                                      tl.get("bbox_original") or c["crop_bbox"],
                                      c))
            else:
                for r in entry.get(key, []):
                    preds.append((r["text_raw"], r["confidence"],
                                  r["detector_bbox_original"],
                                  next(c for c in entry["crops"]
                                       if c["crop_file"] == r["crop_file"])))
            # GT matching
            n_tp = n_fp = n_miss = 0
            ious, per_gt = [], []
            for g in gt:
                gb = g["bbox"]
                best, best_pr = 0.0, None
                for pr in preds:
                    v = box_iou(pr[2], gb)
                    if v > best:
                        best, best_pr = v, pr
                ious.append(best)
                hit = best >= 0.5
                per_gt.append({"gt": g["text"][:40], "best_iou": round(best, 3),
                               "hit": hit,
                               "pred_text": (best_pr[0][:40] if best_pr else None)})
                if hit:
                    n_tp += 1
                else:
                    n_miss += 1
            n_fp = sum(1 for pr in preds if all(
                box_iou(pr[2], g["bbox"]) < 0.5 for g in gt))
            # text accuracy: GT whose best box has matching normalized text
            n_txt = 0
            for g, pg in zip(gt, per_gt):
                if not pg["hit"]:
                    continue
                # find best pred again for text check
                cands = [(pr, box_iou(pr[2], g["bbox"])) for pr in preds]
                cands.sort(key=lambda t: -t[1])
                if cands and norm_text(cands[0][0][0]) == norm_text(g["text"]):
                    n_txt += 1
            all_rows.append({
                "image": iname, "pipeline": pipe,
                "n_gt": len(gt), "n_pred": len(preds),
                "recall": round(n_tp / max(1, len(gt)), 3),
                "mean_iou": round(sum(ious) / max(1, len(ious)), 3),
                "text_acc": round(n_txt / max(1, len(gt)), 3),
                "fp": n_fp, "missed": n_miss,
            })
        # visualizations for this image
        mk_viz(img, entry, gt, iname)
    _print_table(all_rows)
    (OUT_DIR / "local_text_results.json").write_text(
        json.dumps({"rows": all_rows,
                    "manifest": str(OUT_DIR / "manifest.json")}, indent=1))
    print(f"\nJSON: {OUT_DIR / 'local_text_results.json'}")
    print("\nLatency (s):")
    for iname, entry in manifest["images"].items():
        pass  # timings live in entry["<stage>_rec_s"] keys
        n_crops = len(entry["crops"])
        n_tight = len(entry.get("craft_rec", [])) + len(entry.get("dbnet_rec", []))
        print(f"  {iname}: direct={entry.get('direct_rec_s', '?')}s "
              f"tight(craft {entry.get('craft_rec_s', '?')}s + "
              f"dbnet {entry.get('dbnet_rec_s', '?')}s) "
              f"over {n_crops} crops / "
              f"{n_tight} tight boxes")


def norm_text(s):
    return " ".join(str(s or "").lower().split())


def mk_viz(img, entry, gt, iname):
    from PIL import Image, ImageDraw
    # per-pipeline full-image views
    for pipe, key, color in (("direct", None, (255, 0, 0)),
                             ("craft", "craft_rec", (0, 180, 0)),
                             ("dbnet", "dbnet_rec", (255, 0, 0))):
        base = img.copy()
        d = ImageDraw.Draw(base)
        if pipe == "direct":
            # omni boxes blue + Surya line boxes red with text/conf
            seen = set()
            for c in entry["crops"]:
                ob = c["omniparser_bbox"]
                if tuple(ob) not in seen:
                    d.rectangle(ob, outline=(0, 0, 255), width=2)
                    seen.add(tuple(ob))
                for tl in c.get("direct_rec", []):
                    bb = tl.get("bbox_original") or c["crop_bbox"]
                    d.rectangle(bb, outline=(255, 0, 0), width=2)
                    d.text((bb[0], max(0, bb[1] - 12)),
                           f"{tl['text_raw'][:24]}:{tl['confidence'] or 0.0:.2f}",
                           fill=(255, 0, 0))
            base.save(OUT_DIR / f"{iname}_surya_direct.png")
        else:
            for r in entry.get(key, []):
                d.rectangle(r["detector_bbox_original"],
                            outline=(0, 180, 0), width=2)
                d.text((r["detector_bbox_original"][0],
                        max(0, r["detector_bbox_original"][1] - 12)),
                       r["text_raw"][:28], fill=(0, 180, 0))
            base.save(OUT_DIR / f"{iname}_{pipe}_surya.png")
    # comparison: left = best direct crop view, right = best tight view
    crops = entry["crops"]
    best = None
    for c in crops:
        for tl in c.get("direct_rec", []):
            best = c
            break
        if best:
            break
    left = Image.open(CROP_DIR / (best["crop_file"] if best else crops[0]["crop_file"])).convert("RGB")
    tight = None
    for r in (entry.get("craft_rec", []) + entry.get("dbnet_rec", [])):
        tight = r
        break
    if tight is not None:
        b = tight["detector_bbox_original"]
        region = img.crop((int(max(0, b[0] - 20)), int(max(0, b[1] - 20)),
                           int(min(img.width, b[2] + 20)),
                           int(min(img.height, b[3] + 20)))).convert("RGB")
    else:
        region = left.copy()
    w = max(left.width, region.width)
    canvas = Image.new("RGB", (w * 2 + 20, max(left.height, region.height) + 30),
                       "white")
    canvas.paste(left, (0, 30))
    canvas.paste(region, (w + 20, 30))
    dd = ImageDraw.Draw(canvas)
    dd.text((10, 8), "LEFT: Surya direct on Omni crop", fill=(0, 0, 0))
    dd.text((w + 30, 8), "RIGHT: tight box -> Surya", fill=(0, 0, 0))
    canvas.save(OUT_DIR / f"{iname}_comparison.png")


def _print_table(all_rows):
    print("\n" + "=" * 68)
    print("LOCAL TEXT RECOGNITION EXPERIMENT")
    print("=" * 68 + "\n")
    print("Image                Pipeline      Recall   MeanIoU  TxtAcc   FP  Miss")
    print("-" * 68)
    for r in all_rows:
        print(f"{r['image']:20s} {r['pipeline']:12s} "
              f"{r['recall']:<8.2f} {r['mean_iou']:<8.2f} "
              f"{r['text_acc']:<8.2f} {r['fp']:<3d} {r['missed']:<3d}")
    print("=" * 68)


if __name__ == "__main__":
    main()
