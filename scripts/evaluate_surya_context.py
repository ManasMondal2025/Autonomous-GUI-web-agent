#!/usr/bin/env python3
"""Surya 0.4.2 context-sensitivity diagnostic (standalone, Surya-classic ONLY).

Answers: why does Surya read "Blog" from the exact OmniParser crop but fail
when context is added? No VLM, no PaddleOCR/CRAFT/DBNet, no pipeline changes.

RUN WITH the classic venv: /tmp/surya_old/bin/python scripts/evaluate_surya_context.py

Phases (all in this file, batched for speed, identical model/config):
  1. padding sweep (-20..+50%) on the Blog omni region
  2. directional + one-sided padding
  3. canonical crops table (exact/+5/+10/+20/+50)
  4. preprocessing variants on the 0% and +10% crops
  5. crop image strips (0/5/10/20/50 + comparison strip)
  6. context masking on +10%/+20% (GT box preserved, rest neutral gray)
  7. multi-crop consensus over -5/0/+5/+10
  8. size-stratified regions (5 tiny / 5 medium / 5 normal) x paddings
Outputs: reports/surya_context_sensitivity/{results.json,summary.csv,
  blog_padding_comparison.png, blog_preprocessing_comparison.png,
  blog_context_mask_comparison.png}
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.ERROR)

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "reports" / "surya_context_sensitivity"
MANIFEST = REPO / "reports" / "local_text_recognition" / "manifest.json"

GT_BLOG = {"text": "Blog", "bbox": [869, 9, 964, 60]}
BLOG_IMG = Path("/Users/manasmondal/Desktop/GUI-Web-Agent/dataset/ScreenSpot-v2/screenspotv2_image/web_cmyhc5.png")


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def norm_text(s):
    return " ".join(str(s or "").lower().split())


def pad_box_frac(b, fx, fy=None, sides=None, img_w=None, img_h=None):
    """Fractional padding. sides: subset of {'left','right','top','bottom'}
    or None for all sides. Negative fractions shrink toward the center."""
    if fy is None:
        fy = fx
    sides = sides or {"left", "right", "top", "bottom"}
    w, h = b[2] - b[0], b[3] - b[1]
    x1 = b[0] - (w * fx if "left" in sides else 0)
    y1 = b[1] - (h * fy if "top" in sides else 0)
    x2 = b[2] + (w * fx if "right" in sides else 0)
    y2 = b[3] + (h * fy if "bottom" in sides else 0)
    if img_w and img_h:
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(img_w), x2), min(float(img_h), y2)
    return [x1, y1, x2, y2]


class SuryaRunner:
    """Classic Surya recognition, one model load, batched calls."""

    def __init__(self):
        from surya.model.recognition.model import load_model
        from surya.model.recognition.processor import load_processor
        self.model = load_model(device="cpu")
        self.processor = load_processor()
        self.calls = 0

    def recognize(self, crops):
        """crops: list of PIL images. Returns list of line-lists."""
        from surya.ocr import run_recognition
        if not crops:
            return []
        self.calls += 1
        t0 = time.time()
        box_lists = []
        for c in crops:
            w, h = c.size
            box_lists.append([[0, 0, w, h]])
        res = run_recognition(
            crops, [["en"]] * len(crops),
            self.model, self.processor, bboxes=box_lists)
        out = []
        for r in res:
            lines = []
            for tl in r.text_lines:
                poly = getattr(tl, "polygon", None)
                try:
                    import numpy as np
                    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
                    bb = [float(pts[:, 0].min()), float(pts[:, 1].min()),
                          float(pts[:, 0].max()), float(pts[:, 1].max())]
                except Exception:
                    bb = None
                lines.append({"text_raw": tl.text,
                              "confidence": tl.confidence,
                              "poly": bb})
            out.append(lines)
        return out, round(time.time() - t0, 2)


def main(argv=None):
    from PIL import Image, ImageDraw, ImageFilter, ImageOps

    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--strat-boxes", default=None,
                   help="JSON list of [x1,y1,x2,y2] boxes for Exp 8 "
                        "(generate in main env; YOLO weights live there)")
    args = p.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.load(open(MANIFEST))
    omni = manifest["images"]["web_cmyhc5"]["omni_boxes"] \
        if "web_cmyhc5" in manifest["images"] \
        else manifest["images"][list(manifest["images"])[0]]["omni_boxes"]
    # The Omni region overlapping GT most (computed, not hardcoded).
    base_box = max(omni, key=lambda b: box_iou(b, GT_BLOG["bbox"]))
    print(f"Base Omni box (max GT overlap): "
          f"{[round(v, 1) for v in base_box]}", flush=True)

    img = Image.open(BLOG_IMG).convert("RGB")
    img_w, img_h = img.size
    runner = SuryaRunner()
    rec = {"mask_runs": []}

    def run_case(name, crop, meta=None, origin=(0, 0), scale=1.0,
                 gt_override=None):
        # origin: top-left of this crop in screenshot coords; scale: the
        # crop was resized by this factor before inference (2x/4x variants).
        # gt_override: per-case GT box in the SAME frame as shifted polys
        # (synthetic canvases); defaults to the screenshot GT_BLOG.
        ox, oy = origin
        gt = gt_override or GT_BLOG["bbox"]
        if crop.size[0] < 2 or crop.size[1] < 2:
            rec[name] = {"texts": [], "norm": [], "hit_blog": False,
                         "best_iou": 0.0, "best_box": None, "secs": 0.0,
                         "crop_size": list(crop.size),
                         "note": "degenerate crop, skipped"}
            print(f"{name:22s} SKIPPED (degenerate crop)", flush=True)
            return crop
        t0 = time.time()
        out, _ = runner.recognize([crop])
        secs = round(time.time() - t0, 2)
        lines = out[0] if out else []
        texts = [norm_text(l["text_raw"]) for l in lines]
        hit_blog = any("blog" in t for t in texts)
        best_iou, best_box = 0.0, None
        for l in lines:
            if l["poly"] is None:
                continue
            # line polys are in CROP coords: translate to screenshot coords
            import numpy as _np
            pts = _np.asarray(l["poly"], dtype=float).reshape(-1, 2)
            shifted = [float(pts[:, 0].min()) / scale + ox,
                       float(pts[:, 1].min()) / scale + oy,
                       float(pts[:, 0].max()) / scale + ox,
                       float(pts[:, 1].max()) / scale + oy]
            v = box_iou(shifted, gt)
            if v > best_iou:
                best_iou, best_box = v, [round(x, 1) for x in shifted]
        gray = crop.convert("L")
        import numpy as np
        _ = np  # noqa (kept for clarity)
        if best_box is not None:
            ix1, iy1 = max(best_box[0], gt[0]), max(best_box[1], gt[1])
            ix2, iy2 = min(best_box[2], gt[2]), min(best_box[3], gt[3])
            support = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1) / max(
                1e-9, box_area(best_box))
        else:
            support = 0.0
        rec[name] = {"texts": [l["text_raw"] for l in lines],
                     "norm": texts, "hit_blog": hit_blog,
                     "best_iou": round(best_iou, 3),
                     "best_box": best_box,
                     "pixel_support": round(support, 3),
                     "secs": secs, "crop_size": list(crop.size)}
        if meta:
            rec[name].update(meta)
        print(f"{name:22s} text={texts} iou={best_iou:.3f} "
              f"support={rec[name].get('pixel_support', 0.0):.3f} "
              f"{secs:.1f}s", flush=True)
        return crop

    crops_saved = {}

    # --- Exp 1b: one-pixel boundary sweep (per-boundary independence) ---
    for boundary, axis in (("left", 0), ("right", 2), ("top", 1), ("bottom", 3)):
        for off in range(-5, 6):
            b = list(base_box)
            b[axis] += off
            x1, y1, x2, y2 = (int(round(v)) for v in
                              [max(0, b[0]), max(0, b[1]),
                               min(img_w, b[2]), min(img_h, b[3])])
            if x2 <= x1 + 1 or y2 <= y1 + 1:
                continue
            run_case(f"edge_{boundary}_{off:+d}",
                     img.crop((x1, y1, x2, y2)),
                     {"crop_bbox": [x1, y1, x2, y2],
                      "boundary": boundary, "offset_px": off},
                     origin=(x1, y1))

    # --- Exp 2/3: 11x11 left/right and top/bottom heatmaps ---
    for lo in range(-5, 6):
        for ro in range(-5, 6):
            b = [base_box[0] + lo, base_box[1], base_box[2] + ro, base_box[3]]
            x1, y1, x2, y2 = (int(round(v)) for v in
                              [max(0, b[0]), max(0, b[1]),
                               min(img_w, b[2]), min(img_h, b[3])])
            if x2 > x1 + 1 and y2 > y1 + 1:
                run_case(f"lr_{lo:+d}_{ro:+d}",
                         img.crop((x1, y1, x2, y2)),
                         {"crop_bbox": [x1, y1, x2, y2],
                          "left": lo, "right": ro}, origin=(x1, y1))
    for to in range(-5, 6):
        for bo in range(-5, 6):
            b = [base_box[0], base_box[1] + to, base_box[2], base_box[3] + bo]
            x1, y1, x2, y2 = (int(round(v)) for v in
                              [max(0, b[0]), max(0, b[1]),
                               min(img_w, b[2]), min(img_h, b[3])])
            if x2 > x1 + 1 and y2 > y1 + 1:
                run_case(f"tb_{to:+d}_{bo:+d}",
                         img.crop((x1, y1, x2, y2)),
                         {"crop_bbox": [x1, y1, x2, y2],
                          "top": to, "bottom": bo}, origin=(x1, y1))

    # --- Exp 1: fine-grained padding sweep ---
    for pct in [-20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 40, 50]:
        f = pct / 100.0
        pb = pad_box_frac(base_box, f, img_w=img_w, img_h=img_h)
        x1, y1, x2, y2 = (int(round(v)) for v in pb)
        crop = img.crop((x1, y1, x2, y2))
        crops_saved[f"pad{pct:+d}"] = crop
        run_case(f"pad{pct:+d}",
                 crop, {"crop_bbox": [x1, y1, x2, y2]}, origin=(x1, y1))

    # --- Exp 2: directional + one-sided ---
    for frac, sides, tag in [
            (0.05, {"left", "right"}, "h5"), (0.10, {"left", "right"}, "h10"),
            (0.20, {"left", "right"}, "h20"), (0.30, {"left", "right"}, "h30"),
            (0.05, {"top", "bottom"}, "v5"), (0.10, {"top", "bottom"}, "v10"),
            (0.20, {"top", "bottom"}, "v20"), (0.30, {"top", "bottom"}, "v30"),
            (0.05, {"left"}, "left5"), (0.10, {"left"}, "left10"),
            (0.20, {"left"}, "left20"),
            (0.05, {"right"}, "right5"), (0.10, {"right"}, "right10"),
            (0.20, {"right"}, "right20"),
            (0.05, {"top"}, "top5"), (0.10, {"top"}, "top10"),
            (0.20, {"top"}, "top20"),
            (0.05, {"bottom"}, "bot5"), (0.10, {"bottom"}, "bot10"),
            (0.20, {"bottom"}, "bot20")]:
        pb = pad_box_frac(base_box, frac, img_w=img_w, img_h=img_h) \
            if sides == {"left", "right", "top", "bottom"} else None
        # directional: pad only along the chosen axes
        w, h = base_box[2] - base_box[0], base_box[3] - base_box[1]
        fx = frac if sides & {"left", "right"} else 0.0
        fy = frac if sides & {"top", "bottom"} else 0.0
        pb = pad_box_frac(base_box, 0.0, img_w=img_w, img_h=img_h)
        pb = [pb[0] - (w * fx if "left" in sides else 0),
              pb[1] - (h * fy if "top" in sides else 0),
              pb[2] + (w * fx if "right" in sides else 0),
              pb[3] + (h * fy if "bottom" in sides else 0)]
        pb = [max(0.0, pb[0]), max(0.0, pb[1]),
              min(float(img_w), pb[2]), min(float(img_h), pb[3])]
        x1, y1, x2, y2 = (int(round(v)) for v in pb)
        run_case(f"dir_{tag}", img.crop((x1, y1, x2, y2)),
                 {"crop_bbox": [x1, y1, x2, y2], "sides": sorted(sides)},
                 origin=(x1, y1))

    # --- Exp 4: preprocessing on 0% and +10% crops ---
    base0 = crops_saved["pad+0"]
    base10 = crops_saved["pad+10"]
    variants = {
        "orig": lambda c: c,
        "gray": lambda c: c.convert("L").convert("RGB"),
        "contrast": lambda c: ImageOps.autocontrast(c.convert("L")).convert("RGB"),
        "sharpen": lambda c: c.filter(ImageFilter.SHARPEN),
        "2x": lambda c: c.resize((c.width * 2, c.height * 2), Image.LANCZOS),
        "4x": lambda c: c.resize((c.width * 4, c.height * 4), Image.LANCZOS),
    }
    for cname, crop, ox, oy in (("crop0", base0, *rec["pad+0"]["crop_bbox"][:2]),
                            ("crop10", base10, *rec["pad+10"]["crop_bbox"][:2])):
        for vname, fn in variants.items():
            scale = {"2x": 2.0, "4x": 4.0}.get(vname, 1.0)
            run_case(f"pre_{cname}_{vname}", fn(crop), origin=(ox, oy),
                     scale=scale)

    # --- Exp 4 (new): 3x3 position grid on fixed 400x200 canvas ----
    # GT for synthetic canvases = placed Blog box in canvas coords.
    bx = [int(round(v)) for v in base_box]
    exact = img.crop((bx[0], bx[1], bx[2], bx[3]))
    ew, eh = exact.size
    cw, ch = 400, 200
    for pname, (px, py) in {
            "top-left": (0, 0), "top-center": ((cw - ew) // 2, 0),
            "top-right": (cw - ew, 0),
            "middle-left": (0, (ch - eh) // 2),
            "center": ((cw - ew) // 2, (ch - eh) // 2),
            "middle-right": (cw - ew, (ch - eh) // 2),
            "bottom-left": (0, ch - eh),
            "bottom-center": ((cw - ew) // 2, ch - eh),
            "bottom-right": (cw - ew, ch - eh)}.items():
        cv = Image.new("RGB", (cw, ch), (128, 128, 128))
        cv.paste(exact, (px, py))
        run_case(f"pos_{pname}", cv, origin=(0, 0),
                 scale=1.0, gt_override=[px, py, px + ew, py + eh])

    # --- Exp 5 (new): aspect canvases, Blog centered ---
    for rw, rh in [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (4, 1), (1, 4)]:
        s = max((ew + 20) / rw, (eh + 20) / rh, 1.0)
        W, H = int(round(rw * s)), int(round(rh * s))
        cv = Image.new("RGB", (W, H), (128, 128, 128))
        px, py = (W - ew) // 2, (H - eh) // 2
        cv.paste(exact, (px, py))
        run_case(f"aspect_{rw}x{rh}", cv, origin=(0, 0),
                 scale=1.0, gt_override=[px, py, px + ew, py + eh])

    # --- Exp 6 (new): scale sweep on fixed 400x200 canvas ---
    SCW, SCH = 400, 200
    for sc in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0]:
        nw, nh = max(1, int(round(ew * sc))), max(1, int(round(eh * sc)))
        thumb = exact.resize((nw, nh), Image.LANCZOS)
        cv = Image.new("RGB", (SCW, SCH), (128, 128, 128))
        px, py = (SCW - nw) // 2, (SCH - nh) // 2
        cv.paste(thumb, (px, py))
        run_case(f"scale_{sc}x", cv, origin=(0, 0),
                 scale=1.0, gt_override=[px, py, px + nw, py + nh])

    # --- Exp 6: context masking on +10/+20 (GT preserved, rest gray 128) ---
    for key in ["+10", "+20"]:
        crop = crops_saved[f"pad{key}"].copy()
        cb = rec[f"pad{key}"]["crop_bbox"]
        masked = mask_context(crop, cb)
        run_case(f"masked_{key}", masked,
                 {"masked": True, "crop_bbox": cb}, origin=(cb[0], cb[1]))

    # --- Exp 5: crop strips + comparison ---
    strip_keys = ["pad-10", "pad-5", "pad+0", "pad+5", "pad+10", "pad+20",
                  "pad+50"]
    strip = [crops_saved[k] for k in strip_keys if k in crops_saved]
    widths = sum(c.width for c in strip) + 20 * (len(strip) - 1)
    canvas = Image.new("RGB", (widths, max(c.height for c in strip) + 24),
                       "white")
    dd = ImageDraw.Draw(canvas)
    x = 0
    for k, c in zip(strip_keys, strip):
        canvas.paste(c, (x, 24))
        dd.text((x + 4, 4), k, fill=(0, 0, 0))
        x += c.width + 20
    canvas.save(out_dir / "blog_padding_comparison.png")
    for key in ["pad+0", "pad+10"]:
        crops_saved[key].save(out_dir / f"blog_crop_{key[1:]}.png")
    crops_saved["pad+50"].save(out_dir / "blog_crop_50.png")
    crops_saved["pad+20"].save(out_dir / "blog_crop_20.png")

    # --- Exp 3: canonical crops table ---
    print("\nCrop      Text" + " " * 24 + "Conf       IoU       PixelSupport")
    print("-" * 59)
    for k in ["pad+0", "pad+5", "pad+10", "pad+20", "pad+50"]:
        r = rec.get(k, {})
        print(f"{k:9s} {(','.join(r.get('texts', []))[:28]):30s} "
              f"{str(r.get('best_iou', 0))[:8]:10s} "
              f"{r.get('pixel_support', 0)}")

    # --- preprocessing comparison strip (rebuilt variants) ---
    comp_rows = []
    base0 = crops_saved["pad+0"]
    variants = {
        "orig": base0,
        "gray": base0.convert("L").convert("RGB"),
    }
    from PIL import ImageOps, ImageFilter
    variants["contrast"] = ImageOps.autocontrast(
        base0.convert("L")).convert("RGB")
    variants["sharpen"] = base0.filter(ImageFilter.SHARPEN)
    variants["2x"] = base0.resize((base0.width * 2, base0.height * 2),
                                  Image.LANCZOS)
    variants["4x"] = base0.resize((base0.width * 4, base0.height * 4),
                                  Image.LANCZOS)
    comp_rows = [(k, v) for k, v in variants.items()]
    W = sum(v.width for _, v in comp_rows) + 15 * (len(comp_rows) - 1)
    H = max(v.height for _, v in comp_rows) + 24
    canvas = Image.new("RGB", (W, H), "white")
    dd = ImageDraw.Draw(canvas)
    x = 0
    for k, v in comp_rows:
        canvas.paste(v, (x, 24))
        dd.text((x + 4, 4), k, fill=(0, 0, 0))
        x += v.width + 15
    canvas.save(out_dir / "blog_preprocessing_comparison.png")
    # mask comparison: original +10/+20 beside masked versions
    mrows = []
    for key in ["+10", "+20"]:
        mrows.append((f"orig{key}", crops_saved[f"pad{key}"]))
        gray = Image.new("RGB", crops_saved[f"pad{key}"].size, (128, 128, 128))
        cb = rec[f"pad{key}"]["crop_bbox"]
        ox, oy = cb[0], cb[1]
        gray.paste(mask_context(crops_saved[f"pad{key}"], cb), (0, 0))
        mrows.append((f"masked{key}", gray))
    W = sum(v.width for _, v in mrows) + 15 * (len(mrows) - 1)
    H = max(v.height for _, v in mrows) + 24
    canvas = Image.new("RGB", (W, H), "white")
    dd = ImageDraw.Draw(canvas)
    x = 0
    for k, v in mrows:
        canvas.paste(v, (x, 24))
        dd.text((x + 4, 4), k, fill=(0, 0, 0))
        x += v.width + 15
    canvas.save(out_dir / "blog_context_mask_comparison.png")

    # --- Exp 7: multi-crop consensus over -5/0/+5/+10 ---
    print("\nMULTI-CROP CONSENSUS")
    print("--------------------")
    votes = {}
    for k in ["pad-5", "pad+0", "pad+5", "pad+10"]:
        for t in rec.get(k, {}).get("norm", []):
            votes[t] = votes.get(t, 0) + 1
    for k in ["pad-5", "pad+0", "pad+5", "pad+10"]:
        print(f"{k:9s} -> {rec.get(k, {}).get('texts', [])}")
    consensus = max(votes, key=votes.get) if votes else "(none)"
    print(f"\nConsensus: {consensus!r} "
          f"({votes.get(consensus, 0)}/4 crops)")

    # --- Exp 8: size-stratified regions (precomputed [imgkey, box] pairs,
    # since YOLO weights live in the main env, not the Surya venv) ---
    print("\nSIZE-STRATIFIED CHECK")
    print("---------------------")
    from PIL import Image as _I
    _IMG_SRC = {
        "web_cmyhc5": str(BLOG_IMG),
        "product": "/tmp/strat_product.png",
        "minishop": "/tmp/strat_minishop.png",
        "canvas": "/tmp/strat_canvas.png",
        "testpage": "/tmp/testpage.png",
    }
    strat_src = Path(args.strat_boxes) if args.strat_boxes else None
    det_boxes = []
    if strat_src is not None and strat_src.exists():
        _strat = json.load(open(strat_src))
        _imgs = {}
        for _cat in ("tiny", "medium", "normal"):
            for _key, _box in _strat.get(_cat, []):
                if _key not in _imgs:
                    _imgs[_key] = _I.open(_IMG_SRC[_key]).convert("RGB")
                det_boxes.append((_imgs[_key], _box, _cat))
    else:
        print("  (no --strat-boxes file; skipping stratified runs)")
    strat = [("tiny", [t for t in det_boxes if t[2] == "tiny"]),
               ("medium", [t for t in det_boxes if t[2] == "medium"]),
               ("normal", [t for t in det_boxes if t[2] == "normal"])]
    strat_rows = []
    for cname, boxes in strat:
        for bi, (im, bb, _cat) in enumerate(boxes):
            outs = {}
            for pname, ppad in (("exact", 0.0), ("p5", 0.05),
                                ("p10", 0.10), ("p20", 0.20)):
                pb = pad_box_frac(bb, ppad, img_w=im.width, img_h=im.height)
                cx1, cy1, cx2, cy2 = (int(round(v)) for v in pb)
                rname = f"strat_{cname}{bi}_{pname}"
                run_case(rname, im.crop((cx1, cy1, cx2, cy2)),
                         {"strat": cname}, origin=(cx1, cy1))
                outs[pname] = rec[rname]["texts"]
            strat_rows.append((cname, bi, [outs[k] for k in
                                           ("exact", "p5", "p10", "p20")]))
    for cname, bi, texts in strat_rows:
        stable = len({tuple(t) for t in texts}) == 1
        print(f"  {cname:6s}#{bi}: {texts} "
              f"{'STABLE' if stable else 'VARIES'}")

    json.dump(rec, open(out_dir / "results.json", "w"), indent=1,
              default=str)
    # summary.csv across all runs
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "texts", "best_iou", "pixel_support", "secs",
                    "crop_w", "crop_h"])
        for name, r in rec.items():
            if not isinstance(r, dict):
                continue
            cs = r.get("crop_size", ["", ""])
            w.writerow([name, "|".join(r.get("texts", [])),
                        r.get("best_iou", ""), r.get("pixel_support", ""),
                        r.get("secs", ""), cs[0] if cs else "",
                        cs[1] if len(cs) > 1 else ""])
    print(f"\nresults: {out_dir / 'results.json'}")
    print(f"summary: {out_dir / 'summary.csv'}")


def mask_context(crop, crop_bbox):
    """Gray-out everything except the GT box mapped into this crop."""
    from PIL import Image, ImageDraw
    gb = [869, 9, 964, 60]
    ox, oy = crop_bbox[0], crop_bbox[1]
    local = [gb[0] - ox, gb[1] - oy, gb[2] - ox, gb[3] - oy]
    gray = Image.new("RGB", crop.size, (128, 128, 128))
    gray.paste(crop.crop((int(max(0, local[0])), int(max(0, local[1])),
                          int(min(crop.width, local[2])),
                          int(min(crop.height, local[3])))),
               (int(max(0, local[0])), int(max(0, local[1]))))
    return gray


if __name__ == "__main__":
    main()
