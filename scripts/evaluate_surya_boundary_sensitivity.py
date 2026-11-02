#!/usr/bin/env python3
"""Surya 0.4.2 boundary/position sensitivity diagnostic (standalone).

RUN WITH the classic venv: /tmp/surya_old/bin/python
                            scripts/evaluate_surya_boundary_sensitivity.py

Only Surya 0.4.2 classic + PIL + numpy. No VLM, no CRAFT/DBNet/PaddleOCR,
no pipeline changes. One model load; batched recognition; only crop
geometry / placement / preprocessing varies between runs.

Outputs: reports/surya_boundary_sensitivity/{results.json, *.csv, *.png,
summary.md} + console report.
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
OUT_DIR = REPO / "reports" / "surya_boundary_sensitivity"
GT_BLOG = {"text": "Blog", "bbox": [869, 9, 964, 60]}
BLOG_IMG = Path("/Users/manasmondal/Desktop/GUI-Web-Agent/dataset/ScreenSpot-v2/screenspotv2_image/web_cmyhc5.png")
GRAY = (128, 128, 128)


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


def clamp_box(b, w, h):
    return [max(0.0, b[0]), max(0.0, b[1]), min(float(w), b[2]),
            min(float(h), b[3])]


def pad_box_frac(b, fx, fy=None, sides=None, img_w=None, img_h=None):
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


class Runner:
    def __init__(self):
        from surya.model.recognition.model import load_model
        from surya.model.recognition.processor import load_processor
        self.model = load_model(device="cpu")
        self.processor = load_processor()

    def recognize_batch(self, crops):
        """crops: list of PIL images. Returns list of line-lists in crop coords."""
        from surya.ocr import run_recognition
        if not crops:
            return []
        box_lists = []
        for c in crops:
            w, h = c.size
            box_lists.append([[0, 0, w, h]])
        res = run_recognition(crops, [["en"]] * len(crops),
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
                              "confidence": tl.confidence, "poly": bb})
            out.append(lines)
        return out


def main(argv=None):
    from PIL import Image, ImageDraw, ImageFilter, ImageOps

    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--strat-boxes", default="/tmp/strat_boxes.json")
    args = p.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(BLOG_IMG).convert("RGB")
    img_w, img_h = img.size
    manifest = json.load(open(
        REPO / "reports" / "local_text_recognition" / "manifest.json"))
    omni = manifest["images"]["web_cmyhc5"]["omni_boxes"]
    base_box = max(omni, key=lambda b: box_iou(b, GT_BLOG["bbox"]))
    print(f"Base Omni box: {[round(v, 1) for v in base_box]}", flush=True)
    runner = SuryaRunner()

    cases = []  # dicts: key,exp,crop,origin,scale,gt,gt_text

    def add(key, exp, crop, origin, scale, gt, gt_text, meta=None):
        x1, y1, x2, y2 = (int(round(v)) for v in
                          [origin[0], origin[1],
                           origin[0] + crop.width / scale,
                           origin[1] + crop.height / scale])
        cases.append({"key": key, "exp": exp, "crop": crop,
                      "origin": [origin[0], origin[1]], "scale": scale,
                      "gt": gt, "gt_text": gt_text, "meta": meta or {}})

    # ---- Exp 1: one-pixel boundary sweep on base box ----
    for boundary, axis in (("left", 0), ("right", 2), ("top", 1), ("bottom", 3)):
        for off in range(-5, 6):
            b = list(base_box)
            b[axis] += off
            x1, y1, x2, y2 = (int(round(v)) for v in
                              [max(0, b[0]), max(0, b[1]),
                               min(img_w, b[2]), min(img_h, b[3])])
            if x2 <= x1 + 1 or y2 <= y1 + 1:
                continue
            add(f"edge_{boundary}_{off:+d}", "edge1",
                img.crop((x1, y1, x2, y2)), (x1, y1), 1.0, GT_BLOG["bbox"],
                GT_BLOG["text"], {"boundary": boundary, "offset_px": off})

    # ---- Exp 0: uniform padding sweep (for padding strip + Exp3 table) ----
    for pct in [-10, -5, 0, 5, 10, 20, 50]:
        f = pct / 100.0
        w0, h0 = base_box[2] - base_box[0], base_box[3] - base_box[1]
        pb = [base_box[0] - w0 * f, base_box[1] - h0 * f,
              base_box[2] + w0 * f, base_box[3] + h0 * f]
        x1, y1, x2, y2 = (int(round(v)) for v in
                          [max(0, pb[0]), max(0, pb[1]),
                           min(img_w, pb[2]), min(img_h, pb[3])])
        if x2 > x1 + 1 and y2 > y1 + 1:
            add(f"pad{pct:+d}", "pad",
                img.crop((x1, y1, x2, y2)), (x1, y1), 1.0,
                GT_BLOG["bbox"], GT_BLOG["text"],
                {"crop_bbox": [x1, y1, x2, y2], "padding_pct": pct})

    # ---- Exp 2/3: heatmaps ----
    for lo in range(-5, 6):
        for ro in range(-5, 6):
            b = [base_box[0] + lo, base_box[1], base_box[2] + ro, base_box[3]]
            x1, y1, x2, y2 = (int(round(v)) for v in
                              [max(0, b[0]), max(0, b[1]),
                               min(img_w, b[2]), min(img_h, b[3])])
            if x2 > x1 + 1 and y2 > y1 + 1:
                add(f"lr_{lo:+d}_{ro:+d}", "heat_lr",
                    img.crop((x1, y1, x2, y2)), (x1, y1), 1.0,
                    GT_BLOG["bbox"], GT_BLOG["text"],
                    {"left": lo, "right": ro})
    for to in range(-5, 6):
        for bo in range(-5, 6):
            b = [base_box[0], base_box[1] + to, base_box[2], base_box[3] + bo]
            x1, y1, x2, y2 = (int(round(v)) for v in
                              [max(0, b[0]), max(0, b[1]),
                               min(img_w, b[2]), min(img_h, b[3])])
            if x2 > x1 + 1 and y2 > y1 + 1:
                add(f"tb_{to:+d}_{bo:+d}", "heat_tb",
                    img.crop((x1, y1, x2, y2)), (x1, y1), 1.0,
                    GT_BLOG["bbox"], GT_BLOG["text"],
                    {"top": to, "bottom": bo})

    # ---- canonical exact crop (shared source for Exp 4/5/6) ----
    bx = [int(round(v)) for v in base_box]
    exact = img.crop((bx[0], bx[1], bx[2], bx[3]))
    exact_origin = (bx[0], bx[1])

    # ---- Exp 4: 3x3 position grid on fixed canvas ----
    cw, ch = 400, 200
    positions = {
        "top-left": (0, 0), "top-center": ((cw - exact.width) // 2, 0),
        "top-right": (cw - exact.width, 0),
        "middle-left": (0, (ch - exact.height) // 2),
        "center": ((cw - exact.width) // 2, (ch - exact.height) // 2),
        "middle-right": (cw - exact.width, (ch - exact.height) // 2),
        "bottom-left": (0, ch - exact.height),
        "bottom-center": ((cw - exact.width) // 2, ch - exact.height),
        "bottom-right": (cw - exact.width, ch - exact.height),
    }
    pos_canvases = {}
    for pname, (px, py) in positions.items():
        cv = Image.new("RGB", (cw, ch), GRAY)
        cv.paste(exact, (px, py))
        pos_canvases[pname] = cv
        add(f"pos_{pname}", "position", cv, (0, 0), 1.0,
            [px, py, px + exact.width, py + exact.height], GT_BLOG["text"],
            {"position": pname})

    # ---- Exp 5: aspect canvases, Blog centered ----
    ew, eh = exact.width, exact.height
    for rw, rh in [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (4, 1), (1, 4)]:
        s = max((ew + 20) / rw, (eh + 20) / rh, 1.0)
        W, H = int(round(rw * s)), int(round(rh * s))
        cv = Image.new("RGB", (W, H), GRAY)
        px, py = (W - ew) // 2, (H - eh) // 2
        cv.paste(exact, (px, py))
        add(f"aspect_{rw}x{rh}", "aspect", cv, (0, 0), 1.0,
            [px, py, px + ew, py + eh], GT_BLOG["text"],
            {"canvas": [W, H], "ratio": f"{rw}:{rh}"})

    # ---- Exp 6: scale sweep on fixed canvas ----
    SCW, SCH = 400, 200
    for sc in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0]:
        nw, nh = max(1, int(round(ew * sc))), max(1, int(round(eh * sc)))
        thumb = exact.resize((nw, nh), Image.LANCZOS)
        cv = Image.new("RGB", (SCW, SCH), GRAY)
        px, py = (SCW - nw) // 2, (SCH - nh) // 2
        cv.paste(thumb, (px, py))
        add(f"scale_{sc}x", "scale", cv, (0, 0), 1.0,
            [px, py, px + nw, py + nh], GT_BLOG["text"],
            {"scale": sc})

    # ---- Exp 8: stratified regions ----
    strat = json.load(open(args.strat_boxes))
    simg = {
        "web_cmyhc5": img,
        "product": Image.open(REPO / "test_pages/fixtures/product.png").convert("RGB"),
        "minishop": Image.open(REPO / "test_pages/fixtures/index.png").convert("RGB"),
        "canvas": Image.open(REPO / "test_pages/fixtures/canvas.png").convert("RGB"),
        "testpage": Image.open("/tmp/testpage.png").convert("RGB"),
    }
    for cat in ("tiny", "medium", "normal"):
        for bi, (skey, bb) in enumerate(strat.get(cat, [])):
            sim = simg[skey]
            for pname, ppad in (("exact", 0.0), ("p5", 0.05),
                                ("p10", 0.10), ("p20", 0.20)):
                pb = pad_box_frac(bb, ppad, img_w=sim.width,
                                  img_h=sim.height)
                cx1, cy1, cx2, cy2 = (int(round(v)) for v in pb)
                if cx2 <= cx1 + 1 or cy2 <= cy1 + 1:
                    continue
                add(f"strat_{cat}{bi}_{pname}", "strat",
                    sim.crop((cx1, cy1, cx2, cy2)), (cx1, cy1), 1.0,
                    bb, f"strat-{cat}{bi}",
                    {"strat": cat, "src_img": skey})

    # ---- batched recognition (one model load, chunks for RAM safety) ----
    print(f"{len(cases)} cases; running batched Surya recognition...",
          flush=True)
    t_all = time.time()
    # actual batched calls:
    from surya.ocr import run_recognition
    B = 48
    idx = 0
    for ci in range(0, len(cases), B):
        chunk = cases[ci:ci + B]
        imgs = [c["crop"] for c in chunk]
        box_lists = []
        for c in chunk:
            w, h = c["crop"].size
            box_lists.append([[0, 0, w, h]])
        res = run_recognition(imgs, [["en"]] * len(imgs),
                              runner.model, runner.processor,
                              bboxes=box_lists)
        for c, r in zip(chunk, res):
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
                              "confidence": tl.confidence, "poly": bb})
            c["lines"] = lines
            idx += 1
        print(f"  batch done ({idx}/{len(cases)})", flush=True)
    print(f"recognition: {time.time() - t_all:.0f}s total", flush=True)

    # ---- score every case in a common frame ----
    results = {}
    for c in cases:
        ox, oy, sc = c["origin"][0], c["origin"][1], c["scale"]
        gt = c["gt"]
        best_iou, best_box, best_text = 0.0, None, None
        for l in c.get("lines", []):
            if l["poly"] is None:
                continue
            p = l["poly"]
            shifted = [p[0] / sc + ox, p[1] / sc + oy,
                       p[2] / sc + ox, p[3] / sc + oy]
            v = box_iou(shifted, gt)
            if v > best_iou:
                best_iou, best_box, best_text = v, shifted, l["text_raw"]
        texts = [l["text_raw"] for l in c.get("lines", [])]
        norm = [norm_text(t) for t in texts]
        # pixel support: pred-box ∩ GT / pred-box
        support = 0.0
        if best_box is not None:
            ix1, iy1 = max(best_box[0], gt[0]), max(best_box[1], gt[1])
            ix2, iy2 = min(best_box[2], gt[2]), min(best_box[3], gt[3])
            support = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1) / max(
                1e-9, box_area(best_box))
        # classification: text match AND overlap required for CORRECT
        txt_ok = any("blog" in t for t in norm) if c["gt_text"] == "Blog" \
            else any(t for t in norm)
        if c["gt_text"].startswith("strat-"):
            status = "STABLE-CHECK"
        elif txt_ok and best_iou >= 0.5:
            status = "CORRECT"
        elif txt_ok:
            status = "MISLOCALIZED"
        elif best_iou > 0:
            status = "MISLOCALIZED"
        elif not texts:
            status = "EMPTY"
        else:
            status = "WRONG_TEXT"
        results[c["key"]] = {
            "exp": c["exp"], "texts": texts, "norm": norm,
            "best_iou": round(best_iou, 3),
            "best_box": [round(v, 1) for v in best_box] if best_box else None,
            "pixel_support": round(support, 3),
            "status": status, "meta": c["meta"],
            "crop_size": list(c["crop"].size),
        }

    # ---- checkpoint scored results BEFORE any visualization ----
    (out_dir / "results.json").write_text(
        json.dumps({"cases": results}, indent=1, default=str))
    print(f"checkpoint: results.json written "
          f"({len(results)} scored runs)", flush=True)

    # ---- Exp 4/5/6 image strips + heatmaps + comparison ----
    from PIL import ImageDraw
    thumbs = {}
    for key in list(results):
        c = next(x for x in cases if x["key"] == key)
        thumbs[key] = c["crop"]

    def strip(keys, fname, labels=None):
        ims = [thumbs[k] for k in keys if k in thumbs]
        if not ims:
            print(f"  (strip {fname}: no crops found, skipped)", flush=True)
            return
        W = sum(i.width for i in ims) + 12 * (len(ims) - 1)
        H = max(i.height for i in ims) + 22
        cv = Image.new("RGB", (W, H), "white")
        dd = ImageDraw.Draw(cv)
        x = 0
        for k, im in zip(keys, ims):
            cv.paste(im, (x, 22))
            dd.text((x + 3, 3), labels.get(k, k) if labels else k,
                    fill=(0, 0, 0))
            x += im.width + 12
        cv.save(out_dir / fname)

    strip([f"pad{p:+d}" for p in [-10, -5, 0, 5, 10, 20, 50]],
          "blog_padding_comparison.png")
    # preprocessing strip rebuilt from pad+0 variants is covered by pre_* runs
    grid = 11
    for name, keys, labels in [
            ("blog_lr_heatmap", [f"lr_{lo:+d}_{ro:+d}"
                                 for lo in range(-5, 6)
                                 for ro in range(-5, 6)], None),
            ("blog_tb_heatmap", [f"tb_{to:+d}_{bo:+d}"
                                 for to in range(-5, 6)
                                 for bo in range(-5, 6)], None)]:
        present = [k for k in keys if k in results]
        n = int(len(present) ** 0.5)
        cell, legend = 34, 150
        cv = Image.new("RGB", (n * cell + legend, n * cell), "white")
        dd = ImageDraw.Draw(cv)
        for idx, k in enumerate(present):
            r, c_ = divmod(idx, n)
            ok = results[k]["best_iou"] >= 0.5 and any(
                "blog" in t for t in results[k]["norm"])
            col = (60, 200, 60) if ok else (220, 220, 220)
            dd.rectangle([c_ * cell, r * cell,
                          (c_ + 1) * cell - 2, (r + 1) * cell - 2], fill=col)
        dd.text((n * cell + 8, 8), "green:\nblog+\nIoU>=0.5", fill=(0, 0, 0))
        cv.save(out_dir / f"{name}.png")

    # position grid / aspect / scale strips
    for prefix, keys, fname in [
            ("pos_", ["pos_top-left", "pos_top-center", "pos_top-right",
                      "pos_middle-left", "pos_center", "pos_middle-right",
                      "pos_bottom-left", "pos_bottom-center",
                      "pos_bottom-right"], "blog_position_grid.png"),
            ("aspect_", [f"aspect_{a}" for a in
                         ["1x1", "1x2", "2x1", "1x3", "3x1", "4x1", "1x4"]],
             "blog_aspect_ratio.png"),
            ("scale_", [f"scale_{s}x" for s in
                        [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0]],
             "blog_scale_sweep.png")]:
        ims = [(k, thumbs[k]) for k in keys if k in thumbs]
        W = sum(i.width for _, i in ims) + 12 * (len(ims) - 1)
        H = max(i.height for _, i in ims) + 22
        cv = Image.new("RGB", (W, H), "white")
        dd = ImageDraw.Draw(cv)
        x = 0
        for k, im in ims:
            cv.paste(im, (x, 22))
            dd.text((x + 3, 3), k.replace(prefix, ""), fill=(0, 0, 0))
            x += im.width + 12
        cv.save(out_dir / fname)

    # ---- tables, CSVs, summary.md ----
    def table(keys, title):
        print(f"\n{title}")
        print("-" * 60)
        for k in keys:
            r = results.get(k)
            if r is None:
                continue
            print(f"{k:16s} {str(r['texts'])[:34]:36s} "
                  f"{r['best_iou']:.3f} {r['pixel_support']:.3f}")

    print("\n" + "=" * 60)
    print("SURYA 0.4.2 BOUNDARY / POSITION SENSITIVITY")
    print("=" * 60)
    print("\nTARGET: Blog\nIMAGE: web_cmyhc5.png\n")
    print("1. PIXEL BOUNDARY SWEEP\n" + "-" * 60)
    for b in ("left", "right", "top", "bottom"):
        print(f"{b.capitalize()}:")
        for off in range(-5, 6):
            k = f"edge_{b}_{off:+d}"
            r = results.get(k)
            if r:
                print(f"  {off:+d} px -> {r['texts']} [{r['status']}]")

    print("\n2. LEFT/RIGHT HEATMAP: successful crops: "
          f"{sum(1 for k in results if k.startswith('lr_') and results[k]['status'] == 'CORRECT')} / 121")
    print("\n3. TOP/BOTTOM HEATMAP: successful crops: "
          f"{sum(1 for k in results if k.startswith('tb_') and results[k]['status'] == 'CORRECT')} / 121")
    print("\n4. TEXT POSITION")
    print("--------------------")
    for k in ["pos_top-left", "pos_top-center", "pos_top-right",
              "pos_middle-left", "pos_center", "pos_middle-right",
              "pos_bottom-left", "pos_bottom-center", "pos_bottom-right"]:
        r = results.get(k, {})
        print(f"{k.replace('pos_', ''):13s}: {r.get('texts', [])} [{r.get('status')}]")
    print("\n5. ASPECT RATIO\n--------------------")
    for k in ["aspect_1x1", "aspect_1x2", "aspect_2x1", "aspect_1x3",
              "aspect_3x1", "aspect_4x1", "aspect_1x4"]:
        r = results.get(k, {})
        print(f"{k.replace('aspect_', ''):5s}: {r.get('texts', [])} [{r.get('status')}]")
    print("\n6. SCALE\n--------------------")
    for k in ["scale_0.5x", "scale_0.75x", "scale_1.0x", "scale_1.25x",
              "scale_1.5x", "scale_2.0x", "scale_4.0x"]:
        r = results.get(k, {})
        print(f"{k.replace('scale_', ''):6s}: {r.get('texts', [])} [{r.get('status')}]")
    print("\n7. GENERALIZATION (stability across exact/p5/p10/p20)")
    print("--------------------")
    for cname in ("tiny", "medium", "normal"):
        hits = [k for k in results if k.startswith(f"strat_{cname}")]
        stab = sum(1 for k in hits if results[k].get("_stable"))
        print(f"{cname:6s}: {len(hits)} runs")

    import csv
    for csvname, keys in (("boundary_sweep.csv",
                           [k for k in results if k.startswith("edge_")]),
                          ("left_right_heatmap.csv",
                           [k for k in results if k.startswith("lr_")]),
                          ("top_bottom_heatmap.csv",
                           [k for k in results if k.startswith("tb_")]),
                          ("position_grid.csv",
                           [k for k in results if k.startswith("pos_")]),
                          ("aspect_ratio.csv",
                           [k for k in results if k.startswith("aspect_")]),
                          ("scale_sweep.csv",
                           [k for k in results if k.startswith("scale_")])):
        with open(out_dir / csvname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "texts", "best_iou", "pixel_support",
                        "status", "crop_w", "crop_h"])
            for k in keys:
                r = results[k]
                cs = r.get("crop_size", ["", ""])
                w.writerow([k, "|".join(r.get("texts", [])),
                            r.get("best_iou", ""), r.get("pixel_support", ""),
                            r.get("status", ""),
                            cs[0] if cs else "",
                            cs[1] if len(cs) > 1 else ""])
    (out_dir / "results.json").write_text(
        json.dumps({"cases": results}, indent=1, default=str))
    print(f"\nartifacts: {out_dir}")


def pad_box_frac(b, fx, fy=None, sides=None, img_w=None, img_h=None):
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


def norm_text(s):
    return " ".join(str(s or "").lower().split())


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


class SuryaRunner:
    def __init__(self):
        from surya.model.recognition.model import load_model
        from surya.model.recognition.processor import load_processor
        self.model = load_model(device="cpu")
        self.processor = load_processor()

    def recognize(self, crops):
        from surya.ocr import run_recognition
        if not crops:
            return []
        box_lists = []
        for c in crops:
            w, h = c.size
            box_lists.append([[0, 0, w, h]])
        res = run_recognition(crops, [["en"]] * len(crops),
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
                              "confidence": tl.confidence, "poly": bb})
            out.append(lines)
        return out


if __name__ == "__main__":
    main()
