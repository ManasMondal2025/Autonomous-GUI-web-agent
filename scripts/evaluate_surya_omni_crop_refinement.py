#!/usr/bin/env python3
"""Surya 0.4.2 omni-crop refinement diagnostic (standalone).

RUN WITH: /tmp/surya_old/bin/python scripts/evaluate_surya_omni_crop_refinement.py

MISSION: determine whether a lightweight, deterministic refinement of the
OmniParser bbox can recover the 3 GT-crop-rescuable failures (s01, s05, s09)
and improve Surya recognition/localization.

Test ONLY bbox refinement. Surya 0.4.2 classic weights, configuration, and
evaluation are completely fixed. No VLM, no CRAFT/DBNet++, no OCR voting,
no ensemble, no pipeline changes.

For each of the 12 ScreenSpot-mini samples, starting from the original
OmniParser bbox (max-overlap box from /tmp/omni12.json, same rule as the
failure-taxonomy experiment), evaluate these candidate crops:

  OMNI_RAW               original OmniParser bbox
  OMNI_TIGHT             strip uniform empty margins (pixel/edge projection)
  OMNI_CONTENT           smallest rect of non-background pixels (robust thresh)
  OMNI_HORIZONTAL_TIGHT  tighten left/right only
  OMNI_VERTICAL_TIGHT    tighten top/bottom only
  OMNI_EXPAND_TIGHT_2/5/10 = OMNI_TIGHT + fixed 2%/5%/10% margin per side

Refinements use numpy/PIL pixel analysis only. No OCR decides any crop.
No learned model. No ground-truth information (refinement never sees GT).

Outputs: reports/surya_omni_crop_refinement/{results.json, per_sample.csv,
summary.csv, rescued_cases.csv, summary.md, sample_*_*.png, contact_*.png}
+ console report.
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.ERROR)

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "reports" / "surya_omni_crop_refinement"
DATA = REPO / "data" / "screenspot" / "v2_mini.jsonl"

STOPWORDS = {"check", "the", "a", "an", "of", "in", "on", "to", "for",
             "tap", "open", "view", "look", "into", "first"}

METHODS = ["OMNI_RAW", "OMNI_TIGHT", "OMNI_CONTENT",
           "OMNI_HORIZONTAL_TIGHT", "OMNI_VERTICAL_TIGHT",
           "OMNI_EXPAND_TIGHT_2", "OMNI_EXPAND_TIGHT_5",
           "OMNI_EXPAND_TIGHT_10"]

TIGHT_THRESH = 12
TIGHT_MIN_FRAC = 0.005
TIGHT_MIN_PX = 2
MIN_W = 8
MIN_H = 8
MIN_AREA_FRAC = 0.10


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


def expected_tokens(instruction):
    return [w for w in norm_text(instruction).replace("'", "").split()
            if w not in STOPWORDS and len(w) > 1]


def estimate_bg(gray):
    """Background = median of the 2px border. Pure numpy, deterministic."""
    import numpy as np
    h, w = gray.shape
    b = min(2, h // 2, w // 2)
    border = np.concatenate([gray[:b, :].ravel(), gray[-b:, :].ravel(),
                             gray[:, :b].ravel(), gray[:, -b:].ravel()])
    return float(np.median(border))


def fg_mask(gray, bg, thresh):
    import numpy as np
    return np.abs(gray.astype(float) - bg) > thresh


def projection_tight_box(gray, thresh=TIGHT_THRESH,
                         min_frac=TIGHT_MIN_FRAC, min_px=TIGHT_MIN_PX,
                         tighten_h=True, tighten_v=True):
    """First/last row/col whose foreground fraction exceeds the floor.

    Returns crop-relative [x1, y1, x2, y2] (ints) or None if degenerate.
    """
    import numpy as np
    h, w = gray.shape
    bg = estimate_bg(gray)
    m = fg_mask(gray, bg, thresh)
    col_need = max(min_px, min_frac * h)
    row_need = max(min_px, min_frac * w)
    cols = m.sum(axis=0)
    rows = m.sum(axis=1)
    if tighten_h:
        keep_x = [x for x in range(w) if cols[x] >= col_need]
        if not keep_x:
            return None
        x1, x2 = keep_x[0], keep_x[-1] + 1
    else:
        x1, x2 = 0, w
    if tighten_v:
        keep_y = [y for y in range(h) if rows[y] >= row_need]
        if not keep_y:
            return None
        y1, y2 = keep_y[0], keep_y[-1] + 1
    else:
        y1, y2 = 0, h
    if x2 - x1 < MIN_W or y2 - y1 < MIN_H:
        return None
    return [x1, y1, x2, y2]


def content_box(gray):
    """Smallest rect of non-background pixels, robust to antialiasing.

    Threshold adapts to image noise via MAD; leading/trailing rows/cols
    are trimmed by cumulative support (0.5% tails ignored) so isolated
    shadow/antialias pixels cannot force extreme boxes. +1px pad keeps
    glyph edges. Returns crop-relative box or None if degenerate.
    """
    import numpy as np
    h, w = gray.shape
    g = gray.astype(float)
    med = float(np.median(g))
    mad = float(np.median(np.abs(g - med)))
    thresh = max(15.0, 3.0 * mad if mad > 0 else 15.0)
    bg = estimate_bg(gray)
    m = fg_mask(gray, bg, thresh)
    total = int(m.sum())
    if total < 8:
        return None
    rows = m.sum(axis=1).astype(float)
    cols = m.sum(axis=0).astype(float)
    tail = 0.005 * total
    y1 = 0
    c = 0.0
    for y in range(h):
        if c + rows[y] >= tail:
            y1 = y
            break
        c += rows[y]
    y2 = h
    c = 0.0
    for y in range(h - 1, -1, -1):
        if c + rows[y] >= tail:
            y2 = y + 1
            break
        c += rows[y]
    x1 = 0
    c = 0.0
    for x in range(w):
        if c + cols[x] >= tail:
            x1 = x
            break
        c += cols[x]
    x2 = w
    c = 0.0
    for x in range(w - 1, -1, -1):
        if c + cols[x] >= tail:
            x2 = x + 1
            break
        c += cols[x]
    x1, y1 = max(0, x1 - 1), max(0, y1 - 1)
    x2, y2 = min(w, x2 + 1), min(h, y2 + 1)
    if x2 - x1 < MIN_W or y2 - y1 < MIN_H:
        return None
    return [x1, y1, x2, y2]


def expand_box_rel(box, frac, w, h):
    """Expand a crop-relative box by frac of its own size per side."""
    bw, bh = box[2] - box[0], box[3] - box[1]
    return [max(0, int(round(box[0] - bw * frac))),
            max(0, int(round(box[1] - bh * frac))),
            min(w, int(round(box[2] + bw * frac))),
            min(h, int(round(box[3] + bh * frac)))]


def refine_crop(gray, method):
    """Return (crop-relative box, fallback_flag). Never uses OCR/GT."""
    h, w = gray.shape
    full = [0, 0, w, h]
    if method == "OMNI_RAW":
        return full, False
    if method == "OMNI_TIGHT":
        b = projection_tight_box(gray, tighten_h=True, tighten_v=True)
    elif method == "OMNI_CONTENT":
        b = content_box(gray)
    elif method == "OMNI_HORIZONTAL_TIGHT":
        b = projection_tight_box(gray, tighten_h=True, tighten_v=False)
    elif method == "OMNI_VERTICAL_TIGHT":
        b = projection_tight_box(gray, tighten_h=False, tighten_v=True)
    elif method.startswith("OMNI_EXPAND_TIGHT_"):
        frac = int(method.rsplit("_", 1)[1]) / 100.0
        b = projection_tight_box(gray, tighten_h=True, tighten_v=True)
        if b is not None:
            b = expand_box_rel(b, frac, w, h)
    else:
        raise ValueError(method)
    if b is None:
        return full, True
    if box_area(b) < MIN_AREA_FRAC * box_area(full):
        return full, True
    return b, False


def main(argv=None):
    import numpy as np
    from PIL import Image, ImageDraw

    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--omni-boxes", default="/tmp/omni12.json")
    p.add_argument("--batch", type=int, default=24)
    args = p.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from surya.model.recognition.model import load_model
    from surya.model.recognition.processor import load_processor
    from surya.ocr import run_recognition

    samples = [json.loads(l) for l in open(DATA)]
    omni_all = json.load(open(args.omni_boxes))
    print("Loading Surya rec model...", flush=True)
    model = load_model(device="cpu")
    processor = load_processor()

    # ---- build candidate crops: 12 samples x 8 methods = 96 jobs ----
    imgs = {}
    omni_boxes = {}   # per-sample selected omni box (global, clipped)
    jobs = []         # (key, PIL crop, global origin)
    job_meta = {}     # key -> {sample, method, crop_bbox, crop_size,
                      #          pixel_support, fallback}
    for i, s in enumerate(samples):
        img = Image.open(s["img_path"]).convert("RGB")
        imgs[i] = img
        W, H = img.size
        gt = s["bbox"]
        boxes = omni_all[str(i)]
        obs = [[e["bbox"]["x"], e["bbox"]["y"],
                e["bbox"]["x"] + e["bbox"]["width"],
                e["bbox"]["y"] + e["bbox"]["height"]] for e in boxes]
        ob = max(obs, key=lambda b: box_iou(b, gt))
        ob = [max(0.0, ob[0]), max(0.0, ob[1]),
              min(float(W), ob[2]), min(float(H), ob[3])]
        omni_boxes[i] = ob
        ox1, oy1, ox2, oy2 = (int(round(v)) for v in ob)
        omni_crop = img.crop((ox1, oy1, ox2, oy2))
        gray = np.array(omni_crop.convert("L"))
        bg = estimate_bg(gray)
        support_full = int((fg_mask(gray, bg, TIGHT_THRESH)).sum())
        for method in METHODS:
            rel, fallback = refine_crop(gray, method)
            gx1, gy1 = ox1 + rel[0], oy1 + rel[1]
            gx2, gy2 = ox1 + rel[2], oy1 + rel[3]
            crop = img.crop((gx1, gy1, gx2, gy2))
            cg = np.array(crop.convert("L"))
            cb = estimate_bg(cg)
            support = (support_full if (method == "OMNI_RAW" or fallback)
                       else int(fg_mask(cg, cb, TIGHT_THRESH).sum()))
            key = f"s{i:02d}_{method}"
            jobs.append((key, crop, (gx1, gy1)))
            job_meta[key] = {
                "sample": i, "method": method,
                "crop_bbox": [gx1, gy1, gx2, gy2],
                "crop_size": [gx2 - gx1, gy2 - gy1],
                "pixel_support": support, "fallback": fallback,
            }

    # ---- batched Surya recognition (whole-crop bbox each) ----
    t_all = time.time()
    results = {}
    lat_ms = {}
    B = args.batch
    for ci in range(0, len(jobs), B):
        chunk = jobs[ci:ci + B]
        t0 = time.time()
        res = run_recognition(
            [c for _, c, _ in chunk], [["en"]] * len(chunk),
            model, processor,
            bboxes=[[[0, 0, c.size[0], c.size[1]]] for _, c, _ in chunk])
        dt_ms = (time.time() - t0) * 1000.0
        for (key, _, (ox, oy)), r in zip(chunk, res):
            lines = []
            for tl in r.text_lines:
                poly = getattr(tl, "polygon", None)
                try:
                    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
                    cb = [float(pts[:, 0].min()), float(pts[:, 1].min()),
                          float(pts[:, 0].max()), float(pts[:, 1].max())]
                    gb = [cb[0] + ox, cb[1] + oy, cb[2] + ox, cb[3] + oy]
                except Exception:
                    cb, gb = None, None
                lines.append({"text_raw": tl.text,
                              "confidence": tl.confidence,
                              "bbox_crop": cb, "bbox_global": gb})
            results[key] = lines
            lat_ms[key] = round(dt_ms / len(chunk), 1)
        print(f"  batch done ({min(ci + B, len(jobs))}/{len(jobs)})",
              flush=True)
    print(f"recognition: {time.time() - t_all:.0f}s total", flush=True)

    # ---- per-sample per-method analysis (strict rule, same as taxonomy) ----
    records = {}
    for key, meta in job_meta.items():
        i, method = meta["sample"], meta["method"]
        s = samples[i]
        gt = s["bbox"]
        tokens = expected_tokens(s["instruction"])
        lines = results[key]
        best_iou, best_l = 0.0, None
        for l in lines:
            if l["bbox_global"] is None:
                continue
            v = box_iou(l["bbox_global"], gt)
            if v > best_iou:
                best_iou, best_l = v, l
        text_match, match_txt = False, None
        for l in lines:
            nt = norm_text(l["text_raw"])
            if any(t in nt for t in tokens):
                text_match, match_txt = True, l["text_raw"]
                break
        success = bool(text_match and best_iou >= 0.5)
        records[key] = {
            **meta,
            "instruction": s["instruction"], "platform": s["platform"],
            "gt": gt, "tokens": tokens,
            "surya_texts": [(l["text_raw"][:60], l["confidence"],
                             l["bbox_global"]) for l in lines],
            "surya_text": best_l["text_raw"][:60] if best_l else (
                lines[0]["text_raw"][:60] if lines else ""),
            "surya_bbox_global": best_l["bbox_global"] if best_l else None,
            "confidence": best_l["confidence"] if best_l else None,
            "text_match": text_match, "iou": round(best_iou, 3),
            "success": success, "latency_ms": lat_ms[key],
        }

    def succ_count(m):
        return sum(1 for k, r in records.items()
                   if r["method"] == m and r["success"])

    def tm_count(m):
        return sum(1 for k, r in records.items()
                   if r["method"] == m and r["text_match"])

    # best deterministic refinement: strict success, then text_match,
    # then mean IoU, then fixed METHODS order (no voting, post-hoc pick
    # for contact sheet #2 only).
    cands = [m for m in METHODS if m != "OMNI_RAW"]
    def mean_iou(m):
        return sum(r["iou"] for k, r in records.items()
                   if r["method"] == m) / 12.0
    best = sorted(cands, key=lambda m: (
        -succ_count(m), -tm_count(m), -mean_iou(m),
        METHODS.index(m)))[0]
    print(f"best deterministic refinement: {best}", flush=True)

    (out_dir / "results.json").write_text(json.dumps(
        {"records": records, "methods": METHODS,
         "omni_boxes": {str(k): v for k, v in omni_boxes.items()},
         "best_method": best,
         "rules": {"strict": "text_match AND iou>=0.5",
                   "text_match": "any instruction token substring",
                   "best_pick": "strict desc, text_match desc, "
                                "mean_iou desc, METHODS order"}}, indent=1,
        default=str))
    print(f"results.json written ({len(records)} records)")

    # ---- per_sample.csv (required columns + confidence/latency/fallback) ----
    with open(out_dir / "per_sample.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample", "method", "crop_bbox", "crop_size",
                    "surya_text", "surya_bbox_global", "text_match", "IoU",
                    "pixel_support", "success", "confidence", "latency_ms",
                    "fallback"])
        for i in range(12):
            for m in METHODS:
                r = records[f"s{i:02d}_{m}"]
                w.writerow([
                    f"s{i:02d}", m,
                    ",".join(str(int(round(v))) for v in r["crop_bbox"]),
                    f"{r['crop_size'][0]}x{r['crop_size'][1]}",
                    (r["surya_text"] or "")[:60],
                    (",".join(f"{v:.1f}" for v in r["surya_bbox_global"])
                     if r["surya_bbox_global"] else ""),
                    r["text_match"], r["iou"], r["pixel_support"],
                    r["success"], r["confidence"], r["latency_ms"],
                    r["fallback"]])

    # ---- summary.csv ----
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "strict_success/12", "text_match/12",
                    "mean_IoU", "mean_pixel_support", "latency"])
        for m in METHODS:
            rs = [r for k, r in records.items() if r["method"] == m]
            w.writerow([m, sum(r["success"] for r in rs),
                        sum(r["text_match"] for r in rs),
                        round(sum(r["iou"] for r in rs) / 12.0, 3),
                        round(sum(r["pixel_support"] for r in rs) / 12.0, 1),
                        round(sum(r["latency_ms"] for r in rs) / 12.0, 1)])

    # ---- rescued_cases.csv: RAW vs refinements for s01/s05/s09 + s00/s02 ----
    with open(out_dir / "rescued_cases.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample", "method", "crop_bbox", "crop_size",
                    "surya_text", "text_match", "IoU", "success",
                    "delta_vs_raw"])
        for i in [1, 5, 9, 0, 2]:
            raw = records[f"s{i:02d}_OMNI_RAW"]
            for m in METHODS:
                r = records[f"s{i:02d}_{m}"]
                dw = r["crop_size"][0] - raw["crop_size"][0]
                dh = r["crop_size"][1] - raw["crop_size"][1]
                w.writerow([f"s{i:02d}", m,
                            ",".join(str(int(round(v)))
                                     for v in r["crop_bbox"]),
                            f"{r['crop_size'][0]}x{r['crop_size'][1]}",
                            (r["surya_text"] or "")[:60],
                            r["text_match"], r["iou"], r["success"],
                            f"dW={dw:+d},dH={dh:+d}"])

    # ---- visualizations: 12 x 8 full-page PNGs ----
    # GREEN=GT, BLUE=original omni, ORANGE=refined crop, RED=Surya lines.
    def draw_full(i, method, fname):
        s = samples[i]
        img = imgs[i].copy()
        d = ImageDraw.Draw(img)
        for l in results[f"s{i:02d}_{method}"]:
            if l["bbox_global"] is None:
                continue
            b = l["bbox_global"]
            d.rectangle(b, outline=(255, 0, 0), width=2)
            conf = l["confidence"] if l["confidence"] is not None else "?"
            d.text((b[0], max(0, b[1] - 12)),
                   f"{(l['text_raw'] or '')[:24]}:{conf}", fill=(255, 0, 0))
        ob = omni_boxes[i]
        d.rectangle(ob, outline=(0, 0, 255), width=3)
        d.text((ob[0], max(0, ob[1] - 12)), "OMNI", fill=(0, 0, 255))
        cb = job_meta[f"s{i:02d}_{method}"]["crop_bbox"]
        d.rectangle(cb, outline=(255, 165, 0), width=2)
        d.text((cb[0], max(0, cb[3] + 2)),
               f"CROP:{method}", fill=(255, 165, 0))
        gb = s["bbox"]
        d.rectangle(gb, outline=(0, 200, 0), width=4)
        d.text((gb[0], max(0, gb[1] - 14)), "GT", fill=(0, 200, 0))
        r = records[f"s{i:02d}_{method}"]
        d.text((4, 4),
               f"s{i:02d} {method} match={r['text_match']} "
               f"iou={r['iou']} {'SUCCESS' if r['success'] else 'FAIL'}",
               fill=(255, 255, 0))
        img.save(out_dir / fname)

    for i in range(12):
        for m in METHODS:
            draw_full(i, m, f"sample_{i:02d}_{m}.png")
    print("per-sample visualizations done (12x8)", flush=True)

    # ---- contact sheets ----
    def contact_sheet(idxs, method, fname, title):
        thumbs = []
        for i in idxs:
            s = samples[i]
            th = imgs[i].copy()
            th.thumbnail((640, 640))
            sx, sy = th.width / imgs[i].width, th.height / imgs[i].height
            d = ImageDraw.Draw(th)
            for l in results[f"s{i:02d}_{method}"]:
                if l["bbox_global"] is None:
                    continue
                b = l["bbox_global"]
                d.rectangle([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy],
                            outline=(255, 0, 0), width=2)
            ob = omni_boxes[i]
            d.rectangle([ob[0] * sx, ob[1] * sy, ob[2] * sx, ob[3] * sy],
                        outline=(0, 0, 255), width=2)
            cb = job_meta[f"s{i:02d}_{method}"]["crop_bbox"]
            d.rectangle([cb[0] * sx, cb[1] * sy, cb[2] * sx, cb[3] * sy],
                        outline=(255, 165, 0), width=2)
            gb = s["bbox"]
            d.rectangle([gb[0] * sx, gb[1] * sy, gb[2] * sx, gb[3] * sy],
                        outline=(0, 200, 0), width=3)
            r = records[f"s{i:02d}_{method}"]
            d.text((4, 4),
                   f"s{i:02d} {'OK' if r['success'] else '--'}",
                   fill=(255, 255, 0))
            thumbs.append(th)
        W = max(t.width for t in thumbs)
        H = sum(t.height for t in thumbs) + 30
        cv = Image.new("RGB", (W, H), "white")
        ImageDraw.Draw(cv).text((4, 4), title, fill=(0, 0, 0))
        y = 30
        for t in thumbs:
            cv.paste(t, (0, y))
            y += t.height
        cv.save(out_dir / fname)

    contact_sheet(list(range(12)), "OMNI_RAW", "contact_raw.png",
                  "OMNI_RAW all 12 (green=GT blue=OMNI orange=crop red=Surya)")
    contact_sheet(list(range(12)), best, "contact_best.png",
                  f"{best} all 12 (green=GT blue=OMNI orange=crop red=Surya)")
    contact_sheet([1, 5, 9], best, "contact_rescued.png",
                  f"GT-crop-rescuable s01 s05 s09 with {best}")

    # ---- summary.md ----
    sum_rows = list(csv.DictReader(open(out_dir / "summary.csv")))
    raw_ok = succ_count("OMNI_RAW")
    best_ok = succ_count(best)
    regs = [f"s{i:02d}" for i in range(12)
            if records[f"s{i:02d}_OMNI_RAW"]["success"]
            and not records[f"s{i:02d}_{best}"]["success"]]
    recov = [f"s{i:02d}" for i in [1, 5, 9]
             if records[f"s{i:02d}_{best}"]["success"]]
    md = ["# Surya Omni-Crop Refinement",
          "",
          "No VLM, no CRAFT/DBNet++, no OCR voting, no ensemble, "
          "no pipeline changes. Surya 0.4.2 classic fixed.",
          "",
          f"Best deterministic refinement (post-hoc pick, no voting): "
          f"{best}",
          "",
          "## Summary (all 12)",
          "",
          "| method | strict_success/12 | text_match/12 | mean_IoU | "
          "mean_pixel_support | latency |",
          "| --- | --- | --- | --- | --- | --- |"]
    for row in sum_rows:
        md.append(f"| {row['method']} | {row['strict_success/12']} | "
                  f"{row['text_match/12']} | {row['mean_IoU']} | "
                  f"{row['mean_pixel_support']} | {row['latency']} |")
    md += ["",
           "## GT-crop-rescuable cases (s01, s05, s09): RAW vs refinements",
           ""]
    for i in [1, 5, 9]:
        md.append(f"### s{i:02d} "
                  f"{samples[i]['instruction'][:50]}")
        for m in METHODS:
            r = records[f"s{i:02d}_{m}"]
            md.append(f"- {m}: text={(r['surya_text'] or '')[:40]!r} "
                      f"match={r['text_match']} iou={r['iou']} "
                      f"{'SUCCESS' if r['success'] else 'fail'} "
                      f"crop={r['crop_size'][0]}x{r['crop_size'][1]} "
                      f"{'FALLBACK' if r['fallback'] else ''}")
        md.append("")
    md += ["## NOT rescued even by GT crop (s00, s02)",
           ""]
    for i in [0, 2]:
        md.append(f"### s{i:02d} "
                  f"{samples[i]['instruction'][:50]}")
        for m in METHODS:
            r = records[f"s{i:02d}_{m}"]
            md.append(f"- {m}: text={(r['surya_text'] or '')[:40]!r} "
                      f"match={r['text_match']} iou={r['iou']} "
                      f"{'SUCCESS' if r['success'] else 'fail'}")
        md.append("")
    if best_ok > raw_ok and not regs and recov:
        verdict = (f"PROMISING: {best} improves strict success "
                   f"{raw_ok}/12 -> {best_ok}/12 with no regressions "
                   f"and recovers {recov}.")
    elif best_ok > raw_ok and recov:
        verdict = (f"MIXED: {best} improves strict success "
                   f"{raw_ok}/12 -> {best_ok}/12 and recovers {recov} "
                   f"but introduces regressions {regs}.")
    else:
        verdict = (f"NOT PROMISING: best refinement {best} reaches "
                   f"{best_ok}/12 vs RAW {raw_ok}/12, recovering {recov} "
                   f"with regressions {regs}. OmniParser bbox geometry "
                   f"alone is insufficient; move to the next failure "
                   f"mechanism.")
    md += ["## Decision",
           "",
           verdict,
           "",
           "Refinements use only omni-crop pixels (no GT). Strict rule: "
           "text_match AND IoU>=0.5."]
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")

    # ---- console report ----
    print("\n" + "=" * 60)
    print("SURYA OMNI-CROP REFINEMENT")
    print("=" * 60)
    print(f"\nBEST: {best}  RAW strict: {raw_ok}/12  BEST strict: "
          f"{best_ok}/12")
    print("\n------------------------------------------------------------")
    print("SUMMARY")
    print("------------------------------------------------------------\n")
    print("method                    strict  text_m  meanIoU  "
          "meanSup  latency")
    print("------------------------------------------------------------")
    for row in sum_rows:
        print(f"{row['method']:24s}  {row['strict_success/12']:>6s}  "
              f"{row['text_match/12']:>6s}  {row['mean_IoU']:>7s}  "
              f"{row['mean_pixel_support']:>7s}  {row['latency']:>7s}")
    print("\n------------------------------------------------------------")
    print("RESCUED-CASE FOCUS (s01, s05, s09)")
    print("------------------------------------------------------------\n")
    for i in [1, 5, 9]:
        print(f"s{i:02d}:")
        for m in METHODS:
            r = records[f"s{i:02d}_{m}"]
            print(f"  {m:22s} match={r['text_match']!s:5s} "
                  f"iou={r['iou']:.3f} "
                  f"{'SUCCESS' if r['success'] else 'fail'}")
    print("\n------------------------------------------------------------")
    print("DECISION")
    print("------------------------------------------------------------\n")
    print(verdict)
    print(f"\nArtifacts: {out_dir}")


if __name__ == "__main__":
    main()
