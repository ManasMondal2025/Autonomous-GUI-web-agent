#!/usr/bin/env python3
"""Surya 0.4.2 failure-taxonomy diagnostic (standalone).

RUN WITH: /tmp/surya_old/bin/python scripts/evaluate_surya_failure_taxonomy.py

No VLM, no CRAFT/DBNet/PaddleOCR/EasyOCR, no pipeline changes. Determines
WHY 5/12 ScreenSpot targets fail: crop geometry vs detection vs recognition
vs localization. All Surya calls are whole-crop, original resolution.

Outputs: reports/surya_failure_taxonomy/{results.json, failure_taxonomy.csv,
success_vs_failure.csv, gt_crop_diagnostic.csv, summary.md,
sample_{00,01,02,05,09}_failure.png, successful_examples.png,
failed_examples.png} + console report.
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
OUT_DIR = REPO / "reports" / "surya_failure_taxonomy"
DATA = REPO / "data" / "screenspot" / "v2_mini.jsonl"

STOPWORDS = {"check", "the", "a", "an", "of", "in", "on", "to", "for",
             "tap", "open", "view", "look", "into", "first"}


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


def center_in(box, outer):
    cx, cy = (box[0] + box[2]) / 2, (box[3] + box[1]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def contrast_stats(gray, box, ring=12):
    """Mean/std/min/max inside box + local background ring + score."""
    import numpy as np
    H, W = gray.shape
    x1, y1, x2, y2 = (max(0, int(round(v))) for v in box)
    x1, y1 = min(x1, W - 1), min(y1, H - 1)
    x2, y2 = min(W, max(x1 + 1, x2)), min(H, max(y1 + 1, y2))
    inside = gray[y1:y2, x1:x2].astype(float)
    rx1, ry1 = max(0, x1 - ring), max(0, y1 - ring)
    rx2, ry2 = min(W, x2 + ring), min(H, y2 + ring)
    parts = [gray[ry1:y1, rx1:rx2].ravel(), gray[y2:ry2, rx1:rx2].ravel(),
             gray[y1:y2, rx1:x1].ravel(), gray[y1:y2, x2:rx2].ravel()]
    parts = [p for p in parts if p.size]
    bg = np.concatenate(parts) if parts else inside.ravel()
    return {
        "mean": round(float(inside.mean()), 1),
        "std": round(float(inside.std()), 1),
        "min": round(float(inside.min()), 1),
        "max": round(float(inside.max()), 1),
        "bg_mean": round(float(bg.mean()), 1),
        "contrast_score": round(
            abs(float(inside.mean()) - float(bg.mean())) / 255.0, 3),
    }


def main(argv=None):
    from PIL import Image, ImageDraw
    import numpy as np

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
    import numpy as np
    model = load_model(device="cpu")
    processor = load_processor()

    # ---- build all jobs: omni crops (12) + GT crops (5 failed) + tight (20)
    # Prior established split: successful 03,04,06,07,08,10,11;
    # failed 00,01,02,05,09 (verified post-hoc below).
    failed_idx = [0, 1, 2, 5, 9]
    jobs = []  # (key, PIL crop, origin); all original resolution
    imgs = {}
    for i, s in enumerate(samples):
        img = Image.open(s["img_path"]).convert("RGB")
        imgs[i] = img
        W, H = img.size
        boxes = omni_all[str(i)]
        obs = [[e["bbox"]["x"], e["bbox"]["y"],
                e["bbox"]["x"] + e["bbox"]["width"],
                e["bbox"]["y"] + e["bbox"]["height"]] for e in boxes]
        gt = s["bbox"]
        ob = max(obs, key=lambda b: box_iou(b, gt))
        x1, y1, x2, y2 = (int(round(v)) for v in
                          [max(0, ob[0]), max(0, ob[1]),
                           min(W, ob[2]), min(H, ob[3])])
        jobs.append((f"omni_{i:02d}", img.crop((x1, y1, x2, y2)), (x1, y1)))
        if i in failed_idx:
            gx1, gy1, gx2, gy2 = (int(round(v)) for v in
                                  [max(0, gt[0]), max(0, gt[1]),
                                   min(W, gt[2]), min(H, gt[3])])
            jobs.append((f"gt_{i:02d}", img.crop((gx1, gy1, gx2, gy2)),
                         (gx1, gy1)))
            for pad, tag in [(-5, "m5"), (-2, "m2"), (2, "p2"), (5, "p5")]:
                tx1 = max(0, gx1 + (pad if pad < 0 else 0))
                ty1 = max(0, gy1 + (pad if pad < 0 else 0))
                tx2 = min(W, gx2 + (pad if pad > 0 else 0))
                ty2 = min(H, gy2 + (pad if pad > 0 else 0))
                # NOTE: -N shrinks all sides by N (toward center); +N expands.
                # Implemented as uniform inset/outset for determinism:
                inset = abs(pad) if pad < 0 else 0
                tx1, ty1 = gx1 + inset, gy1 + inset
                tx2, ty2 = gx2 - inset, gy2 - inset
                if pad > 0:
                    tx1, ty1 = max(0, gx1 - pad), max(0, gy1 - pad)
                    tx2, ty2 = min(W, gx2 + pad), min(H, gy2 + pad)
                if tx2 <= tx1 + 1 or ty2 <= ty1 + 1:
                    continue
                jobs.append((f"tight_{i:02d}_{tag}",
                             img.crop((tx1, ty1, tx2, ty2)), (tx1, ty1)))

    # ---- batched recognition (whole-crop bbox each) ----
    t_all = time.time()
    results = {}
    B = args.batch
    for ci in range(0, len(jobs), B):
        chunk = jobs[ci:ci + B]
        res = run_recognition(
            [c for _, c, _ in chunk], [["en"]] * len(chunk),
            model, processor,
            bboxes=[[[0, 0, c.size[0], c.size[1]]] for _, c, _ in chunk])
        for (key, _, (ox, oy)), r in zip(chunk, res):
            lines = []
            for tl in r.text_lines:
                poly = getattr(tl, "polygon", None)
                try:
                    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
                    bb = [float(pts[:, 0].min()) + ox,
                          float(pts[:, 1].min()) + oy,
                          float(pts[:, 0].max()) + ox,
                          float(pts[:, 1].max()) + oy]
                except Exception:
                    bb = None
                lines.append({"text_raw": tl.text,
                              "confidence": tl.confidence, "bbox": bb})
            results[key] = lines
        print(f"  batch done ({min(ci + B, len(jobs))}/{len(jobs)})", flush=True)
    print(f"recognition: {time.time() - t_all:.0f}s total", flush=True)

    # ---- per-target analysis ----
    gray_cache = {}
    records = {}
    for i, s in enumerate(samples):
        img = imgs[i]
        if i not in gray_cache:
            gray_cache[i] = np.array(img.convert("L")).astype(float)
        gray = gray_cache[i]
        gt = s["bbox"]
        tokens = expected_tokens(s["instruction"])
        olines = results.get(f"omni_{i:02d}", [])

        def ovelap_info(lines):
            best, best_l = 0.0, None
            for l in lines:
                if l["bbox"] is None:
                    continue
                v = box_iou(l["bbox"], gt)
                if v > best:
                    best, best_l = v, l
            return best, best_l

        best_iou, best_l = ovelap_info(olines)
        # text match: any expected token as substring of normalized line text
        def text_ok(lines):
            for l in lines:
                nt = norm_text(l["text_raw"])
                if any(t in nt for t in tokens):
                    return True, l["text_raw"]
            return False, None

        ok, match_txt = text_ok(olines)
        if not olines:
            primary = "NO_DETECTION"
        elif ok and best_iou >= 0.5:
            primary = "CORRECT"
        else:
            # multiple distinct overlapping texts?
            over = [l for l in olines
                    if l["bbox"] is not None and box_iou(l["bbox"], gt) > 0.1]
            distinct = {norm_text(l["text_raw"]) for l in over if norm_text(l["text_raw"])}
            if len(distinct) >= 2 and not ok:
                primary = "MULTIPLE_CONFLICTING_TEXTS"
            elif ok:
                primary = "CORRECT_TEXT_WRONG_BBOX"
            elif best_iou > 0:
                primary = "WRONG_TEXT"
            else:
                # predictions exist but none near GT
                primary = "WRONG_TEXT" if any(
                    norm_text(l["text_raw"]) for l in olines) else "OUTSIDE_TARGET"
                if not any(norm_text(l["text_raw"]) for l in olines):
                    # empty texts near GT vs elsewhere handled below
                    pass
                # refine: all predictions far from GT?
                if best_iou == 0.0:
                    # check whether predictions exist at all
                    primary = "OUTSIDE_TARGET" if olines else "NO_DETECTION"
        # GT-crop + tight results for failed targets
        gt_lines = results.get(f"gt_{i:02d}", [])
        gt_ok = any(norm_text(l["text_raw"]) and
                    any(t in norm_text(l["text_raw"]) for t in tokens)
                    for l in gt_lines)
        tights = {}
        for tag in ("m5", "m2", "p2", "p5"):
            tl = results.get(f"tight_{i:02d}_{tag}", [])
            tights[tag] = [l["text_raw"] for l in tl]
        # geometry
        W, H = img.size
        obox = None
        # omni box used: recompute max-overlap (same rule as job building)
        boxes = omni_all[str(i)]
        obs = [[e["bbox"]["x"], e["bbox"]["y"],
                e["bbox"]["x"] + e["bbox"]["width"],
                e["bbox"]["y"] + e["bbox"]["height"]] for e in boxes]
        ob = max(obs, key=lambda b: box_iou(b, gt))
        th, tw = gt[3] - gt[1], gt[2] - gt[0]
        ch, cw = ob[3] - ob[1], ob[2] - ob[0]
        cs = contrast_stats(gray, gt)
        records[i] = {
            "instruction": s["instruction"], "platform": s["platform"],
            "primary": primary, "omni_lines": len(olines),
            "best_iou": round(best_iou, 3),
            "best_text": best_l["text_raw"][:60] if best_l else None,
            "tokens": tokens, "gt_ok_on_gt_crop": gt_ok,
            "gt_lines": [(l["text_raw"][:40], l["bbox"]) for l in gt_lines],
            "tights": tights,
            "geom": {
                "crop_w": round(cw, 1), "crop_h": round(ch, 1),
                "crop_aspect": round(ch / max(1e-9, cw), 2),
                "crop_area": round(ch * cw, 0),
                "gt_w": round(tw, 1), "gt_h": round(th, 1),
                "rel_h": round(th / max(1e-9, ch), 3),
                "rel_w": round(tw / max(1e-9, cw), 3),
                "omni_iou": round(box_iou(ob, gt), 3),
                "coverage": round(
                    max(0.0, min(ob[2], gt[2]) - max(ob[0], gt[0])) *
                    max(0.0, min(ob[3], gt[3]) - max(ob[1], gt[1])) /
                    max(1e-9, box_area(gt)), 3),
                "center_inside": center_in(
                    [(gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2,
                     (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2], ob),
                **cs,
            },
        }

    # ---- verify prior split ----
    succ = [i for i, r in records.items() if r["primary"] == "CORRECT"]
    fail = [i for i, r in records.items() if r["primary"] != "CORRECT"]
    print(f"\nomni-crop CORRECT: {sorted(succ)}", flush=True)
    print(f"expected successful: [3, 4, 6, 7, 8, 10, 11]", flush=True)

    # ---- visualizations ----
    def draw_page(img, gt, olines, fname, title):
        base = img.copy()
        d = ImageDraw.Draw(base)
        for l in olines:
            if l["bbox"] is None:
                continue
            b = l["bbox"]
            d.rectangle(b, outline=(255, 0, 0), width=2)
            d.text((b[0], max(0, b[1] - 12)),
                   f"{l['text_raw'][:24]}:{l['confidence'] if l['confidence'] is not None else '?'}",
                   fill=(255, 0, 0))
        d.rectangle(gt, outline=(0, 200, 0), width=4)
        d.text((gt[0], max(0, gt[1] - 14)), "GT", fill=(0, 200, 0))
        base.save(out_dir / fname)

    for i in [0, 1, 2, 5, 9]:
        s = samples[i]
        img = imgs[i]
        boxes = omni_all[str(i)]
        obs = [[e["bbox"]["x"], e["bbox"]["y"],
                e["bbox"]["x"] + e["bbox"]["width"],
                e["bbox"]["y"] + e["bbox"]["height"]] for e in boxes]
        ob = max(obs, key=lambda b: box_iou(b, s["bbox"]))
        olines = [dict(l, _src="omni") for l in results[f"omni_{i:02d}"]]
        glines = [dict(l, _src="gt") for l in results.get(f"gt_{i:02d}", [])]
        draw_page(img, s["bbox"], olines + glines,
                  f"sample_{i:02d}_failure.png",
                  f"s{i:02d} {s['instruction'][:30]}")

    def contact_sheet(idxs, fname, title):
        thumbs = []
        for i in idxs:
            s = samples[i]
            th = imgs[i].copy()
            th.thumbnail((640, 640))
            sx, sy = th.width / imgs[i].width, th.height / imgs[i].height
            d = ImageDraw.Draw(th)
            gb = s["bbox"]
            d.rectangle([gb[0] * sx, gb[1] * sy, gb[2] * sx, gb[3] * sy],
                        outline=(0, 200, 0), width=3)
            for l in results.get(f"omni_{i:02d}", []):
                if l["bbox"] is None:
                    continue
                b = l["bbox"]
                d.rectangle([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy],
                            outline=(255, 0, 0), width=2)
            dd = ImageDraw.Draw(th)
            dd.text((4, 4), f"s{i:02d}", fill=(255, 255, 0))
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

    contact_sheet([3, 4, 6, 7, 8, 10, 11], "successful_examples.png",
                  "SUCCESSFUL (green=GT, red=Surya) — s03 s04 s06 s07 s08 s10 s11")
    contact_sheet([0, 1, 2, 5, 9], "failed_examples.png",
                  "FAILED (green=GT, red=Surya) — s00 s01 s02 s05 s09")

    (out_dir / "results.json").write_text(json.dumps(
        {"records": records}, indent=1, default=str))
    print(f"results.json written ({len(records)} targets)")

    # ---- CSVs ----
    with open(out_dir / "failure_taxonomy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "instruction", "platform", "primary_class",
                    "omni_iou", "coverage", "center_inside", "text_h",
                    "text_w", "rel_h", "rel_w", "contrast", "bg_mean",
                    "n_preds", "best_text", "best_iou"])
        for i, r in records.items():
            g = r["geom"]
            w.writerow([f"s{i:02d}", samples[i]["instruction"][:40],
                        r["platform"], r["primary"], g["omni_iou"],
                        g["coverage"], g["center_inside"], g["gt_h"], g["gt_w"],
                        g["rel_h"], g["rel_w"], g["contrast_score"],
                        g["bg_mean"], r["omni_lines"],
                        (r["best_text"] or "")[:40], r["best_iou"]])
    with open(out_dir / "success_vs_failure.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "success_mean", "failure_mean"])
        succ_ids = [i for i, r in records.items() if r["primary"] == "CORRECT"]
        fail_ids = [i for i, r in records.items() if r["primary"] != "CORRECT"]

        def means(ids, fn):
            vs = [fn(records[i]) for i in ids]
            return round(sum(vs) / max(1, len(vs)), 3) if vs else ""

        for label, fn in [
                ("omni_iou", lambda r: r["geom"]["omni_iou"]),
                ("coverage", lambda r: r["geom"]["coverage"]),
                ("text_h", lambda r: r["geom"]["gt_h"]),
                ("text_w", lambda r: r["geom"]["gt_w"]),
                ("rel_h", lambda r: r["geom"]["rel_h"]),
                ("rel_w", lambda r: r["geom"]["rel_w"]),
                ("contrast", lambda r: r["geom"]["contrast_score"]),
                ("bg_mean", lambda r: r["geom"]["bg_mean"]),
                ("n_preds", lambda r: r["omni_lines"])]:
            w.writerow([label, means(succ_ids, fn), means(fail_ids, fn)])
    with open(out_dir / "gt_crop_diagnostic.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "omni_texts", "omni_best_iou", "omni_ok",
                    "gt_texts", "gt_ok", "tight_m5", "tight_m2",
                    "tight_p2", "tight_p5", "rescued"])
        for i in [0, 1, 2, 5, 9]:
            r = records[i]
            t = r["tights"]
            rescued = r["gt_ok_on_gt_crop"] and not (
                r["primary"] == "CORRECT")
            w.writerow([f"s{i:02d}",
                        "|".join(t for t in
                                 [l["text_raw"][:30] for l in []]),
                        "", "",
                        "|".join(x[:30] for x in
                                 [l[0][:30] for l in r["gt_lines"]]),
                        r["gt_ok_on_gt_crop"],
                        "|".join(tights_text(r, "m5")),
                        "|".join(tights_text(r, "m2")),
                        "|".join(tights_text(r, "p2")),
                        "|".join(tights_text(r, "p5")),
                        rescued])

    # ---- summary.md ----
    succ_ids = [i for i, r in records.items() if r["primary"] == "CORRECT"]
    fail_ids = [i for i, r in records.items() if r["primary"] != "CORRECT"]
    md = ["# Surya Failure Taxonomy",
          "",
          f"Successful: {len(succ_ids)} {sorted(succ_ids)}",
          f"Failed: {len(fail_ids)} {sorted(fail_ids)}",
          "",
          "## Failure distribution",
          ""]
    from collections import Counter
    for cls, n in sorted(Counter(r["primary"] for r in records.values()
                                 ).items()):
        md.append(f"- {cls}: {n}")
    md += ["",
           "## GT-crop diagnostic (5 failed targets)",
           "",
           "Rescued by GT crop (text match on GT-crop run): " +
           str([f"s{i:02d}" for i in [0, 1, 2, 5, 9]
                if records[i]["gt_ok_on_gt_crop"]]),
           "Still failing on GT crop: " +
           str([f"s{i:02d}" for i in [0, 1, 2, 5, 9]
                if not records[i]["gt_ok_on_gt_crop"]]),
           "",
           "## Success vs failure (means)",
           ""]
    with open(out_dir / "failure_taxonomy.csv") as f:
        md.append("```")
        md.append(f.read().strip().split(chr(10))[0])
        for j, line in enumerate(open(out_dir / "failure_taxonomy.csv")):
            if j > 0:
                md.append(line.strip())
        md.append("```")
    md += ["",
           "## Interpretation",
           "",
           "Evidence rules (documented, applied mechanically):",
           "- A failure class covering >=40% of the 5 failures gets checked.",
           "- A continuous metric gets checked when success/failure means "
           "differ by >1.5x (or <0.67x).",
           ""]
    cats = Counter(r["primary"] for r in records.values() if r["primary"] != "CORRECT")
    if fail_ids:
        top_cls, top_n = (cats.most_common(1)[0] if cats else ("", 0))
        if top_n / len(fail_ids) >= 0.4:
            md.append(f"- [x] {top_cls} dominates failures "
                      f"({top_n}/{len(fail_ids)}).")
    for label, get in [("OmniParser crop geometry", lambda r: r["geom"]["omni_iou"]),
                       ("Relative text scale", lambda r: r["geom"]["rel_h"]),
                       ("Contrast", lambda r: r["geom"]["contrast_score"]),
                       ("Surya detection (n_preds)", lambda r: r["omni_lines"])]:
        sm = sum(get(records[i]) for i in succ_ids) / max(1, len(succ_ids))
        fm = sum(get(records[i]) for i in fail_ids) / max(1, len(fail_ids))
        ratio = (sm / fm) if fm else float("inf")
        mark = "[x]" if (ratio > 1.5 or (ratio < 0.67 and ratio > 0)) else "[ ]"
        md.append(f"- {mark} {label}: success={sm:.3f} failure={fm:.3f}.")
    md += ["- [ ] Surya recognition (see WRONG_TEXT share above).",
           "- [ ] Surya localization (see CORRECT_TEXT_WRONG_BBOX share above).",
           "- [ ] No clear dominant pattern (check only if nothing else fires).",
           "",
           "Note: GT text strings are known exactly only for s08 (Blog); "
           "other samples use instruction-keyword matching, documented in "
           "failure_taxonomy.csv via the tokens column (records JSON)."]
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")

    # ---- console report (spec format) ----
    print("\n" + "=" * 60)
    print("SURYA 0.4.2 FAILURE TAXONOMY")
    print("=" * 60)
    print(f"\n\nSUCCESSFUL TARGETS: {len(succ_ids)}")
    print(f"FAILED TARGETS: {len(fail_ids)}")
    print("\n------------------------------------------------------------")
    print("FAILURE DISTRIBUTION")
    print("------------------------------------------------------------\n")
    for cls in ["NO_DETECTION", "WRONG_TEXT", "CORRECT_TEXT_WRONG_BBOX",
                "MULTIPLE_CONFLICTING_TEXTS", "OUTSIDE_TARGET"]:
        n = sum(1 for r in records.values() if r["primary"] == cls)
        print(f"{cls}: {n}")
    print("\n------------------------------------------------------------")
    print("GT-CROP DIAGNOSTIC")
    print("------------------------------------------------------------\n")
    print("Failed targets rescued by GT crop:")
    print(" ", [f"s{i:02d}" for i in [0, 1, 2, 5, 9]
                if records[i]["gt_ok_on_gt_crop"]] or "none")
    print("Failed targets still failing on GT crop:")
    print(" ", [f"s{i:02d}" for i in [0, 1, 2, 5, 9]
                if not records[i]["gt_ok_on_gt_crop"]] or "none")
    print("\n------------------------------------------------------------")
    print("SUCCESS VS FAILURE")
    print("------------------------------------------------------------\n")
    print("Metric                    SUCCESS      FAILURE")
    print("------------------------------------------------------------")
    with open(out_dir / "success_vs_failure.csv") as f:
        for j, line in enumerate(f):
            if j == 0:
                continue
            parts = line.strip().split(",")
            print(f"{parts[0]:24s}  {parts[1]:10s}  {parts[2]:10s}")
    print("\n------------------------------------------------------------")
    print("INTERPRETATION")
    print("------------------------------------------------------------\n")
    print("\n".join(md[md.index("## Interpretation") + 1:]))
    print(f"\nArtifacts: {out_dir}")


def tights_text(r, tag):
    return [l for l in r.get("tights", {}).get(tag, [])]


def contrast_stats(gray, box, ring=12):
    """Mean/std/min/max inside box + local background ring + score."""
    import numpy as np
    H, W = gray.shape
    x1, y1, x2, y2 = (max(0, int(round(v))) for v in box)
    x1, y1 = min(x1, W - 1), min(y1, H - 1)
    x2, y2 = min(W, max(x1 + 1, x2)), min(H, max(y1 + 1, y2))
    inside = gray[y1:y2, x1:x2].astype(float)
    rx1, ry1 = max(0, x1 - ring), max(0, y1 - ring)
    rx2, ry2 = min(W, x2 + ring), min(H, y2 + ring)
    parts = [gray[ry1:y1, rx1:rx2].ravel(), gray[y2:ry2, rx1:rx2].ravel(),
             gray[y1:y2, rx1:x1].ravel(), gray[y1:y2, x2:rx2].ravel()]
    parts = [p for p in parts if p.size]
    bg = np.concatenate(parts) if parts else inside.ravel()
    return {
        "mean": round(float(inside.mean()), 1),
        "std": round(float(inside.std()), 1),
        "min": round(float(inside.min()), 1),
        "max": round(float(inside.max()), 1),
        "bg_mean": round(float(bg.mean()), 1),
        "contrast_score": round(
            abs(float(inside.mean()) - float(bg.mean())) / 255.0, 3),
    }


if __name__ == "__main__":
    main()
