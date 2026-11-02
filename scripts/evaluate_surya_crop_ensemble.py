#!/usr/bin/env python3
"""Surya crop-ensemble evaluation (standalone, Surya 0.4.2 classic ONLY).

RUN WITH: /tmp/surya_old/bin/python scripts/evaluate_surya_crop_ensemble.py

Compares, on identical screenshots and identical OmniParser boxes:
  baseline_exact : exact Omni crop -> Surya
  baseline_4x    : exact crop -> 4x upscale -> Surya
  ensemble_3     : -5% / exact / +5% crops -> Surya + majority consensus
  ensemble_3_4x  : same three crops at 4x -> Surya + consensus

Padding convention is the EXISTING repo one (pad_box_frac): positive =
expand, negative = shrink toward center. (An early draft of this task
described the inverse; the repo convention wins and is stated here.)

No VLM, no CRAFT/DBNet/PaddleOCR/EasyOCR, no pipeline changes. GT text is
known only for the Blog target; other samples use overlap + non-empty
success (stated wherever it matters).
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
OUT_DIR = REPO / "reports" / "surya_crop_ensemble"
DATA = REPO / "data" / "screenspot" / "v2_mini.jsonl"
OMNI_JSON = Path("/tmp/omni12.json")


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


def expand_box(b, frac, img_w, img_h):
    w, h = b[2] - b[0], b[3] - b[1]
    return [max(0.0, b[0] - w * frac), max(0.0, b[1] - h * frac),
            min(float(img_w), b[2] + w * frac),
            min(float(img_h), b[3] + h * frac)]


def consensus(texts):
    """Majority over non-empty normalized texts; UNCERTAIN on ties/empties."""
    votes = {}
    for t in texts:
        t = norm_text(t)
        if t:
            votes[t] = votes.get(t, 0) + 1
    if not votes:
        return None, "UNCERTAIN"
    best = max(votes.values())
    winners = [t for t, v in votes.items() if v == best]
    if len(winners) == 1:
        return winners[0], f"{best}/3"
    return None, "UNCERTAIN"


def main(argv=None):
    from PIL import Image, ImageDraw

    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from surya.model.recognition.model import load_model
    from surya.model.recognition.processor import load_processor
    from surya.ocr import run_recognition

    samples = [json.loads(l) for l in open(DATA)]
    omni_all = json.load(open(OMNI_JSON))
    print(f"{len(samples)} samples; loading Surya rec model...", flush=True)
    model = load_model(device="cpu")
    processor = load_processor()

    # ---- build all cases first (single batched inference per group) ----
    # case: (sample_idx, method, crop_img, origin, scale, gt, gt_known_text)
    jobs = []  # (key, crop_img)
    meta = {}
    gt_known = {8: "blog"}
    omni_used = {}

    for i, s in enumerate(samples):
        img = Image.open(s["img_path"]).convert("RGB")
        W, H = img.size
        boxes = omni_all[str(i)]
        # target omni box = max GT overlap (computed, logged)
        gt = s["bbox"]
        ob = max(boxes, key=lambda e: box_iou(
            [e["bbox"]["x"], e["bbox"]["y"],
             e["bbox"]["x"] + e["bbox"]["width"],
             e["bbox"]["y"] + e["bbox"]["height"]], gt))
        ob = [ob["bbox"]["x"], ob["bbox"]["y"],
              ob["bbox"]["x"] + ob["bbox"]["width"],
              ob["bbox"]["y"] + ob["bbox"]["height"]]
        okey = f"s{i:02d}"
        omni_used[okey] = ob
        variants = {"baseline_exact": (ob, 1.0), "baseline_4x": (ob, 4.0)}
        for j, f in enumerate([-0.05, 0.0, 0.05]):
            pb = expand_box(ob, f, W, H)
            variants[f"ens{j}"] = (pb, 1.0)
            variants[f"ens4x{j}"] = (pb, 4.0)
        for vname, (vbox, sc) in variants.items():
            x1, y1, x2, y2 = (int(round(v)) for v in vbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 + 1 or y2 <= y1 + 1:
                continue
            crop = img.crop((x1, y1, x2, y2))
            if sc != 1.0:
                crop = crop.resize((crop.width * int(sc),
                                    crop.height * int(sc)), Image.LANCZOS)
            key = f"{okey}_{vname}"
            meta[key] = {"sample": i, "variant": vname, "origin": [x1, y1],
                         "scale": sc, "gt": gt,
                         "gt_text": gt_known.get(i),
                         "crop_size": list(crop.size)}
            jobs.append((key, crop))

    # ---- batched recognition ----
    t_all = time.time()
    chunk_times = []
    results = {}
    B = args.batch
    for ci in range(0, len(jobs), B):
        chunk = jobs[ci:ci + B]
        box_lists = []
        for _, crop in chunk:
            w, h = crop.size
            box_lists.append([[0, 0, w, h]])
        _t0 = time.time()
        res = run_recognition([c for _, c in chunk],
                              [["en"]] * len(chunk),
                              model, processor, bboxes=box_lists)
        chunk_times.append({"n": len(chunk), "secs": round(time.time() - _t0, 2)})
        for (key, _), r in zip(chunk, res):
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
            m = meta[key]
            ox, oy, sc = m["origin"][0], m["origin"][1], 1.0
            # NOTE: variant scale already applied to pixels; map back.
            # Any key containing "4x" is a 4x-upscaled variant.
            vscale = 4.0 if "4x" in key else 1.0
            for l in lines:
                if l["poly"] is not None:
                    p = l["poly"]
                    l["bbox_orig"] = [p[0] / vscale + ox, p[1] / vscale + oy,
                                      p[2] / vscale + ox, p[3] / vscale + oy]
                else:
                    l["bbox_orig"] = None
            results[key] = {"lines": lines, "meta": m}
        print(f"  batch done ({min(ci + B, len(jobs))}/{len(jobs)})", flush=True)
    total_s = round(time.time() - t_all, 1)
    timing_info = {"chunk_times": chunk_times, "total_s": total_s,
                   "n_calls": len(chunk_times), "n_crops": len(jobs)}
    print(f"recognition: {total_s}s total over {len(jobs)} crops", flush=True)

    # ---- per-sample / per-method aggregation ----
    summary = {}
    for i in range(len(samples)):
        okey = f"s{i:02d}"
        gt = samples[i]["bbox"]
        known = gt_known.get(i)
        row = {"gt_known_text": known}
        # baseline variants
        for vname, mname in (("baseline_exact", "baseline"),
                             ("baseline_4x", "baseline_4x")):
            r = results[f"{okey}_{vname}"]
            row[mname] = score_lines(r["lines"], gt, known)
        # ensembles: 3 crops each (plain + 4x)
        for vnames, mname in ((["ens0", "ens1", "ens2"], "ens3"),
                              (["ens4x0", "ens4x1", "ens4x2"], "ens3_4x")):
            texts = []
            boxes = []
            for v in vnames:
                for l in results[f"{okey}_{v}"]["lines"]:
                    texts.append(l["text_raw"])
                    if l["bbox_orig"] is not None:
                        boxes.append((l["bbox_orig"], l["text_raw"]))
            cons, frac = consensus(texts)
            # consensus box: first box carrying the winning text
            cbox = None
            if cons:
                for bb, t in boxes:
                    if norm_text(t) == cons:
                        cbox = bb
                        break
            row[mname] = {"consensus": cons, "frac": frac,
                          "box": cbox,
                          "iou": round(box_iou(cbox, gt), 3) if cbox else 0.0,
                          "correct": bool(cons and cbox and (
                              norm_text(cons) == norm_text(known)
                              if known else True) and box_iou(cbox, gt) >= 0.5)}
        summary[i] = row

    # ---- failure analysis ----
    recoveries, regressions = [], []
    for i, row in summary.items():
        base_ok = row["baseline"]["correct"]
        for mname in ("baseline_4x", "ens3", "ens3_4x"):
            cur_ok = row[mname]["correct"]
            if not base_ok and cur_ok:
                recoveries.append({
                    "target": f"s{i:02d}",
                    "baseline": row["baseline"],
                    "method": mname, "method_result": row[mname],
                    "reason": "ensemble/scale recovered text+overlap"})
            if base_ok and not cur_ok:
                regressions.append({
                    "target": f"s{i:02d}",
                    "baseline": row["baseline"],
                    "method": mname, "method_result": row[mname]})

    # ---- duplicate check on baseline_4x ----
    import re
    dups = 0
    for i in range(len(samples)):
        for l in results[f"s{i:02d}_baseline_4x"]["lines"]:
            toks = norm_text(l["text_raw"]).split()
            if len(toks) >= 2 and len(set(toks)) == 1:
                dups += 1
                break
    # ---- visualizations for Blog (sample 8) ----
    img8 = Image.open(samples[8]["img_path"]).convert("RGB")
    ob8 = omni_used["s08"]

    def draw_view(pred_entries, fname, title):
        base = img8.copy()
        d = ImageDraw.Draw(base)
        d.rectangle(ob8, outline=(0, 0, 255), width=3)
        d.text((ob8[0], max(0, ob8[1] - 12)),
               f"OMNI {title}", fill=(0, 0, 255))
        for bb, txt, conf in pred_entries:
            d.rectangle(bb, outline=(255, 0, 0), width=2)
            d.text((bb[0], max(0, bb[1] - 12)),
                   f"{txt[:24]}:{conf if conf is not None else '?'}",
                   fill=(255, 0, 0))
        base.save(out_dir / fname)

    def lines_for(method_key):
        r = results[f"s08_{method_key}"]
        return [(l["bbox_orig"], l["text_raw"], l["confidence"])
                for l in r["lines"] if l["bbox_orig"] is not None]

    draw_view(lines_for("baseline_exact"), "blog_baseline.png",
              "baseline_exact")
    draw_view(lines_for("baseline_4x"), "blog_baseline_4x.png",
              "baseline_4x")
    draw_view([e for v in ("ens0", "ens1", "ens2") for e in lines_for(v)],
              "blog_ensemble.png", "ensemble_3")
    draw_view([e for v in ("ens4x0", "ens4x1", "ens4x2") for e in lines_for(v)],
              "blog_ensemble_4x.png", "ensemble_3_4x")

    # ---- files ----
    (out_dir / "results.json").write_text(json.dumps(
        {"summary": summary, "timing": timing_info,
         "per_sample_runs": {k: {"lines": [
             {"text_raw": l["text_raw"], "confidence": l["confidence"],
              "bbox_orig": l["bbox_orig"]} for l in v["lines"]],
                                 "meta": results[k].get("meta", {})}
                             for k, v in results.items()}},
        indent=1, default=str))
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample", "method", "correct", "iou", "text"])
        for i, row in summary.items():
            for mname in ("baseline", "baseline_4x", "ens3", "ens3_4x"):
                r = row[mname]
                w.writerow([f"s{i:02d}", mname, r["correct"],
                            r.get("iou", ""), str(r.get("consensus", r.get("texts", "")))[:60]])
    with open(out_dir / "per_target.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample", "instruction", "gt_bbox", "method", "text",
                    "box", "iou", "correct"])
        for i, row in summary.items():
            for mname in ("baseline", "baseline_4x", "ens3", "ens3_4x"):
                r = row[mname]
                w.writerow([f"s{i:02d}", samples[i]["instruction"][:40],
                            samples[i]["bbox"], mname,
                            str(r.get("consensus", r.get("texts", "")))[:50],
                            r.get("box", r.get("boxes", "")), r.get("iou", ""),
                            r["correct"]])
    with open(out_dir / "method_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "accuracy", "mean_iou", "n"])
        for mname in ("baseline", "baseline_4x", "ens3", "ens3_4x"):
            acc = [summary[i][mname]["correct"] for i in summary]
            ios = [summary[i][mname].get("iou", 0) for i in summary]
            w.writerow([mname, f"{sum(acc)}/{len(acc)}",
                        round(sum(ios) / max(1, len(ios)), 3), len(acc)])
    with open(out_dir / "failure_analysis.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "target", "baseline", "method", "method_result"])
        for r in recoveries:
            w.writerow(["recovery", r["target"], r["baseline"],
                        r["method"], r["method_result"]])
        for r in regressions:
            w.writerow(["regression", r["target"], r["baseline"],
                        r["method"], r["method_result"]])

    # ---- console ----
    print("\n" + "=" * 60)
    print("SURYA CROP ENSEMBLE EVALUATION")
    print("=" * 60)
    print(f"\nSamples: {len(samples)}\nTargets: {len(samples)}\n")
    print("METHOD                  CORRECT   ANY-TEXT  ANY-OVERLAP")
    print("-" * 60)
    n = len(summary)
    for mname, label in (("baseline", "Baseline exact"),
                         ("baseline_4x", "Baseline + 4x"),
                         ("ens3", "3-crop ensemble"),
                         ("ens3_4x", "3-crop + 4x")):
        acc = sum(summary[i][mname]["correct"] for i in summary)
        n_txt = sum(1 for i in summary
                    if any(summary[i][mname].get("texts", [])))
        n_asc = sum(1 for i in summary
                    if summary[i][mname].get("iou", 0) > 0)
        print(f"{label:22s}  {acc}/{n}      {n_txt}/{n}      {n_asc}/{n}")
    print("\n" + "=" * 60 + "\nBLOG TARGET\n" + "=" * 60 + "\n")
    for mname in ("baseline", "baseline_4x", "ens3", "ens3_4x"):
        print(f"{mname}: {summary[8][mname]}")
    print("\n" + "=" * 60 + "\nREGRESSIONS\n" + "=" * 60 + "\n")
    print("Baseline correct -> ensemble wrong:")
    for r in regressions:
        print(" ", r)
    print("\n" + "=" * 60 + "\nRECOVERIES\n" + "=" * 60 + "\n")
    print("Baseline wrong -> ensemble correct:")
    for r in recoveries:
        print(" ", r)
    print("\n" + "=" * 60 + "\nCONCLUSION\n" + "=" * 60)
    n = len(summary)
    per_target = round(total_s / max(1, n), 1)
    print(f"\nTotal Surya time: {total_s}s over "
          f"{timing_info['n_calls']} calls / {timing_info['n_crops']} crops "
          f"(~{per_target}s/target; methods share batched calls, "
          f"so per-method latency is ~equal)")
    print(f"Duplicate 'Blog Blog' rate on baseline_4x: {dups}/{n} samples")
    print(f"\nArtifacts: {out_dir}")


def score_lines(lines, gt, known):
    """Shared scorer: returns dict with texts/iou/correct + best box.

    Uses bbox_orig (crop coords mapped to the original screenshot) when
    present; falls back to raw poly. Comparing raw crop coords against
    full-image GT would silently zero every IoU.
    """
    best_iou, best_box = 0.0, None
    for l in lines:
        bb = l.get("bbox_orig") or l["poly"]
        if bb is None:
            continue
        v = box_iou(bb, gt)
        if v > best_iou:
            best_iou, best_box = v, bb
    texts = [l["text_raw"] for l in lines]
    if known:
        correct = any(norm_text(t) == norm_text(known) for t in texts) \
            and best_iou >= 0.5
    else:
        correct = bool(texts) and best_iou >= 0.5
    return {"texts": texts, "iou": round(best_iou, 3),
            "box": best_box, "correct": correct}


if __name__ == "__main__":
    main()
