#!/usr/bin/env python3
"""A/B leg C: Qwen planner -> GUI-Actor grounder on ScreenSpot JSONL.

For each sample: planner VLM (screenshot + task -> JSON
{action, instruction, value}), then grounder VLM (instruction -> x, y),
scored point-in-bbox vs GT. Parse failures count as misses (no retries:
this measures planner JSON discipline as well as grounding).

Example:
    python scripts/evaluate_plan_ground.py --data data/screenspot/v2_mini.jsonl \\
        --planner-path <Qwen3B> --grounder-path <GUI-Actor-3B> \\
        --device cuda --output experiments/plan_ground_mini
"""

import logging  # stdlib first
import sys
import json
import argparse
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.screenspot_eval import load_samples
from evaluation.metrics import (
    point_in_bbox, click_accuracy, mean_distance_error,
    mean_normalized_distance, breakdown_accuracy)
from models.model_factory import get_vlm

PLANNER_PROMPT = (
    "You are the planner of a GUI agent. Look at the screenshot. "
    "User query: {query}. Decide the single next action. Reply with ONLY "
    'a JSON object: {{"action": "click"|"type"|"scroll"|"wait"|"done"|"fail", '
    '"instruction": "<short imperative referring expression naming the '
    'visible target>", "value": "<text to type, or null>"}}. No other text.'
)


def parse_plan(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None, "no_json_object"
    try:
        plan = json.loads(m.group(0))
    except Exception:
        return None, "json_parse_error"
    if not isinstance(plan, dict) or not plan.get("instruction"):
        return None, "missing_instruction"
    return plan, None


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--data-root", default=".")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--planner", default="qwen2.5-vl")
    p.add_argument("--planner-path", required=True)
    p.add_argument("--grounder", default="gui-actor")
    p.add_argument("--grounder-path", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--output", default="./experiments/plan_ground_out")
    return p


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    from PIL import Image
    planner = get_vlm(args.planner, model_path=args.planner_path,
                      device=args.device, max_new_tokens=args.max_tokens)
    grounder = get_vlm(args.grounder, model_path=args.grounder_path,
                       device=args.device, max_new_tokens=256)

    samples = load_samples(Path(args.data))
    if args.limit is not None:
        samples = samples[:max(0, args.limit)]

    predictions, gts, records = [], [], []
    for i, s in enumerate(samples):
        img = Image.open(Path(args.data_root) / s["img_path"] \
            if not Path(s["img_path"]).is_absolute()
            else Path(s["img_path"])).convert("RGB")
        instruction, gt = s["instruction"], s["bbox"]
        stage, plan_out, x, y = "planner", None, None, None
        try:
            plan_raw = planner.generate(
                img, PLANNER_PROMPT.format(query=instruction))
            plan_out, err = parse_plan(plan_raw)
            if err:
                stage = f"planner_fail:{err}"
            else:
                g_out = grounder.generate(img, plan_out["instruction"])
                try:
                    d = json.loads(g_out)
                except Exception:
                    d = {}
                x, y = d.get("x"), d.get("y")
                stage = "grounded" if isinstance(
                    x, (int, float)) and isinstance(y, (int, float)) \
                    else "grounding_fail"
        except Exception as e:
            stage = f"error:{type(e).__name__}"
            logging.warning("Sample %d failed: %s", i, e)
        ok = (x is not None and y is not None
              and point_in_bbox(x, y, gt))
        predictions.append((x if x is not None else -1,
                            y if y is not None else -1))
        gts.append(gt)
        records.append({"index": i, "instruction": instruction,
                        "platform": s.get("platform", "unknown"),
                        "elem_type": s.get("elem_type", "unknown"),
                        "plan": plan_out, "stage": stage,
                        "predicted": [x, y], "ground_truth": gt,
                        "correct": bool(ok)})

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_size = tuple(samples[0].get("img_size", [1280, 720])) \
        if samples else (1280, 720)
    report = {
        "method": "plan_ground_split",
        "planner": args.planner, "grounder": args.grounder,
        "n": len(samples),
        "click_accuracy": round(click_accuracy(predictions, gts), 4),
        "mean_distance_error": round(mean_distance_error(predictions, gts), 2),
        "mean_normalized_distance": round(
            mean_normalized_distance(predictions, gts, img_size), 4),
        "breakdown": breakdown_accuracy(records),
    }
    with open(out_dir / "screenspot_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
