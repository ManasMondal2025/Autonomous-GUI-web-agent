#!/usr/bin/env python3
"""Oracle-instruction grounding probe: hand-written instructions straight
into GUI-Actor, scored vs GT. Isolates grounder quality from planner
quality (no planner model involved). GT boxes + URLs come from the tasks
file (machine-specific, re-measured per machine); instruction lists come
from a separate oracle file (portable, pre-registered).

Example:
    python scripts/oracle_ground.py --tasks data/ab_tasks.json \\
        --only t04,t07 --oracle data/oracle_instructions.json \\
        --guiactor-path <GUI-Actor-3B> --device cuda \\
        --output ./experiments/oracle_t04_t07
"""

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

VIEWPORT = (1280, 720)
SETTLE_S = 1.5

log = logging.getLogger("oracle_ground")


def resolve_url(rel: str) -> str:
    if rel.startswith("file://") and not rel.startswith("file:///"):
        return f"file://{(Path.cwd() / rel[len('file://'):]).resolve()}"
    return rel


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Oracle instruction grounding.")
    ap.add_argument("--tasks", default="data/ab_tasks.json")
    ap.add_argument("--only", default=None,
                    help="comma list; task matches if id contains the token")
    ap.add_argument("--oracle", default="data/oracle_instructions.json")
    ap.add_argument("--guiactor-path", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="./experiments/oracle_out")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)

    from PIL import Image
    from models.model_factory import get_vlm
    from browser.browser_manager import BrowserManager

    tasks = json.load(open(args.tasks))
    oracle = json.load(open(args.oracle))
    if args.only:
        toks = [t.strip() for t in args.only.split(",") if t.strip()]
        tasks = [t for t in tasks
                 if any(tok in t["task_id"] for tok in toks)]
    if not tasks:
        log.error("No tasks selected.")
        return 2

    grounder = get_vlm("gui-actor", model_path=args.guiactor_path,
                       device=args.device)
    browser = BrowserManager(headless=True,
                             viewport_width=VIEWPORT[0],
                             viewport_height=VIEWPORT[1])
    browser.launch()
    rows = []
    try:
        for t in tasks:
            browser.new_page()
            browser.navigate(resolve_url(t["start_url"]))
            browser.get_page().wait_for_timeout(int(SETTLE_S * 1000))
            shot = browser.screenshot()
            image = Image.open(io.BytesIO(shot)).convert("RGB")
            for ins in oracle.get(t["task_id"], []):
                t0 = time.perf_counter()
                try:
                    d = json.loads(grounder.generate(image, ins))
                    x, y = d.get("x"), d.get("y")
                except Exception as e:
                    x, y = None, None
                    log.warning("grounder failed: %s", e)
                dt = round(time.perf_counter() - t0, 2)
                gt = t["gt"]
                hit = isinstance(x, (int, float)) and isinstance(
                    y, (int, float)) and gt[0] <= x <= gt[2] \
                    and gt[1] <= y <= gt[3]
                rows.append({"task_id": t["task_id"],
                             "instruction": ins,
                             "coord": [x, y], "gt": gt,
                             "hit": bool(hit), "latency_s": dt})
                print(f"{t['task_id']} | {ins[:60]:60s} | "
                      f"({x},{y}) -> {'HIT' if hit else 'MISS'} ({dt}s)",
                      flush=True)
    finally:
        browser.close()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    hits = sum(1 for r in rows if r["hit"])
    report = {"n": len(rows), "hits": hits,
              "accuracy": round(hits / max(1, len(rows)), 4),
              "rows": rows}
    (out / "oracle_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("n", "hits", "accuracy")},
                     indent=2))
    return 0


if __name__ == "__main__":
    main()
