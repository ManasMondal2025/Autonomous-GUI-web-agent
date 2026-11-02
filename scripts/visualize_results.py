#!/usr/bin/env python3
"""Generate plots from real experiment outputs (headless-safe, Agg backend).

Inputs: one or more screenspot_report.json + failure_report.json files.
Outputs: accuracy_comparison.png, platform_breakdown.png, type_breakdown.png,
         failure_distribution.png, and trajectory strips if --trajectory given.

Example:
  python scripts/visualize_results.py --reports experiments/*/screenspot_report.json \\
      --failures experiments/*/failure_report.json --output experiments/figures/
"""

import matplotlib
matplotlib.use("Agg")
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def _load_reports(patterns):
    out = {}
    for pat in patterns or []:
        for p in sorted(Path(".").glob(pat)):
            try:
                out[p.parent.name or p.stem] = json.load(open(p))
            except Exception:
                pass
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reports", nargs="*", default=[])
    p.add_argument("--failures", nargs="*", default=[])
    p.add_argument("--trajectory", default=None)
    p.add_argument("--output", default="./experiments/figures/")
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    reports = _load_reports(args.reports)
    made = []

    if reports:
        names = list(reports.keys())
        accs = [reports[n].get("click_accuracy", 0) for n in names]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(names, accs)
        ax.set_ylabel("Click accuracy")
        ax.set_title("Coordinate vs SoM (real outputs)")
        ax.set_ylim(0, max(1.0, max(accs + [0.1])))
        plt.xticks(rotation=15)
        fig.tight_layout()
        fig.savefig(out / "accuracy_comparison.png", dpi=150)
        plt.close(fig)
        made.append("accuracy_comparison.png")

        # platform + type breakdowns from each report's breakdown dict
        for key, fname, title in [
            ("platform:", "platform_breakdown.png", "Accuracy by platform"),
            ("elem_type:", "type_breakdown.png", "Accuracy by element type"),
        ]:
            cats = sorted({k.split(":", 1)[1] for r in reports.values()
                           for k in r.get("breakdown", {}) if k.startswith(key)})
            if not cats:
                continue
            x = range(len(cats))
            fig, ax = plt.subplots(figsize=(8, 4))
            for i, n in enumerate(names):
                bd = reports[n].get("breakdown", {})
                vals = [bd.get(f"{key}{c}", 0) for c in cats]
                ax.bar([v + i * 0.8 / len(names) for v in x], vals,
                       width=0.8 / len(names), label=n)
            ax.set_xticks(list(x))
            ax.set_xticklabels(cats)
            ax.set_ylabel("Accuracy")
            ax.set_title(title)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / fname, dpi=150)
            plt.close(fig)
            made.append(fname)

    failures = _load_reports(args.failures)
    if failures:
        name, rep = next(iter(failures.items()))
        cats = rep.get("by_category", {})
        cats = {k: v for k, v in cats.items() if v}
        if cats:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(list(cats.values()), labels=list(cats.keys()), autopct="%1.0f%%")
            ax.set_title(f"Failure distribution ({name})")
            fig.tight_layout()
            fig.savefig(out / "failure_distribution.png", dpi=150)
            plt.close(fig)
            made.append("failure_distribution.png")

    if args.trajectory:
        traj = Path(args.trajectory)
        steps = sorted(traj.glob("step_*.png"))
        if steps:
            from PIL import Image
            thumbs = [Image.open(s).convert("RGB").resize((320, 180)) for s in steps[:8]]
            strip = Image.new("RGB", (320 * len(thumbs), 180), "white")
            for i, t in enumerate(thumbs):
                strip.paste(t, (i * 320, 0))
            strip.save(out / "trajectory_strip.png")
            made.append("trajectory_strip.png")

    print(json.dumps({"output_dir": str(out), "figures": made}, indent=2))


if __name__ == "__main__":
    main()
