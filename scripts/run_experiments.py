#!/usr/bin/env python3
"""Run the experiment matrix from configs/experiments.yaml on a ScreenSpot JSONL set.

For each experiment: build the grounder per config, evaluate, write
experiments/<name>/screenspot_report.json, then regenerate experiments/results.md
— populated ONLY from real report files. Missing/failed experiments render as
"n/a (not run)", never fabricated.

Example:
  python scripts/run_experiments.py --data data/screenspot/smoke/smoke.jsonl \\
      --data-root data/screenspot/smoke --vlm mock
"""

import logging  # stdlib first
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.config import load_yaml_config
from evaluation.screenspot_eval import ScreenSpotEvaluator, load_samples
from grounding.methods import (resolve_method, resolve_experiment_config,
                               add_method_argument)
from grounding.coordinate_grounder import CoordinateGrounder
from grounding.som_grounder import SoMGrounder
from models.model_factory import get_vlm

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "experiment_logger",
    str(Path(__file__).resolve().parents[1] / "src" / "logging" / "experiment_logger.py"))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ExperimentLogger = _mod.ExperimentLogger


def select_experiments(cfg, cli_method):
    """CLI --method runs a single method (overrides the YAML matrix);
    otherwise the full YAML matrix runs unchanged."""
    experiments = cfg.get("experiments", {})
    if cli_method is not None:
        resolve_method(cli_method)  # fail fast on typos
        return {cli_method: {"method": cli_method}}
    return experiments


def build_grounder(exp_cfg, vlm_name, model_path, device, mock_coords):
    grounding, include_dom = resolve_experiment_config(exp_cfg)
    if vlm_name == "mock":
        if grounding == "som":
            vlm = get_vlm("mock", responses=[
                json.dumps({"action": "click", "target_id": 0, "reason": "matrix"})] * 100000)
            return SoMGrounder(vlm=vlm, include_dom_in_prompt=include_dom), "mock"
        mx, my = (int(v) for v in mock_coords.split(","))
        vlm = get_vlm("mock", responses=[
            json.dumps({"action": "click", "x": mx, "y": my, "reason": "matrix"})] * 100000)
        return CoordinateGrounder(vlm=vlm), "mock"
    vlm = get_vlm(vlm_name, model_path=model_path, device=device)
    if grounding == "som":
        return SoMGrounder(vlm=vlm, include_dom_in_prompt=include_dom), vlm_name
    return CoordinateGrounder(vlm=vlm), vlm_name


def write_results_table(experiments_root: Path, exp_names) -> Path:
    """Build results.md from real report JSONs only."""
    header = ("| Method | Overall | Desktop | Mobile | Web | Text | Icon |\n"
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    rows = []
    for name in exp_names:
        report_path = experiments_root / name / "screenspot_report.json"
        if report_path.exists():
            try:
                rep = json.load(open(report_path))
                bd = rep.get("breakdown", {})
                def cell(k):
                    v = bd.get(k)
                    return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"
                rows.append(
                    f"| {name} | {rep.get('click_accuracy', 'n/a')} "
                    f"| {cell('platform:desktop')} | {cell('platform:mobile')} "
                    f"| {cell('platform:web')} | {cell('elem_type:text')} "
                    f"| {cell('elem_type:icon')} |")
            except Exception as e:
                rows.append(f"| {name} | error: {e} | n/a | n/a | n/a | n/a | n/a |")
        else:
            rows.append(f"| {name} | n/a (not run) | n/a | n/a | n/a | n/a | n/a |")
    out = experiments_root / "results.md"
    out.write_text("# Experiment results (auto-generated, real outputs only)\n\n"
                   + header + "\n" + "\n".join(rows) + "\n")
    return out


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--data-root", default=".")
    p.add_argument("--config", default="configs/experiments.yaml")
    p.add_argument("--method", default=None,
                   help="Run a single method instead of the YAML matrix "
                        "(overrides matrix grounding/proposer/DOM settings). "
                        f"Choices: coordinate, som, som_dom, cascade_som, "
                        "cascade_som_dom. NOTE: cascade_* need live-browser "
                        "proposals and fail on static ScreenSpot data.")
    p.add_argument("--vlm", default="mock")
    p.add_argument("--model-path", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mock-coords", default="640,360")
    p.add_argument("--experiments-root", default="./experiments/")
    return p


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    root = Path(__file__).resolve().parents[1]
    cfg = load_yaml_config(root / args.config)
    experiments = select_experiments(cfg, args.method)
    exp_root = Path(args.experiments_root)
    samples = load_samples(Path(args.data))

    for name, exp_cfg in experiments.items():
        out_dir = exp_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        ExperimentLogger(out_dir).log(
            {"seed": cfg.get("seed", 42),
             "model": {"name": args.vlm, "temperature": 0.0},
             "viewport": {"width": 1280, "height": 720},
             "dataset": {"path": args.data, "version": "smoke" if "smoke" in args.data else "screenspot"}},
            model_name=args.vlm,
            extra={"experiment": name, "experiment_config": exp_cfg})
        grounder, _ = build_grounder(exp_cfg, args.vlm, args.model_path,
                                     args.device, args.mock_coords)
        report = ScreenSpotEvaluator(grounder, output_dir=out_dir).evaluate(
            samples, data_root=Path(args.data_root))
        print(f"[{name}] accuracy={report['click_accuracy']} n={report['n']}")

    table = write_results_table(exp_root, list(experiments.keys()))
    print(f"Results table: {table}")


if __name__ == "__main__":
    main()
