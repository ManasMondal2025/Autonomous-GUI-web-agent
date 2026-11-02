#!/usr/bin/env python3
"""Evaluate a grounder on a ScreenSpot-format JSONL dataset.

Grounding is selected with ONE high-level flag:

  --method {coordinate,som,som_dom}

NOTE: cascade_* methods are rejected here UNLESS --ocr is given: static
ScreenSpot images provide no live browser DOM for DOM-first proposals.
With --ocr, OmniParser detection + OCR enrichment supply visual proposals
instead (Florence stays disabled); without it we fail clearly rather than
fabricating proposals. --method som_no_dom is image-only by definition and
always takes the visual path (YOLO detect + OCR enrich, DOM never read).

Example:
  python scripts/evaluate_screenspot.py --data data/screenspot/sample.jsonl \\
      --data-root data/screenspot --method coordinate --vlm mock \\
      --mock-coords 640,360 --output experiments/baseline_coordinate/
"""

import logging  # stdlib first
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.screenspot_eval import ScreenSpotEvaluator, load_samples
from grounding.methods import resolve_method, build_grounder, add_method_argument
from models.model_factory import get_vlm


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--data-root", default=".")
    # Default coordinate preserves the historical default.
    add_method_argument(p, default="coordinate")
    p.add_argument("--vlm", default="mock")
    p.add_argument("--model-path", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mock-coords", default="640,360",
                   help="Fixed point the mock coordinate VLM returns (for smoke tests)")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--ocr", action="store_true",
                   help="For cascade_* methods: detect OmniParser proposals and "
                        "enrich with PaddleOCR text (Florence stays disabled). "
                        "Requires paddleocr + paddlepaddle.")
    p.add_argument("--ocr-recover", action="store_true",
                   help="With --ocr: validate OCR boxes by edge support and "
                        "relocalize failures before association (experiment).")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N samples (default: all).")
    p.add_argument("--output", default="./experiments/eval_out")
    return p


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    method = resolve_method(args.method)
    if method["proposer"] == "cascade" and not args.ocr:
        p.error(
            f"--method {args.method} requires DOM-first proposals from a live "
            "browser page, but static ScreenSpot images provide none. Re-run "
            "with --ocr for the OmniParser+OCR visual cascade, or use "
            "--method coordinate/som/som_dom/som_no_dom here."
        )
    mx, my = (int(v) for v in args.mock_coords.split(","))

    if method["proposer"] == "hybrid":
        # Static images have no live DOM page: HybridProposer runs DOM-empty
        # plus the real GPA pipeline (+OCR text), i.e. the validated
        # hybrid_som visual path. Proposals are pre-injected per sample so
        # the shared evaluator flow below is untouched.
        from browser.proposers import HybridProposer
        from PIL import Image as _Image
        vlm = get_vlm(args.vlm, model_path=args.model_path, device=args.device,
                      max_new_tokens=args.max_tokens) \
            if args.vlm != "mock" else get_vlm(
                "mock", responses=[
                    json.dumps({"action": "click", "target_id": 0,
                                "reason": "smoke"})] * 10000)
        grounder = build_grounder(method, vlm)
        proposer = HybridProposer()
    elif method["proposer"] == "visual" or args.ocr:
        # Image-only path (existing methods untouched): OmniParser
        # proposals + OCR enrichment, Florence disabled. Works with any vlm.
        # `som_no_dom` always takes this path (it never touches DOM);
        # cascade_* take it only with --ocr.
        from grounding.ocr_grounder import OCRSoMGrounder
        vlm = get_vlm(args.vlm, model_path=args.model_path, device=args.device,
                      max_new_tokens=args.max_tokens) \
            if args.vlm != "mock" else get_vlm(
                "mock", responses=[
                    json.dumps({"action": "click", "target_id": 0,
                                "reason": "smoke"})] * 10000)
        grounder = OCRSoMGrounder(
            vlm, include_dom_in_prompt=method["include_dom_in_prompt"],
            recover_text_geometry=args.ocr_recover)
    elif args.vlm == "mock" and method["grounding"] == "coordinate":
        vlm = get_vlm("mock", responses=[
            json.dumps({"action": "click", "x": mx, "y": my, "reason": "smoke"})] * 10000)
        grounder = build_grounder(method, vlm)
    elif args.vlm == "mock" and method["grounding"] == "som":
        # mock SoM: always pick marker 0
        vlm = get_vlm("mock", responses=[
            json.dumps({"action": "click", "target_id": 0, "reason": "smoke"})] * 10000)
        grounder = build_grounder(method, vlm)
    else:
        vlm = get_vlm(args.vlm, model_path=args.model_path, device=args.device,
                      max_new_tokens=args.max_tokens)
        grounder = build_grounder(method, vlm)

    samples = load_samples(Path(args.data))
    if args.limit is not None:
        samples = samples[:max(0, args.limit)]
    if method["proposer"] == "hybrid":
        for s in samples:
            img = _Image.open(
                Path(args.data_root) / s["img_path"]
                if not Path(s["img_path"]).is_absolute()
                else Path(s["img_path"])).convert("RGB")
            s["dom_elements"] = proposer.propose(None, img, "", {})
    evaluator = ScreenSpotEvaluator(grounder, output_dir=Path(args.output))
    report = evaluator.evaluate(samples, data_root=Path(args.data_root))
    report["method_config"] = {**method, "ocr": bool(args.ocr),
                                 "ocr_recover": bool(args.ocr_recover)}
    if Path(args.output).exists():
        with open(Path(args.output) / "screenspot_report.json", "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
