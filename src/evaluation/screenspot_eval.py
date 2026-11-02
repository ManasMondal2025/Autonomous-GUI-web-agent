"""ScreenSpot evaluator.

Dataset layout (JSONL, one sample per line):
  {"img_path": "screenspot/mobile/001.png", "instruction": "click X",
   "bbox": [x1, y1, x2, y2], "platform": "mobile|desktop|web",
   "elem_type": "text|icon|widget", "img_size": [w, h]}

bbox coordinates are absolute pixels in the screenshot's own space.
The evaluator loads each image, runs the grounder, and checks whether the
predicted point falls inside the GT box. No results are fabricated — every
cell of the report table comes from real grounder outputs.

Usage:
  python scripts/evaluate_screenspot.py --data data/screenspot/sample.jsonl \
      --grounding som --vlm mock --output experiments/run1/
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image

from .metrics import (click_accuracy, mean_distance_error,
                      mean_normalized_distance, breakdown_accuracy)

logger = logging.getLogger(__name__)


class ScreenSpotEvaluator:
    def __init__(self, grounder, output_dir: Optional[Path] = None):
        self.grounder = grounder
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, samples: List[Dict[str, Any]],
                 data_root: Path = Path(".")) -> Dict[str, Any]:
        predictions, gts, records = [], [], []
        for i, s in enumerate(samples):
            img_path = Path(data_root) / s["img_path"]
            image = Image.open(img_path).convert("RGB")
            instruction = s["instruction"]
            dom_elements = s.get("dom_elements", [])

            try:
                pred = self.grounder.ground(image, instruction, dom_elements)
                if hasattr(self.grounder, "to_action"):
                    action = self.grounder.to_action(pred)
                    x, y = action.get("x"), action.get("y")
                else:
                    x, y = pred.get("x"), pred.get("y")
                if x is None or y is None:
                    raise ValueError("Grounder produced no coordinates")
            except Exception as e:
                logger.warning(f"Sample {i} grounding failed: {e}")
                x, y = -1, -1

            from .metrics import point_in_bbox
            gt = s["bbox"]
            correct = point_in_bbox(x, y, gt)
            predictions.append((x, y))
            gts.append(gt)
            records.append({
                "index": i,
                "instruction": instruction,
                "platform": s.get("platform", "unknown"),
                "elem_type": s.get("elem_type", "unknown"),
                "predicted": [x, y],
                "ground_truth": gt,
                "correct": correct,
            })

        img_size = tuple(samples[0].get("img_size", [1280, 720])) if samples else (1280, 720)
        report = {
            "method": getattr(self.grounder, "name", "unknown"),
            "n": len(samples),
            "click_accuracy": round(click_accuracy(predictions, gts), 4),
            "mean_distance_error": round(mean_distance_error(predictions, gts), 2),
            "mean_normalized_distance": round(
                mean_normalized_distance(predictions, gts, img_size), 4),
            "breakdown": breakdown_accuracy(records),
        }
        if self.output_dir:
            with open(self.output_dir / "screenspot_report.json", "w") as f:
                json.dump(report, f, indent=2)
            with open(self.output_dir / "screenspot_records.jsonl", "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
        return report


def load_samples(jsonl_path: Path) -> List[Dict[str, Any]]:
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples
