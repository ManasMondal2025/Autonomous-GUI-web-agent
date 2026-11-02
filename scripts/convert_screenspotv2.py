#!/usr/bin/env python3
"""Convert ScreenSpot-v2 (original JSON + image dir) to our eval JSONL.

Source layout:
  <src>/screenspot_{desktop,mobile,web}_v2.json
  <src>/screenspotv2_image/<img_filename>
Annotation bbox is [x, y, w, h] in absolute screenshot pixels → converted to
[x1, y1, x2, y2]. img_size is read from the actual image file.

Example:
  python scripts/convert_screenspotv2.py \\
      --src /Users/manasmondal/Desktop/GUI-Web-Agent/dataset/ScreenSpot-v2 \\
      --out data/screenspot/v2_full.jsonl
  python scripts/convert_screenspotv2.py --src ... --out data/screenspot/v2_mini.jsonl \\
      --limit-per-platform 4 --seed 42
"""

import logging  # stdlib first
import sys
import json
import argparse
import random
from pathlib import Path
from PIL import Image

PLATFORMS = ("desktop", "mobile", "web")


def convert(src: Path, limit_per_platform=None, seed=42):
    rng = random.Random(seed)
    samples = []
    for platform in PLATFORMS:
        anns = json.load(open(src / f"screenspot_{platform}_v2.json"))
        rng.shuffle(anns)
        if limit_per_platform:
            anns = anns[:limit_per_platform]
        for a in anns:
            img_path = src / "screenspotv2_image" / a["img_filename"]
            try:
                w, h = Image.open(img_path).size
            except Exception as e:
                logging.warning(f"Skipping {a['img_filename']}: {e}")
                continue
            x, y, bw, bh = a["bbox"]
            samples.append({
                "img_path": str(img_path),
                "instruction": a["instruction"],
                "bbox": [x, y, x + bw, y + bh],
                "platform": platform,
                "elem_type": a.get("data_type", "unknown"),
                "data_source": a.get("data_source", ""),
                "img_size": [w, h],
            })
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit-per-platform", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    samples = convert(Path(args.src), args.limit_per_platform, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    by_plat = {}
    for s in samples:
        by_plat[s["platform"]] = by_plat.get(s["platform"], 0) + 1
    print(f"Wrote {len(samples)} samples to {out} {by_plat}")


if __name__ == "__main__":
    main()
