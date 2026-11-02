#!/usr/bin/env python3
"""Rule-based failure analysis over eval records.

Example:
  python scripts/analyze_failures.py \\
      --input experiments/baseline_coordinate/screenspot_records.jsonl \\
      --output experiments/baseline_coordinate/
"""

import logging  # stdlib first, before src hits sys.path
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.failure_analysis import analyze_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    report = analyze_file(Path(args.input), Path(args.output))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
