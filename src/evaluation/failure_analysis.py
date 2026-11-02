"""Failure taxonomy + statistics over eval records / trajectories.

Reads screenspot_records.jsonl, webarena_records.jsonl, or trajectory
actions.json files and buckets every failure into the 10-category taxonomy.
Writes failure_report.json + failure_report.md. No VLM needed (rule-based).
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter

from .metrics import FAILURE_CATEGORIES, classify_failure

logger = logging.getLogger(__name__)


def analyze_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    failures = [r for r in records if not r.get("correct", r.get("success", True))]
    cats = Counter()
    for r in records:
        if r.get("correct", r.get("success", True)):
            continue
        sample = {
            "error": r.get("error", ""),
            "verification_reason": str(r.get("verification", "")),
        }
        cats[classify_failure(sample)] += 1
    total = len(records)
    return {
        "n_total": total,
        "n_failed": len(failures),
        "failure_rate": round(len(failures) / max(total, 1), 4),
        "by_category": {c: cats.get(c, 0) for c in FAILURE_CATEGORIES},
    }


def analyze_file(path: Path, output_dir: Path) -> Dict[str, Any]:
    path, output_dir = Path(path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    if path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    elif path.suffix == ".json":
        data = json.load(open(path))
        records = data if isinstance(data, list) else data.get("records", data.get("actions", []))
    else:
        raise ValueError(f"Unsupported input: {path.suffix}")
    report = analyze_records(records)
    with open(output_dir / "failure_report.json", "w") as f:
        json.dump(report, f, indent=2)
    lines = ["# Failure analysis (rule-based taxonomy)\n",
             f"Total: {report['n_total']}, Failed: {report['n_failed']} "
             f"({report['failure_rate']*100:.1f}%)\n",
             "| Category | Count |", "| --- | ---: |"]
    for c in FAILURE_CATEGORIES:
        lines.append(f"| {c} | {report['by_category'][c]} |")
    (output_dir / "failure_report.md").write_text("\n".join(lines) + "\n")
    return report
