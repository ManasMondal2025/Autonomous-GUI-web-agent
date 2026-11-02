"""Grounding metrics: click accuracy, distance errors, breakdowns.

Click accuracy (ScreenSpot standard): predicted point inside GT bbox → correct.
"""

import logging
import math
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


def point_in_bbox(x: float, y: float, bbox: List[float]) -> bool:
    """bbox as [x1, y1, x2, y2] (absolute pixels)."""
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def click_accuracy(predictions: List[Tuple[float, float]],
                   ground_truths: List[List[float]]) -> float:
    if not ground_truths:
        return 0.0
    correct = sum(1 for (x, y), gt in zip(predictions, ground_truths)
                  if point_in_bbox(x, y, gt))
    return correct / len(ground_truths)


def mean_distance_error(predictions: List[Tuple[float, float]],
                        ground_truths: List[List[float]]) -> float:
    """Mean Euclidean distance from predicted point to GT box center."""
    if not ground_truths:
        return 0.0
    total = 0.0
    for (x, y), gt in zip(predictions, ground_truths):
        cx, cy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
        total += math.hypot(x - cx, y - cy)
    return total / len(ground_truths)


def mean_normalized_distance(predictions, ground_truths,
                             img_size: Tuple[int, int]) -> float:
    diag = math.hypot(img_size[0], img_size[1])
    if diag == 0 or not ground_truths:
        return 0.0
    return mean_distance_error(predictions, ground_truths) / diag


def breakdown_accuracy(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    """Per-category accuracy. Each sample: {correct: bool, platform, elem_type}."""
    result: Dict[str, float] = {}
    overall = sum(1 for s in samples if s.get("correct")) / max(len(samples), 1)
    result["overall"] = round(overall, 4)
    for key in ("platform", "elem_type"):
        groups: Dict[str, List[bool]] = {}
        for s in samples:
            groups.setdefault(str(s.get(key, "unknown")), []).append(bool(s.get("correct")))
        for g, vals in groups.items():
            result[f"{key}:{g}"] = round(sum(vals) / len(vals), 4)
    result["n"] = len(samples)
    return result


FAILURE_CATEGORIES = [
    "wrong_element_grounding",
    "correct_element_wrong_action",
    "incorrect_action_parameters",
    "planning_failure",
    "verification_failure",
    "recovery_failure",
    "infinite_loop",
    "browser_environment_failure",
    "timing_loading_failure",
    "task_impossible",
]


def classify_failure(sample: Dict[str, Any]) -> str:
    """Rule-based failure taxonomy from a trajectory record."""
    err = str(sample.get("error", "")).lower()
    ver = str(sample.get("verification_reason", "")).lower()
    if "loop" in err or "loop" in ver:
        return "infinite_loop"
    if "timeout" in err or "load" in err or "networkidle" in err:
        return "timing_loading_failure"
    if "browser" in err or "playwright" in err or "closed" in err:
        return "browser_environment_failure"
    if "invalid action" in err or "validation" in err:
        return "incorrect_action_parameters"
    if "no visual" in ver or "no state change" in ver or "missed" in ver:
        return "wrong_element_grounding"
    if "verification failed" in err:
        return "verification_failure"
    return "planning_failure"
