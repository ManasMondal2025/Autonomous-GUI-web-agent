"""Coordinate transformation utilities."""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np


def clamp_coordinates(
    x: float, y: float, width: int, height: int
) -> Tuple[int, int]:
    """Clamp coordinates to be within the given width/height bounds."""
    x_clamped = max(0, min(int(round(x)), width - 1))
    y_clamped = max(0, min(int(round(y)), height - 1))
    return x_clamped, y_clamped


def point_in_bbox(
    x: float, y: float, bbox: Dict[str, float]
) -> bool:
    """Check if a point is inside a bounding box."""
    bx = bbox.get("x", 0)
    by = bbox.get("y", 0)
    bw = bbox.get("width", 0)
    bh = bbox.get("height", 0)
    return bx <= x <= bx + bw and by <= y <= by + bh


def bbox_center(bbox: Dict[str, float]) -> Tuple[float, float]:
    """Get the center of a bounding box."""
    x = bbox.get("x", 0) + bbox.get("width", 0) / 2
    y = bbox.get("y", 0) + bbox.get("height", 0) / 2
    return (x, y)


def bbox_area(bbox: Dict[str, float]) -> float:
    """Get the area of a bounding box."""
    return bbox.get("width", 0) * bbox.get("height", 0)


def bbox_iou(
    bbox1: Dict[str, float], bbox2: Dict[str, float]
) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    x1_1 = bbox1.get("x", 0)
    y1_1 = bbox1.get("y", 0)
    x2_1 = x1_1 + bbox1.get("width", 0)
    y2_1 = y1_1 + bbox1.get("height", 0)

    x1_2 = bbox2.get("x", 0)
    y1_2 = bbox2.get("y", 0)
    x2_2 = x1_2 + bbox2.get("width", 0)
    y2_2 = y1_2 + bbox2.get("height", 0)

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = bbox1_area + bbox2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def scale_coordinates(
    x: float, y: float,
    src_size: Tuple[int, int],
    dst_size: Tuple[int, int],
) -> Tuple[int, int]:
    """Scale coordinates from source size to destination size."""
    sx = dst_size[0] / src_size[0]
    sy = dst_size[1] / src_size[1]
    return int(x * sx), int(y * sy)


def compute_normalized_distance(
    pred: Tuple[float, float],
    target_bbox: Dict[str, float],
    img_size: Tuple[int, int],
) -> float:
    """Compute normalized distance from predicted point to target bbox center."""
    target_center = bbox_center(target_bbox)
    dx = pred[0] - target_center[0]
    dy = pred[1] - target_center[1]
    dist = np.sqrt(dx * dx + dy * dy)
    diag = np.sqrt(img_size[0] ** 2 + img_size[1] ** 2)
    if diag == 0:
        return 0.0
    return float(dist / diag)


def compute_distance(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
) -> float:
    """Compute Euclidean distance between two points."""
    return float(np.sqrt(
        (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
    ))


def denormalize_coordinates(
    x: float, y: float, width: int, height: int
) -> Tuple[int, int]:
    """Convert normalized [0,1] coordinates to pixel coordinates."""
    return int(x * width), int(y * height)


def normalize_coordinates(
    x: float, y: float, width: int, height: int
) -> Tuple[float, float]:
    """Convert pixel coordinates to normalized [0,1] coordinates."""
    return x / width, y / height
