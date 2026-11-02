"""Grounding utilities shared by grounders: coercion + Qwen-native parsing.

Real VLMs rarely obey JSON schemas perfectly:
  - target_id may come back as "10" (string) or [10] (single-element list)
  - Qwen2.5-VL grounding models emit <points x1=".." y1=".."> in 0-1000 space
These helpers normalize such outputs instead of rejecting them.
"""

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def coerce_int(value: Any) -> Optional[int]:
    """Coerce int-like values (int, '10', [10], (10,)) to int, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        s = value.strip().strip('"').strip("'")
        try:
            return int(float(s)) if s else None
        except ValueError:
            return None
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return coerce_int(value[0])
    return None


def coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return coerce_float(value[0])
    return None


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def coerce_point(x: Any, y: Any) -> Tuple[Optional[float], Optional[float]]:
    """Coerce an (x, y) pair from messy model outputs.

    Handles: plain numbers, numeric strings, single-element lists, and the
    observed quirk where the model packs both coords into x as a 2-list
    (x=[453,605] with y missing, or x == y == [a,b]).
    Returns (None, None) when no unambiguous reading exists.
    """
    if _is_num(x) and _is_num(y):
        return float(x), float(y)

    def as_pair(v):
        if isinstance(v, (list, tuple)) and len(v) == 2 \
                and _is_num(v[0]) and _is_num(v[1]):
            return float(v[0]), float(v[1])
        return None

    px, py = as_pair(x), as_pair(y)
    if px is not None and (y is None or py == px):
        return px
    if py is not None and (x is None or px == py):
        return py

    fx, fy = coerce_float(x), coerce_float(y)
    if fx is not None and fy is not None:
        return fx, fy
    return None, None


ACTION_SYNONYMS = {
    "left_click": "click",
    "single_click": "click",
    "leftclick": "click",
    "tap": "click",
    "keypress": "type",
    "input": "type",
    "scroll_down": "scroll",
    "scroll_up": "scroll",
    "finish": "done",
    "complete": "done",
    "give_up": "fail",
    "abort": "fail",
}


def normalize_action_name(action: Any) -> Any:
    if isinstance(action, str):
        a = action.strip().lower()
        return ACTION_SYNONYMS.get(a, a)
    return action


def normalize_prediction(pred: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce common fields of a grounder prediction in place."""
    # Accept the `action_type` alias (some prompts/examples use it).
    if "action" not in pred and "action_type" in pred:
        pred["action"] = pred.pop("action_type")
    if "action" in pred:
        pred["action"] = normalize_action_name(pred["action"])
    # Accept the `text` alias for a type payload (some prompts use it).
    if "value" not in pred and isinstance(pred.get("text"), str):
        pred["value"] = pred.pop("text")
    if "target_id" in pred:
        coerced = coerce_int(pred["target_id"])
        if coerced is not None:
            pred["target_id"] = coerced
    # Qwen-style coordinates / point / bbox aliases
    for coord_key in ("coordinate", "coordinates", "point", "coords"):
        if coord_key in pred and ("x" not in pred or "y" not in pred):
            val = pred.pop(coord_key)
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                pred["x"], pred["y"] = val[0], val[1]
                break
    if "bbox" in pred and ("x" not in pred or "y" not in pred):
        b = pred.pop("bbox")
        if isinstance(b, (list, tuple)) and len(b) == 4:
            # Center of [x1, y1, x2, y2]
            pred["x"] = (b[0] + b[2]) / 2.0
            pred["y"] = (b[1] + b[3]) / 2.0
    if "x" in pred or "y" in pred:
        cx, cy = coerce_point(pred.get("x"), pred.get("y"))
        if cx is not None and cy is not None:
            pred["x"], pred["y"] = cx, cy
    if "value" in pred and pred["value"] is not None \
            and not isinstance(pred["value"], str):
        pred["value"] = str(pred["value"])
    return pred


_QWEN_POINTS_RE = re.compile(
    r"<points[^>]*x1?\s*=\s*\"?(\d+(?:\.\d+)?)\"?[^>]*y1?\s*=\s*\"?(\d+(?:\.\d+)?)\"?",
    re.IGNORECASE,
)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_vlm_json(response: str) -> Dict[str, Any]:
    """Extract a JSON object from raw VLM text (fences tolerated).

    Uses incremental `raw_decode` from each `{` so multi-object chatter
    (e.g. an echoed example plus the answer, common with long prompts)
    yields the first parseable object instead of an "Extra data" error.
    """
    text = response.strip()
    fence = _FENCE_RE.search(text)
    candidates = [fence.group(1)] if fence else []
    if not candidates:
        brace = _JSON_BLOCK_RE.search(text)
        if brace:
            candidates.append(brace.group(0))
        else:
            candidates.append(text)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        start = candidate.find("{")
        while start != -1:
            try:
                data, _ = decoder.raw_decode(candidate[start:])
            except json.JSONDecodeError:
                start = candidate.find("{", start + 1)
                continue
            if isinstance(data, dict):
                return normalize_prediction(data)
            start = candidate.find("{", start + 1)
    raise ValueError(f"Could not parse VLM response: {response[:200]}")


def parse_qwen_points(response: str,
                      image_size: Tuple[int, int]) -> Optional[Dict[str, Any]]:
    """Fallback: Qwen native <points x1= y1=> in 0-1000 space → pixel coords."""
    m = _QWEN_POINTS_RE.search(response)
    if not m:
        return None
    img_w, img_h = image_size
    x = float(m.group(1)) / 1000.0 * img_w
    y = float(m.group(2)) / 1000.0 * img_h
    return {
        "action": "click",
        "x": int(round(x)),
        "y": int(round(y)),
        "reason": "parsed from Qwen <points> format",
    }
