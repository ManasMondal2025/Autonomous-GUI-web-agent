"""Action parsing: raw VLM text → validated Action (no execution here)."""

import json
import logging
import re
from typing import Dict, Any

from .action_schema import Action, parse_action

logger = logging.getLogger(__name__)


def extract_json(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return brace.group(0)
    return text


def parse_vlm_response(response: str) -> Action:
    """Parse raw VLM text into a validated Action. Raises on malformed input."""
    data = json.loads(extract_json(response))
    if not isinstance(data, dict):
        raise ValueError("VLM response is not a JSON object")
    return parse_action(data)


def safe_parse_vlm_response(response: str) -> Dict[str, Any]:
    """Never-raising variant: returns {'ok': True, 'action': Action} or
    {'ok': False, 'error': str}."""
    try:
        return {"ok": True, "action": parse_vlm_response(response)}
    except Exception as e:
        logger.warning(f"Failed to parse VLM response: {e}")
        return {"ok": False, "error": str(e)}
