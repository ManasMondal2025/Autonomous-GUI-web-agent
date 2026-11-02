"""Coordinate grounder: predicts raw (x, y) pixel coordinates.

This is the BASELINE. The VLM receives the raw screenshot and a natural
language instruction, and must predict pixel coordinates for the target.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image

from .base_grounder import BaseGrounder

logger = logging.getLogger(__name__)


class CoordinateGrounder(BaseGrounder):
    """Grounder that asks the VLM for raw pixel coordinates.

    Prediction format:
        {
            "action": "click" | "type" | "press" | "scroll" | "wait" | "ask_user" | "done" | "fail",
            "x": int, "y": int,
            "value": str (for type),
            "question": str (for ask_user only),
            "reason": str,
        }
    """

    SUPPORTED_ACTIONS = {"click", "type", "press", "scroll", "wait", "ask_user", "done", "fail"}

    def __init__(
        self,
        vlm: Optional[Any] = None,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        max_retries: int = 2,
    ):
        super().__init__(name="coordinate")
        self.vlm = vlm
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.max_retries = max_retries

    def ground(
        self,
        screenshot: Image.Image,
        instruction: str,
        dom_elements: Optional[List[Dict[str, Any]]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Predict an action and pixel coordinates from a screenshot.

        If no VLM is attached, returns a safe 'fail' prediction.
        """
        if self.vlm is None:
            logger.warning("No VLM attached to CoordinateGrounder; returning fail")
            return {
                "action": "fail",
                "reason": "No VLM configured",
                "method": self.name,
            }

        prompt = self._build_prompt(instruction, history)
        for attempt in range(self.max_retries + 1):
            try:
                raw_response = self.vlm.generate(screenshot, prompt)
                # The VLM may downscale large inputs: parse in the fed-image
                # space, then map coordinates back to the caller's image.
                fed_size = getattr(self.vlm, "last_input_size", None) \
                    or screenshot.size
                orig_size = getattr(self.vlm, "last_original_size", None) \
                    or screenshot.size
                prediction = self._parse_response(raw_response, fed_size)
                if self._validate_prediction(prediction, fed_size):
                    sx = orig_size[0] / fed_size[0]
                    sy = orig_size[1] / fed_size[1]
                    if "x" in prediction and isinstance(
                            prediction["x"], (int, float)):
                        prediction["x"] = prediction["x"] * sx
                    if "y" in prediction and isinstance(
                            prediction["y"], (int, float)):
                        prediction["y"] = prediction["y"] * sy
                    prediction["method"] = self.name
                    return prediction
                logger.warning(
                    f"Invalid prediction on attempt {attempt + 1}: {prediction}"
                )
            except Exception as e:
                logger.error(f"VLM call failed on attempt {attempt + 1}: {e}")

        return {
            "action": "fail",
            "reason": "Failed to get a valid coordinate prediction after retries",
            "method": self.name,
        }

    def to_action(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a coordinate prediction to an executable action dict."""
        action = {
            "action_type": prediction.get("action", "fail"),
            "x": prediction.get("x"),
            "y": prediction.get("y"),
            "value": prediction.get("value"),
            "key": prediction.get("key"),
            "target_id": None,
            "question": prediction.get("question"),
            "reason": prediction.get("reason", ""),
            "confidence": prediction.get("confidence"),
            "method": self.name,
            "raw_prediction": prediction,
        }
        return action

    def _build_prompt(
        self, instruction: str, history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Construct the prompt for the VLM."""
        history_text = ""
        if history:
            qa_lines = []
            for h in history:
                ans = h.get("human_answer") or ""
                if not ans and "Answer:" in str(h.get("reason", "")):
                    ans = str(h.get("reason", "")).split("Answer:", 1)[-1].strip()
                if ans:
                    q = h.get("question") or ""
                    qa_lines.append(f"  - User provided: \"{ans}\"" + (f" (for question: \"{q}\")" if q else ""))

            recent = history[-5:]
            recent_lines = []
            for h in recent:
                act = h.get("action", "?")
                step_num = h.get("step", "?")
                if act == "ask_user":
                    ans = h.get("human_answer") or ""
                    if not ans and "Answer:" in str(h.get("reason", "")):
                        ans = str(h.get("reason", "")).split("Answer:", 1)[-1].strip()
                    recent_lines.append(f"  Step {step_num}: ask_user → User answered: \"{ans}\"")
                elif act == "type":
                    recent_lines.append(f"  Step {step_num}: action=type at {h.get('coordinates')} value={h.get('value', '')!r} → {h.get('result', '?')}")
                else:
                    coords = h.get("coordinates")
                    coord_str = f" at {coords}" if coords else ""
                    recent_lines.append(f"  Step {step_num}: action={act}{coord_str} → {h.get('result', '?')}")

            if qa_lines:
                history_text += "\n\nInformation provided by user:\n" + "\n".join(qa_lines)
            if recent_lines:
                history_text += "\n\nRecent actions:\n" + "\n".join(recent_lines)

        return f"""You are a highly capable web automation agent. Your objective is to achieve the user's goal by navigating and interacting with web pages. You will be given the current URL, the user's goal, and a screenshot of the web page.

Task: {instruction}
Viewport: {self.viewport_width} x {self.viewport_height} pixels.{history_text}

        AVAILABLE ACTIONS:
- `{{"action": "click", "x": <pixel_x_integer>, "y": <pixel_y_integer>}}`: Click at coordinates (SINGLE integers, never a list or range).
- `{{"action": "type", "x": <pixel_x_integer>, "y": <pixel_y_integer>, "value": "<string>"}}`: Type text at coordinates.
- `{{"action": "press", "key": "Enter"}}`: Press a key on the focused element (key is Enter, Tab, or Escape; default Enter). Use this to submit a search/form AFTER typing.
- `{{"action": "scroll"}}`: Scroll the page.
- `{{"action": "done"}}`: The user's overall goal is successfully completed.
- `{{"action": "ask_user", "question": "<string>"}}`: Stop and ask the human for missing information (no coordinates needed).

SUBMIT RULE: after typing a query into a search box, your next action MUST be `press` (Enter) or a click on a suggestion/search button. NEVER type the same text twice: if the text is already in the box, submit it instead of re-typing.
NAVIGATION RULE: if you click a navigation link, menu item, or tab and the page does not change, it is already active. NEVER click the same menu item or tab again; proceed directly to the next step or content area of your task.

CRITICAL RULE (FATAL ERROR):
Guessing passwords, user preferences, or making up fake personal data is a FATAL ERROR. If you are on a login screen, checkout page, or face an ambiguous choice and do not have explicit data for a required field in your history, you MUST output the `ask_user` action immediately. NEVER invent data or use placeholders.

PRE-ACTION GROUNDING CHECK (CRITICAL):
Before you output a `click` or `type` action, you must internally verify:
1. Did the user explicitly provide the exact value/option I am about to select?
2. Is there only one logical path forward?
If the answer to either is NO (e.g., there are multiple sizes/colors and the user didn't specify, or a required search field lacks parameters), you are FORBIDDEN from guessing. You MUST output `ask_user`.

### EXAMPLES OF WHEN TO ASK THE USER

Example 1 (Missing Credentials):
- Goal: "Log in to the portal"
- Current State: Username and Password fields are visible.
- Action: `{{"action": "ask_user", "question": "I am at the login screen. Please provide the username and password."}}`

Example 2 (Ambiguous Task):
- Goal: "Buy the UltraBook Laptop"
- Current State: Page shows Color options [Silver, Black].
- Action: `{{"action": "ask_user", "question": "The UltraBook comes in Silver or Black. Which color should I select?"}}`

Example 3 (Ambiguous Search/Form):
- Goal: "Search for a flight to New York."
- Current State: The search form requires a "Departure Date", but the user prompt did not provide one.
- Action: `{{"action": "ask_user", "question": "What date would you like to depart for New York?"}}`

Output ONLY a single JSON object representing your next action."""

    def _parse_response(self, response: str,
                          image_size=None) -> Dict[str, Any]:
        """Parse VLM response: JSON first, Qwen <points> fallback."""
        from .grounding_utils import parse_vlm_json, parse_qwen_points

        try:
            return parse_vlm_json(response)
        except Exception:
            pass
        if image_size is not None:
            fallback = parse_qwen_points(response, image_size)
            if fallback is not None:
                logger.info("Parsed Qwen-native <points> format")
                return fallback
        raise ValueError(f"Could not parse VLM response: {response[:200]}")

    def _validate_prediction(
        self, prediction: Dict[str, Any], image_size: Tuple[int, int],
    ) -> bool:
        """Validate that a prediction has the right structure and coordinates in range."""
        action = prediction.get("action")
        if action not in self.SUPPORTED_ACTIONS:
            return False
        if action == "ask_user":
            q = prediction.get("question")
            return isinstance(q, str) and bool(q.strip())
        if action in ("click", "type", "scroll"):
            x = prediction.get("x")
            y = prediction.get("y")
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                return False
            img_w, img_h = image_size
            if not (0 <= x < img_w and 0 <= y < img_h):
                return False
        if action == "type" and "value" not in prediction:
            return False
        return True
