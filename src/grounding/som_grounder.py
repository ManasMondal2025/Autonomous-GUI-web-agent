"""SoM grounder: predicts a marker_id referencing a pre-annotated DOM element.

This is the PROPOSED method. The VLM receives a screenshot with numbered
markers overlaid on each interactive element, and only has to output a
discrete marker ID — not continuous coordinates.

The grounder also receives the structured DOM list, so it can:
    - Resolve marker_id → element_id → bbox
    - Compute click coordinates from the bbox center
    - (Optional) Pass DOM text/role hints to the VLM in the prompt
"""

import logging
from typing import Dict, Any, Optional, List
from PIL import Image

from .base_grounder import BaseGrounder
from .som_overlay import SoMOverlay, get_click_coordinates

logger = logging.getLogger(__name__)


class SoMGrounder(BaseGrounder):
    """Grounder using Set-of-Mark visual prompting.

    Prediction format:
        {
            "action": "click" | "type" | "press" | "scroll" | "wait" | "ask_user" | "done" | "fail",
            "target_id": int,
            "value": str (for type),
            "question": str (for ask_user only),
            "reason": str,
        }

    The grounder is responsible:
        1. Drawing the SoM overlay on the screenshot
        2. Asking the VLM for a marker_id
        3. Looking up the DOM element and computing click coordinates
    """

    SUPPORTED_ACTIONS = {"click", "type", "press", "scroll", "wait", "ask_user", "done", "fail"}

    def __init__(
        self,
        vlm: Optional[Any] = None,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        max_retries: int = 2,
        include_dom_in_prompt: bool = True,
        marker_size: int = 18,
    ):
        super().__init__(name="som")
        self.vlm = vlm
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.max_retries = max_retries
        self.include_dom_in_prompt = include_dom_in_prompt
        self.overlay_module = SoMOverlay(marker_size=marker_size)
        self._last_mapping: Optional[Dict[str, Dict[str, Any]]] = None
        self._last_annotated: Optional[Image.Image] = None

    @property
    def last_mapping(self) -> Optional[Dict[str, Dict[str, Any]]]:
        return self._last_mapping

    @property
    def last_annotated_screenshot(self) -> Optional[Image.Image]:
        return self._last_annotated

    def ground(
        self,
        screenshot: Image.Image,
        instruction: str,
        dom_elements: Optional[List[Dict[str, Any]]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Overlay markers, ask the VLM for an ID, resolve to coordinates."""
        if dom_elements is None:
            dom_elements = []

        annotated, mapping = self.overlay_module.overlay(screenshot, dom_elements)
        self._last_mapping = mapping
        self._last_annotated = annotated

        if not mapping:
            return {
                "action": "fail",
                "reason": "No interactive elements available to ground on",
                "method": self.name,
            }

        if self.vlm is None:
            return {
                "action": "fail",
                "reason": "No VLM configured",
                "method": self.name,
                "available_markers": list(mapping.keys()),
            }

        stale = self._stale_markers(mapping, history)
        dropdown_triggers = self._dropdown_trigger_markers(mapping, history)
        if stale:
            logger.info("[SOM] stale markers (clicked, no effect): %s",
                        sorted(stale, key=int))
        if dropdown_triggers:
            logger.info("[SOM] dropdown open trigger markers (do not re-click): %s",
                        sorted(dropdown_triggers, key=int))
        prompt = self._build_prompt(
            instruction, mapping, history,
            stale=stale, dropdown_triggers=dropdown_triggers,
        )
        fallback = None
        for attempt in range(self.max_retries + 1):
            try:
                raw_response = self.vlm.generate(annotated, prompt)
                prediction = self._parse_response(raw_response)
                if self._validate_prediction(prediction, mapping):
                    prediction["method"] = self.name
                    prediction["annotated_screenshot"] = annotated
                    prediction["mapping"] = mapping
                    tid = prediction.get("target_id")
                    if (prediction.get("action") == "click"
                            and tid is not None and str(tid) in stale):
                        # Already-active element: reject and re-ask once more.
                        fallback = fallback or prediction
                        logger.warning(
                            "[SOM] rejected stale marker %s on attempt %d",
                            tid, attempt + 1)
                        prompt += (
                            f"\n\nYour previous answer clicked marker [{tid}], "
                            "which was ALREADY clicked and had NO effect (it is "
                            "already active/open). That is FORBIDDEN. Pick a "
                            "DIFFERENT marker for the NEXT step of the task.")
                        continue
                    if (prediction.get("action") == "click"
                            and tid is not None and str(tid) in dropdown_triggers):
                        # Trigger button that just opened the dropdown: re-clicking closes it.
                        fallback = fallback or prediction
                        logger.warning(
                            "[SOM] rejected clicking dropdown trigger %s (would close open menu) on attempt %d",
                            tid, attempt + 1)
                        prompt += (
                            f"\n\nYour previous answer clicked marker [{tid}], "
                            "which is the TRIGGER BUTTON that just opened the dropdown menu. "
                            "Clicking it again will CLOSE the menu! That is FORBIDDEN. "
                            "Pick one of the NEW items INSIDE the open dropdown menu instead.")
                        continue
                    return prediction
                logger.warning(
                    f"Invalid SoM prediction on attempt {attempt + 1}: {prediction}"
                )
            except Exception as e:
                logger.error(f"VLM call failed on attempt {attempt + 1}: {e}")
        if fallback is not None:
            return fallback

        return {
            "action": "fail",
            "reason": "Failed to get a valid SoM prediction after retries",
            "method": self.name,
            "available_markers": list(mapping.keys()),
        }

    def to_action(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a marker_id prediction to click coordinates + action dict."""
        action_type = prediction.get("action", "fail")
        target_id = prediction.get("target_id")

        x, y = None, None
        resolved_element = None

        if action_type in ("click", "type") and target_id is not None:
            mapping = prediction.get("mapping") or self._last_mapping
            if mapping and str(target_id) in mapping:
                info = mapping[str(target_id)]
                x, y = info["center"]
                resolved_element = {
                    "element_id": info.get("element_id"),
                    "tag": info.get("tag"),
                    "text": info.get("text"),
                    "bbox": info.get("bbox"),
                }
            else:
                logger.error(
                    f"target_id {target_id} not found in mapping (available: "
                    f"{list((prediction.get('mapping') or {}).keys())[:5]}...)"
                )
                return {
                    "action_type": "fail",
                    "x": None, "y": None,
                    "value": prediction.get("value"),
                    "target_id": target_id,
                    "reason": f"Marker ID {target_id} not in current mapping",
                    "confidence": prediction.get("confidence"),
                    "method": self.name,
                    "raw_prediction": prediction,
                }

        return {
            "action_type": action_type,
            "x": x,
            "y": y,
            "value": prediction.get("value"),
            "key": prediction.get("key"),
            "target_id": target_id,
            "question": prediction.get("question"),
            "resolved_element": resolved_element,
            "reason": prediction.get("reason", ""),
            "confidence": prediction.get("confidence"),
            "method": self.name,
            "raw_prediction": prediction,
        }

    @staticmethod
    def _stale_markers(
        mapping: Dict[str, Dict[str, Any]],
        history: Optional[List[Dict[str, Any]]],
        tol: float = 6.0,
    ) -> Dict[str, str]:
        """Markers whose previous click produced no page change.

        Matched by click coordinates (robust to marker renumbering).
        Any later successful click clears ALL dead markers because the
        page state has changed — elements shift positions, so stale
        coordinates from a prior state must not pollute the new overlay.
        """
        if not history or not mapping:
            return {}
        dead: List[tuple] = []
        for h in history:
            res = str(h.get("result", ""))
            if res == "success":
                # A successful action means the page changed: ALL prior
                # dead markers are invalidated (elements may have moved
                # or been replaced), not just the one at these coords.
                dead.clear()
            elif h.get("action") == "click" and h.get("coordinates"):
                if "No visual/URL/DOM change" in res:
                    try:
                        cx, cy = (float(v) for v in h["coordinates"])
                        dead.append((cx, cy))
                    except (TypeError, ValueError):
                        pass
        out: Dict[str, str] = {}
        for mid, info in mapping.items():
            c = info.get("center")
            if not c:
                continue
            for dx, dy in dead:
                if abs(c[0] - dx) <= tol and abs(c[1] - dy) <= tol:
                    out[str(mid)] = "clicked, no effect"
                    break
        return out

    @staticmethod
    def _dropdown_trigger_markers(
        mapping: Dict[str, Dict[str, Any]],
        history: Optional[List[Dict[str, Any]]],
        tol: float = 6.0,
    ) -> Dict[str, str]:
        """Detect if the last action was a click that opened a dropdown/menu.

        Returns {mid: "dropdown trigger"} for markers matching the trigger
        button coordinates so re-clicking is forbidden.
        """
        if not history or not mapping:
            return {}
        last = history[-1]
        if (last.get("action") == "click"
                and last.get("result") == "success"
                and last.get("verification") is not False
                and last.get("coordinates")):
            prev_count = last.get("_dom_count_before")
            cur_count = len(mapping)
            if prev_count is not None and cur_count > prev_count:
                try:
                    lx, ly = (float(v) for v in last["coordinates"])
                    out: Dict[str, str] = {}
                    for mid, info in mapping.items():
                        c = info.get("center")
                        if c and abs(c[0] - lx) <= tol and abs(c[1] - ly) <= tol:
                            out[str(mid)] = "dropdown trigger button (menu already open; do not re-click)"
                    return out
                except (TypeError, ValueError):
                    return {}
        return {}

    def _build_prompt(
        self,
        instruction: str,
        mapping: Dict[str, Dict[str, Any]],
        history: Optional[List[Dict[str, Any]]] = None,
        stale: Optional[Dict[str, str]] = None,
        dropdown_triggers: Optional[Dict[str, str]] = None,
    ) -> str:
        """Construct the SoM prompt, optionally with DOM text hints."""
        stale = stale or {}
        dropdown_triggers = dropdown_triggers or {}
        marker_list = sorted(mapping.keys(), key=int)
        if self.include_dom_in_prompt:
            lines = []
            for mid in marker_list[:50]:
                info = mapping[mid]
                text = (info.get("text") or info.get("aria_label") or "").strip()
                text = text[:40] + ("..." if len(text) > 40 else "")
                tag = info.get("tag", "")
                flag = ""
                if str(mid) in stale:
                    flag = ("  <-- ALREADY CLICKED, NO EFFECT (already active); "
                            "DO NOT CLICK")
                elif str(mid) in dropdown_triggers:
                    flag = ("  <-- DROPDOWN TRIGGER BUTTON (MENU ALREADY OPEN); "
                            "DO NOT CLICK (will close menu)")
                lines.append(f"  [{mid}] <{tag}> {text}{flag}".rstrip())
            elements_text = "\n".join(lines)
        else:
            elements_text = f"Available markers: {', '.join(marker_list)}"
        if stale:
            elements_text += (
                "\n\nALREADY DONE: markers "
                + ", ".join(f"[{m}]" for m in sorted(stale, key=int))
                + " were clicked and the page did not change, meaning that "
                "step is already complete. Move on to the NEXT step of the task "
                "using a different marker.")

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
                    recent_lines.append(f"  Step {step_num}: target={h.get('target_id', '?')} action=type value={h.get('value', '')!r} → {h.get('result', '?')}")
                else:
                    target_id = h.get("target_id")
                    target_str = f" target={target_id}" if target_id is not None else ""
                    elem_desc = ""
                    elem_text = (h.get("element_text") or "").strip()
                    elem_tag = h.get("element_tag") or ""
                    if elem_text:
                        elem_desc = f' ("{elem_text[:35]}")'
                    elif elem_tag:
                        elem_desc = f" <{elem_tag}>"
                    recent_lines.append(f"  Step {step_num}:{target_str}{elem_desc} action={act} → {h.get('result', '?')}")

            if qa_lines:
                history_text += "\n\nInformation provided by user:\n" + "\n".join(qa_lines)
            if recent_lines:
                history_text += "\n\nRecent actions:\n" + "\n".join(recent_lines)

            # Detect dropdown/menu just opened: if the last action was a
            # successful click and the current mapping has MORE elements
            # than before, a menu/dropdown appeared. Tell the VLM.
            if dropdown_triggers:
                trig_str = ", ".join(f"[{m}]" for m in sorted(dropdown_triggers, key=int))
                last = history[-1] if history else {}
                btn_name = last.get("element_text") or f"marker {trig_str}"
                history_text += (
                    f"\n\n⚠️ DROPDOWN/MENU IS NOW OPEN: the last click on \"{btn_name}\" "
                    f"opened a dropdown with new options. Button {trig_str} is the open trigger. "
                    "You MUST click one of the NEW items inside the menu (e.g. 'Download Receipt'). "
                    f"Do NOT click {trig_str} again (that would CLOSE the menu).")

        return f"""You are a highly capable web automation agent. Your objective is to achieve the user's goal by navigating and interacting with web pages. You will be given the current URL, the user's goal, and a visual representation of the web page with numbered bounding boxes (Set-of-Mark) over interactive elements.

Task: {instruction}

Interactive elements (by marker ID):
{elements_text}{history_text}

        AVAILABLE ACTIONS:
- `{{"action": "click", "target_id": <int>}}`: Click an element by its ID.
- `{{"action": "type", "target_id": <int>, "value": "<string>"}}`: Type text into an element.
- `{{"action": "press", "key": "Enter"}}`: Press a key on the focused element (key is Enter, Tab, or Escape; default Enter). Use this to submit a search/form AFTER typing. No `target_id` needed.
- `{{"action": "scroll"}}`: Scroll the page.
- `{{"action": "done"}}`: The user's overall goal is successfully completed.
- `{{"action": "ask_user", "question": "<string>"}}`: Stop and ask the human for missing information (no `target_id` needed).

SUBMIT RULE: after typing a query into a search box, your next action MUST be `press` (Enter) or a click on a suggestion/search button. NEVER type the same text twice: if the text is already in the box, submit it instead of re-typing.
NAVIGATION RULE: if you click a navigation link, menu item, or tab and the page does not change, it is already active. NEVER click the same menu item or tab again; proceed directly to the next step or content area of your task.
DROPDOWN RULE: if you click a button and a dropdown menu appears with new options, your NEXT action must click one of the NEW items INSIDE the dropdown menu (not the trigger button again). Re-clicking the trigger will CLOSE the menu.

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

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse VLM response, coercing int-like target_ids ('10', [10])."""
        from .grounding_utils import parse_vlm_json

        return parse_vlm_json(response)

    def _validate_prediction(
        self, prediction: Dict[str, Any], mapping: Dict[str, Dict[str, Any]],
    ) -> bool:
        """Validate structure and that the target_id exists in current mapping."""
        action = prediction.get("action")
        if action not in self.SUPPORTED_ACTIONS:
            return False
        if action == "ask_user":
            q = prediction.get("question")
            return isinstance(q, str) and bool(q.strip())
        if action in ("click", "type"):
            target_id = prediction.get("target_id")
            if not isinstance(target_id, int):
                return False
            if str(target_id) not in mapping:
                return False
        if action == "type" and "value" not in prediction:
            return False
        return True
