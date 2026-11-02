"""State verifier: visual diff + URL change + DOM change (+ optional VLM check)."""

import logging
from typing import Dict, Any, Optional, List
from PIL import Image
import io

logger = logging.getLogger(__name__)


class StateVerifier:
    def __init__(
        self,
        visual_threshold: float = 0.995,
        vlm: Optional[Any] = None,
        use_vlm_verification: bool = False,
    ):
        self.visual_threshold = visual_threshold
        self.vlm = vlm
        self.use_vlm_verification = use_vlm_verification

    def verify(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any],
        action: Dict[str, Any],
        task: str = "",
    ) -> Dict[str, Any]:
        """Compare before/after observations. Returns verification dict."""
        from utils.image_utils import compute_image_similarity

        verdict: Dict[str, Any] = {
            "success": True,
            "checks": {},
            "reason": "",
        }

        # 1. URL change (strong signal for navigation actions)
        before_url = before.get("url", "")
        after_url = after.get("url", "")
        url_changed = before_url != after_url
        verdict["checks"]["url_changed"] = url_changed

        # 2. Visual change
        try:
            img_b = Image.open(io.BytesIO(before["screenshot_bytes"]))
            img_a = Image.open(io.BytesIO(after["screenshot_bytes"]))
            sim = compute_image_similarity(img_b, img_a)
        except Exception as e:
            logger.warning(f"Visual compare failed: {e}")
            sim = 1.0
        verdict["checks"]["visual_similarity"] = round(float(sim), 4)
        visual_changed = sim < self.visual_threshold
        verdict["checks"]["visual_changed"] = visual_changed

        # 3. DOM change (element count / text set)
        dom_b = before.get("dom_elements", [])
        dom_a = after.get("dom_elements", [])
        verdict["checks"]["dom_count_before"] = len(dom_b)
        verdict["checks"]["dom_count_after"] = len(dom_a)
        dom_changed = self._dom_changed(dom_b, dom_a)
        verdict["checks"]["dom_changed"] = dom_changed

        # 4. Execution-level failure always fails verification
        exec_ok = after.get("exec_success", True)
        if not exec_ok:
            verdict["success"] = False
            verdict["reason"] = f"Action execution failed: {after.get('exec_error', '?')}"
            return verdict

        action_type = action.get("action_type", action.get("action", ""))
        if action_type in ("done", "fail"):
            verdict["success"] = True
            verdict["reason"] = f"Terminal action '{action_type}'"
            return verdict
        if action_type == "ask_user":
            # Defense-in-depth: controllers should bypass verify() entirely
            # for ask_user (no re-observe). If reached here anyway, pass
            # without diffing: no DOM/URL change is expected.
            verdict["success"] = True
            verdict["reason"] = ("No-state-change action 'ask_user' "
                                 "(human prompt, browser untouched)")
            verdict["checks"]["no_state_change_expected"] = True
            return verdict

        # Heuristic: click/type/press should change something;
        # wait/scroll may not
        if action_type in ("click", "type", "press"):
            if url_changed or visual_changed or dom_changed:
                verdict["success"] = True
                verdict["reason"] = "State changed after action"
            else:
                verdict["success"] = False
                verdict["reason"] = (
                    "No visual/URL/DOM change after "
                    f"{action_type} (element may already be active/selected; do not retry this element, proceed to next step)")
        else:
            verdict["success"] = True
            verdict["reason"] = f"Action '{action_type}' executed"

        # 5. Optional VLM second opinion (only if heuristic says success,
        # to catch cases where something changed but wrong thing happened)
        if self.use_vlm_verification and self.vlm is not None and verdict["success"]:
            try:
                vlm_ok = self._vlm_check(before, after, action, task)
                verdict["checks"]["vlm_verification"] = vlm_ok
                if vlm_ok is False:
                    verdict["success"] = False
                    verdict["reason"] = "VLM judged the action did not help"
            except Exception as e:
                logger.warning(f"VLM verification failed: {e}")

        return verdict

    def _dom_changed(self, before: List[Dict], after: List[Dict]) -> bool:
        if len(before) != len(after):
            return True
        sig_b = {(e.get("tag"), (e.get("text") or "")[:50], (e.get("value") or "")[:50]) for e in before}
        sig_a = {(e.get("tag"), (e.get("text") or "")[:50], (e.get("value") or "")[:50]) for e in after}
        return sig_b != sig_a

    def _vlm_check(self, before, after, action, task) -> Optional[bool]:
        import json as _json
        prompt = (
            f"Task: {task}\nAction taken: {action}\n"
            "Did this action move the interface toward the goal? "
            'Reply ONLY {"success": true/false, "reason": "..."}'
        )
        img_a = Image.open(io.BytesIO(after["screenshot_bytes"]))
        resp = self.vlm.generate(img_a, prompt)
        try:
            import re
            m = re.search(r"\{.*\}", resp, re.DOTALL)
            data = _json.loads(m.group(0) if m else resp)
            return bool(data.get("success"))
        except Exception:
            return None
