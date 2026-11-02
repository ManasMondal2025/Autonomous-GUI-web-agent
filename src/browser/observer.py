"""Observer module for capturing browser state and screenshots."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import io
import base64
import json
from datetime import datetime

from .browser_manager import BrowserManager
from .dom_extractor import DOMExtractor

logger = logging.getLogger(__name__)


class Observer:
    """Observes browser state including screenshots and DOM."""

    def __init__(
        self,
        browser: BrowserManager,
        dom_extractor: Optional[DOMExtractor] = None,
        save_dir: Optional[Path] = None,
        proposer=None,
    ):
        self.browser = browser
        self.dom_extractor = dom_extractor or DOMExtractor()
        if proposer is None:
            from .proposers import DOMProposer
            proposer = DOMProposer(self.dom_extractor)
        self.proposer = proposer
        self.save_dir = save_dir
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
        self._step_counter = 0

    def observe(self, save_screenshot: bool = True) -> Dict[str, Any]:
        """Capture the current browser state."""
        page = self.browser.get_page()
        self._step_counter += 1

        try:
            screenshot_bytes = self.browser.screenshot()
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
            raise

        image = Image.open(io.BytesIO(screenshot_bytes))
        actual_width, actual_height = image.size

        viewport = self.browser.get_viewport_size()
        if actual_width != viewport["width"] or actual_height != viewport["height"]:
            logger.warning(
                f"Screenshot size ({actual_width}x{actual_height}) differs from "
                f"viewport ({viewport['width']}x{viewport['height']})"
            )

        url = self.browser.get_url()
        scroll_position = self.browser.get_scroll_position()

        try:
            dom_elements = self.proposer.propose(
                page, image, url, scroll_position)
        except Exception as e:
            logger.error(f"Failed to extract proposals: {e}")
            dom_elements = []

        try:
            visible_text = page.evaluate(
                "(document.body ? document.body.innerText : '').substring(0, 4000)"
            ) or ""
        except Exception as e:
            logger.warning(f"Failed to capture visible text: {e}")
            visible_text = ""

        observation = {
            "step": self._step_counter,
            "screenshot_bytes": screenshot_bytes,
            "screenshot_size": {
                "width": actual_width,
                "height": actual_height,
            },
            "viewport": viewport,
            "url": url,
            "visible_text": visible_text,
            "scroll_position": scroll_position,
            "dom_elements": dom_elements,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if save_screenshot and self.save_dir:
            screenshot_path = self.save_dir / f"step_{self._step_counter:03d}.png"
            with open(screenshot_path, "wb") as f:
                f.write(screenshot_bytes)

            json_path = self.save_dir / f"step_{self._step_counter:03d}.json"
            json_data = {
                k: v for k, v in observation.items() if k != "screenshot_bytes"
            }
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2)

        logger.debug(
            f"Observation: step={self._step_counter}, url={observation['url']}, "
            f"elements={len(dom_elements)}"
        )

        return observation

    def reset_step_counter(self) -> None:
        """Reset the step counter."""
        self._step_counter = 0

    def get_screenshot_base64(self, observation: Dict[str, Any]) -> str:
        """Get base64-encoded screenshot from observation."""
        if "screenshot_bytes" in observation:
            return base64.b64encode(observation["screenshot_bytes"]).decode("utf-8")
        return ""

    def save_screenshot_from_observation(
        self, observation: Dict[str, Any], path: Path
    ) -> None:
        """Save screenshot from observation to a file."""
        if "screenshot_bytes" in observation:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                f.write(observation["screenshot_bytes"])
