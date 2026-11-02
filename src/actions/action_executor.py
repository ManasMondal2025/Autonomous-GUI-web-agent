"""Action executor: translates validated Actions into Playwright calls.

Only supports the whitelisted action types (click/type/press/scroll/wait
plus terminal ask/done/fail routing). No eval/exec of model output.
Each execute() returns a result dict and captures a post-action screenshot.
"""

import logging
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, Any, Optional
from PIL import Image
import io

from .action_schema import Action, ActionType

logger = logging.getLogger(__name__)


def _show_x_cursor(x, y, pause=1.0):
    """Move the real X11 cursor so VNC viewers see the click position.

    Playwright's mouse.click is synthetic (CDP) and never moves the X
    pointer, so x11vnc shows page changes but no cursor glide. Best effort:
    no DISPLAY or no xdotool -> silently skip (headless safe).
    """
    if not os.environ.get("DISPLAY"):
        return
    if shutil.which("xdotool") is None:
        return
    try:
        subprocess.run(
            ["xdotool", "mousemove", "--sync", str(int(x)), str(int(y))],
            timeout=2, check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(pause)
    except Exception:
        pass


class ActionExecutor:
    def __init__(
        self,
        browser,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        default_wait_ms: int = 1000,
        type_delay_ms: int = 100,
        move_steps: int = 50,
        cursor_pause: float = 1.0,
    ):
        self.browser = browser
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.default_wait_ms = default_wait_ms
        self.type_delay_ms = type_delay_ms
        self.move_steps = move_steps
        self.cursor_pause = cursor_pause

    def execute(self, action: Action) -> Dict[str, Any]:
        """Execute a validated action. Returns result dict."""
        at = getattr(action, "action_type", "")
        if at == "ask_user" or str(at) == "ask_user":
            raise ValueError(
                "ask_user must be handled by the controller (HITL prompt), "
                "never sent to the browser executor")
        page = self.browser.get_page()
        before_url = self.browser.get_url()
        result: Dict[str, Any] = {
            "action_type": action.action_type,
            "success": False,
            "before_url": before_url,
            "after_url": before_url,
            "error": None,
        }

        try:
            if action.action_type == "click":
                self._click(page, action)
            elif action.action_type == "type":
                self._type(page, action)
            elif action.action_type == "press":
                self._press(page, action)
            elif action.action_type == "scroll":
                self._scroll(page, action)
            elif action.action_type == "wait":
                self._wait(page, action)
            elif action.action_type in ("done", "fail"):
                result["success"] = True
                result["terminal"] = True
                result["after_url"] = self.browser.get_url()
                return result
            else:
                raise ValueError(f"Unsupported action: {action.action_type}")

            # settle + post screenshot: prefer domcontentloaded and short networkidle
            # so persistent background requests do not freeze execution for 10s
            try:
                page.wait_for_load_state("domcontentloaded", timeout=2000)
            except Exception:
                pass
            try:
                page.wait_for_load_state("networkidle", timeout=1000)
            except Exception:
                pass
            result["success"] = True
            result["after_url"] = self.browser.get_url()
            result["url_changed"] = result["after_url"] != before_url
            try:
                result["screenshot_bytes"] = page.screenshot()
            except Exception as e:
                logger.warning(f"Post-action screenshot failed: {e}")
        except Exception as e:
            logger.error(f"Action {action.action_type} failed: {e}")
            result["success"] = False
            result["error"] = str(e)

        return result

    def _validate_point(self, x: float, y: float) -> None:
        if x is None or y is None:
            raise ValueError("click/type requires x and y coordinates")
        if not (0 <= x < self.viewport_width and 0 <= y < self.viewport_height):
            raise ValueError(
                f"Coordinates ({x},{y}) outside viewport "
                f"{self.viewport_width}x{self.viewport_height}"
            )

    def _click(self, page, action: Action) -> None:
        self._validate_point(action.x, action.y)
        _show_x_cursor(action.x, action.y, pause=self.cursor_pause)
        logger.info(f"Click at ({action.x:.0f}, {action.y:.0f})")
        try:
            page.mouse.move(float(action.x), float(action.y), steps=self.move_steps)
        except TypeError:
            page.mouse.move(float(action.x), float(action.y))
        page.wait_for_timeout(600)
        page.mouse.click(float(action.x), float(action.y))
        page.wait_for_timeout(500)

    def _type(self, page, action: Action) -> None:
        if not action.value:
            raise ValueError("type action requires 'value' text")
        self._validate_point(action.x, action.y)
        _show_x_cursor(action.x, action.y, pause=self.cursor_pause)
        logger.info(f"Type '{action.value[:30]}' at ({action.x:.0f}, {action.y:.0f})")
        page.mouse.click(float(action.x), float(action.y))
        page.wait_for_timeout(300)
        # select existing text then type ("ControlOrMeta" only exists in
        # Playwright>=1.42, so pick the modifier per platform for compat)
        select_all = "Meta+a" if sys.platform == "darwin" else "Control+a"
        page.keyboard.press(select_all)
        page.keyboard.type(action.value, delay=self.type_delay_ms)
        page.wait_for_timeout(300)

    def _press(self, page, action: Action) -> None:
        from .action_schema import normalize_press_key
        key = normalize_press_key(getattr(action, "key", None))
        # Acts on the focused element (e.g. the just-typed search box);
        # any x/y on the action are intentionally ignored and the mouse
        # is never moved, so noVNC viewers see no spurious cursor jumps.
        logger.info(f"Press '{key}' on focused element")
        page.keyboard.press(key)
        page.wait_for_timeout(500)

    def _scroll(self, page, action: Action) -> None:
        dx = float(action.dx or 0.0)
        dy = float(action.dy or 500.0)
        # scroll at viewport center
        cx, cy = self.viewport_width / 2, self.viewport_height / 2
        logger.info(f"Scroll dx={dx} dy={dy} at ({cx:.0f},{cy:.0f})")
        _show_x_cursor(cx, cy, pause=self.cursor_pause)
        page.mouse.move(cx, cy)
        page.mouse.wheel(dx, dy)
        page.wait_for_timeout(500)

    def _wait(self, page, action: Action) -> None:
        ms = self.default_wait_ms
        if action.value:
            try:
                ms = int(float(action.value) * 1000) if float(action.value) < 100 else int(float(action.value))
            except ValueError:
                pass
        logger.info(f"Wait {ms}ms")
        page.wait_for_timeout(ms)
