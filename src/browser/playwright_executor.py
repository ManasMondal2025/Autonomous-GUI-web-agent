"""Low-level Playwright primitives (browser layer).

The high-level ActionExecutor (src/actions/) owns policy (validation, timing);
this module owns raw page primitives so benchmark adapters can reuse them
without importing agent policy.
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class PlaywrightExecutor:
    """Thin wrapper around a Playwright Page."""

    def __init__(self, browser):
        self.browser = browser

    @property
    def page(self):
        return self.browser.get_page()

    def click(self, x: float, y: float) -> None:
        self.page.mouse.click(float(x), float(y))

    def type_text(self, x: float, y: float, text: str, delay_ms: int = 20) -> None:
        self.page.mouse.click(float(x), float(y))
        self.page.keyboard.press("ControlOrMeta+a")
        self.page.keyboard.type(text, delay=delay_ms)

    def scroll(self, dx: float = 0.0, dy: float = 500.0,
               x: Optional[float] = None, y: Optional[float] = None) -> None:
        vp = self.browser.get_viewport_size()
        self.page.mouse.move(x or vp["width"] / 2, y or vp["height"] / 2)
        self.page.mouse.wheel(float(dx), float(dy))

    def wait(self, ms: int) -> None:
        self.page.wait_for_timeout(int(ms))

    def screenshot(self) -> bytes:
        return self.page.screenshot()

    def fill_by_selector(self, selector: str, text: str) -> None:
        self.page.fill(selector, text)

    def click_selector(self, selector: str) -> None:
        self.page.click(selector)
