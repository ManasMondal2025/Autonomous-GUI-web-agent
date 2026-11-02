"""Browser manager for Playwright-based browser automation."""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page

logger = logging.getLogger(__name__)


class BrowserManager:
    """Manages Playwright browser lifecycle and page interactions."""

    def __init__(
        self,
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        timeout_ms: int = 30000,
        user_agent: Optional[str] = None,
        slow_mo_ms: int = 0,
        disable_images: bool = False,
        disable_javascript: bool = False,
    ):
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.timeout_ms = timeout_ms
        self.user_agent = user_agent
        self.slow_mo_ms = slow_mo_ms
        self.disable_images = disable_images
        self.disable_javascript = disable_javascript

        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        # File downloads observed in this session (Playwright `download`
        # events). Each entry: {filename, url, path, _obj}.
        self.downloads_dir = Path(os.getenv("AGENT_DOWNLOADS_DIR", "downloads"))
        self.downloads: List[Dict[str, Any]] = []

        logger.info(
            f"BrowserManager initialized: viewport={viewport_width}x{viewport_height}, "
            f"headless={headless}"
        )

    def launch(self) -> None:
        """Launch the browser."""
        if self._browser is not None:
            logger.warning("Browser already launched")
            return

        self._playwright = sync_playwright().start()
        launch_args = ["--no-sandbox", "--disable-dev-shm-usage",
                       "--disable-gpu", "--disable-software-rasterizer"]
        if not self.headless:
            # Headed (noVNC/Xvfb, no window manager): pin the OS window to
            # the top-left and size it to the VNC desktop. Without this
            # Chromium opens small/offset and the noVNC pane shows a tiny
            # browser floating in black. device-scale is forced to 1 so X
            # DPI quirks can't shrink the rendering. Headless runs are
            # unaffected.
            launch_args.append("--window-position=0,0")
            launch_args.append(
                f"--window-size={self.viewport_width},"
                f"{self.viewport_height + 80}")
            launch_args.append("--force-device-scale-factor=1")
        self._browser = self._playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo_ms,
            args=launch_args,
        )

        context_options = {
            "viewport": {
                "width": self.viewport_width,
                "height": self.viewport_height,
            },
            "accept_downloads": True,
        }
        if self.user_agent:
            context_options["user_agent"] = self.user_agent

        self._context = self._browser.new_context(**context_options)
        self._context.set_default_timeout(self.timeout_ms)
        self._context.set_default_navigation_timeout(self.timeout_ms)
        self._context.on("page", lambda p: p.on("download", self._on_download))

        if self.disable_images:
            self._context.route(
                "**/*.{png,jpg,jpeg,gif,svg,webp}",
                lambda route: route.abort(),
            )

        if self.disable_javascript:
            self._context.add_init_script("javascript:void(0)")

        logger.info("Browser launched successfully")

    DOM_CURSOR_JS = """(() => {
      const id = 'playwright-agent-cursor';
      const install = () => {
        if (document.getElementById(id)) return;
        const cursor = document.createElement('div');
        cursor.id = id;
        Object.assign(cursor.style, {
          position: 'fixed', top: '0', left: '0',
          width: '18px', height: '18px',
          backgroundColor: 'rgba(255,0,0,0.7)',
          borderRadius: '50%', pointerEvents: 'none',
          zIndex: '999999',
          transition: 'left 0.05s linear, top 0.05s linear',
          transform: 'translate(-50%,-50%)',
        });
        (document.body || document.documentElement).appendChild(cursor);
        document.addEventListener('mousemove', (e) => {
          cursor.style.left = e.clientX + 'px';
          cursor.style.top = e.clientY + 'px';
        });
        document.addEventListener('mousedown', () => {
          cursor.style.transform = 'translate(-50%,-50%) scale(0.7)';
          cursor.style.backgroundColor = 'rgba(0,255,0,0.9)';
        });
        document.addEventListener('mouseup', () => {
          cursor.style.transform = 'translate(-50%,-50%) scale(1)';
          cursor.style.backgroundColor = 'rgba(255,0,0,0.7)';
        });
      };
      if (document.readyState === 'loading') {
        window.addEventListener('DOMContentLoaded', install, { once: true });
      } else {
        install();
      }
    })();"""

    def new_page(self) -> Page:
        """Create a new page."""
        if self._browser is None:
            self.launch()

        if self._page is not None:
            self._page.close()

        self._page = self._context.new_page()
        try:
            self._page.add_init_script(self.DOM_CURSOR_JS)
        except Exception as e:
            logger.debug(f"DOM cursor inject failed: {e}")
        logger.info("New page created")
        return self._page

    def navigate(self, url: str, wait_until: str = "load") -> None:
        """Navigate to a URL."""
        if self._page is None:
            raise RuntimeError("No active page. Call new_page() first.")

        logger.info(f"Navigating to: {url}")
        self._page.goto(url, wait_until=wait_until, timeout=self.timeout_ms)
        try:
            self._page.wait_for_load_state("networkidle", timeout=min(self.timeout_ms, 3000))
        except Exception:
            pass

    def screenshot(self, path: Optional[Path] = None) -> bytes:
        """Take a screenshot of the current page."""
        if self._page is None:
            raise RuntimeError("No active page.")

        screenshot_bytes = self._page.screenshot(path=str(path) if path else None)
        if path:
            logger.debug(f"Screenshot saved to: {path}")
        return screenshot_bytes

    def get_page(self) -> Page:
        """Get the current page object."""
        if self._page is None:
            raise RuntimeError("No active page.")
        return self._page

    def get_url(self) -> str:
        """Get the current page URL."""
        if self._page is None:
            return ""
        return self._page.url

    def get_viewport_size(self) -> Dict[str, int]:
        """Get the viewport size."""
        return {
            "width": self.viewport_width,
            "height": self.viewport_height,
        }

    def get_scroll_position(self) -> Dict[str, int]:
        """Get the current scroll position."""
        if self._page is None:
            return {"x": 0, "y": 0}
        return {
            "x": self._page.evaluate("window.scrollX || window.pageXOffset || 0"),
            "y": self._page.evaluate("window.scrollY || window.pageYOffset || 0"),
        }

    # -- downloads ----------------------------------------------------
    def _on_download(self, download) -> None:
        """Record a download event (cheap; saving happens later)."""
        try:
            name = download.suggested_filename
            url = download.url
        except Exception:
            name, url = "download", ""
        self.downloads.append({"filename": name, "url": url,
                               "path": None, "_obj": download})
        logger.info(f"[DOWNLOAD] started: {name}")

    def download_count(self) -> int:
        return len(self.downloads)

    def save_downloads(self) -> List[str]:
        """Persist pending downloads into downloads_dir; return saved paths."""
        saved = []
        for d in self.downloads:
            if d.get("path") or d.get("_obj") is None:
                if d.get("path"):
                    saved.append(d["path"])
                continue
            try:
                self.downloads_dir.mkdir(parents=True, exist_ok=True)
                target = self.downloads_dir / d["filename"]
                d["_obj"].save_as(str(target))
                d["path"] = str(target.resolve())
                saved.append(d["path"])
                logger.info(f"[DOWNLOAD] saved: {d['path']}")
            except Exception as e:
                logger.warning(f"[DOWNLOAD] save failed for {d['filename']}: {e}")
        return saved

    def close(self) -> None:
        """Close the browser and cleanup."""
        try:
            self.save_downloads()
        except Exception:
            pass
        try:
            if self._page is not None:
                self._page.close()
                self._page = None
            if self._context is not None:
                self._context.close()
                self._context = None
            if self._browser is not None:
                self._browser.close()
                self._browser = None
            if self._playwright is not None:
                self._playwright.stop()
                self._playwright = None
            logger.info("Browser closed")
        except Exception as e:
            logger.error(f"Error during browser close: {e}")

    def __enter__(self):
        self.launch()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
