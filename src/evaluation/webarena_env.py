"""WebArena environment adapter (decoupled from the core agent).

The core agent never imports this module — scripts wire it up instead.
Task definition format (JSON list):
  {"task_id": "t1", "start_url": "http://.../...", "goal": "do X",
   "check": {"type": "url_contains", "value": "checkout"},
   # or {"type": "element_exists", "selector": "#cart"},
   # or {"type": "input_value", "selector": "#search", "value": "laptop"}}
   # or {"type": "element_text_contains", "selector": "#total", "value": "$9"}
   # or {"type": "all_of", "checks": [{...}, {...}]}  (every sub-check must pass)}
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

logger = logging.getLogger(__name__)


class WebArenaEnvironment:
    def __init__(self, browser, base_url: str = "", timeout_ms: int = 30000):
        self.browser = browser
        self.base_url = base_url.rstrip("/")
        self.timeout_ms = timeout_ms
        self.current_task: Optional[Dict[str, Any]] = None

    # -- task management -------------------------------------------------
    def reset(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Navigate to the task start URL. Returns initial info dict."""
        self.current_task = task
        url = task["start_url"]
        if url.startswith("/") and self.base_url:
            url = self.base_url + url
        if url.startswith("file://") and not url.startswith("file:///"):
            rel = url[len("file://"):]
            url = f"file://{(Path.cwd() / rel).resolve()}"
        self.browser.new_page()
        self.browser.navigate(url)
        logger.info(f"WebArena reset: {task.get('task_id')} -> {url}")
        return {"task_id": task.get("task_id"), "url": self.browser.get_url()}

    def check_success(self, task: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Programmatic success check (no VLM involved)."""
        task = task or self.current_task
        if not task:
            return {"success": False, "reason": "no active task"}
        check = task.get("check", {})
        ctype = check.get("type", "")
        page = self.browser.get_page()
        try:
            if ctype == "url_contains":
                ok = check["value"] in self.browser.get_url()
                return {"success": ok, "reason": f"url contains '{check['value']}' = {ok}"}
            if ctype == "element_exists":
                ok = page.locator(check["selector"]).count() > 0
                return {"success": ok, "reason": f"selector '{check['selector']}' exists = {ok}"}
            if ctype == "input_value":
                ok = page.input_value(check["selector"]) == check["value"]
                return {"success": ok, "reason": f"input value check = {ok}"}
            if ctype == "element_text_contains":
                el = page.locator(check["selector"]).first
                txt = el.inner_text() if el.count() > 0 else ""
                ok = check["value"] in txt
                return {"success": ok,
                        "reason": f"text of '{check['selector']}' contains "
                                  f"'{check['value']}' = {ok}"}
            if ctype == "all_of":
                subs = check.get("checks", [])
                if not subs:
                    return {"success": False, "reason": "all_of with no sub-checks"}
                results = [self.check_success(
                    {"check": s, "task_id": (task or {}).get("task_id")})
                    for s in subs]
                ok = all(r.get("success") for r in results)
                return {"success": ok,
                        "reason": "all_of [" + "; ".join(
                            r.get("reason", "?") for r in results) + "]"}
            if ctype == "always_true":  # smoke tests only
                return {"success": True, "reason": "smoke check"}
            return {"success": False, "reason": f"unknown check type '{ctype}'"}
        except Exception as e:
            return {"success": False, "reason": f"check error: {e}"}

    # -- thin observe/execute passthroughs (agent owns the real logic) ---
    def observe(self, observer) -> Dict[str, Any]:
        return observer.observe(save_screenshot=False)

    def execute(self, executor, action) -> Dict[str, Any]:
        return executor.execute(action)


def load_tasks(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("tasks", [])
