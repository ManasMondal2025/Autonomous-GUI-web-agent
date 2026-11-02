#!/usr/bin/env python3
"""Action-sequence run with per-step CAPTCHA detection (observe-only on CAPTCHAs).

Performs a normal navigation task (open URL, search, open first result) while
running the standalone CaptchaDetector after EVERY action. Screenshots and a
complete action log (actions.json) are saved to the run folder.

Safety: if a BLOCKING captcha is detected, the run STOPS before touching it.
The script never clicks/selects/solves any CAPTCHA widget.

Example:
  python scripts/captcha_search_run.py --url https://www.google.com/ \\
      --query headphones --outdir data/captcha_ss/run_headphones
"""

import argparse
import json
import logging  # noqa: F401  (stdlib logging BEFORE src hits sys.path)
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser.browser_manager import BrowserManager
from captcha.detector import CaptchaDetector

logging.basicConfig(level=logging.WARNING, format="%(message)s")


def _step(page, detector, outdir, name, action_desc, extra=None):
    """Screenshot + detect + log one step. Returns (record, blocked)."""
    shot = outdir / f"{name}.png"
    page.screenshot(path=str(shot))
    result = detector.detect(page)
    d = result.to_dict()
    rec = {
        "step": name,
        "action": action_desc,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "url": page.url,
        "title": page.title(),
        "screenshot": str(shot),
        "detection": d,
    }
    if extra:
        rec.update(extra)
    print(f"[{name}] {action_desc}")
    print(f"  url={page.url}")
    print(f"  detected={d['detected']} blocking={d['blocking']} "
          f"conf={d['confidence']:.2f} type={d['captcha_type']} "
          f"provider={d['provider']} signals={d['signals'] or 'none'}")
    print(f"  shot={shot}")
    return rec, bool(d["blocking"])


def main() -> int:
    ap = argparse.ArgumentParser(description="Search run with per-step CAPTCHA checks")
    ap.add_argument("--url", default="https://www.google.com/")
    ap.add_argument("--query", default="headphones")
    ap.add_argument("--outdir", default="data/captcha_ss/run_headphones")
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--no-headless", dest="headless", action="store_false")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log = {"task": f"open {args.url}, search '{args.query}', open first link",
           "started_utc": datetime.now(timezone.utc).isoformat(),
           "steps": []}
    detector = CaptchaDetector()
    browser = BrowserManager(headless=args.headless)
    rc = 0
    try:
        browser.launch()
        page = browser.new_page()

        page.goto(args.url, wait_until="load")
        try:
            page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass
        rec, blocked = _step(page, detector, outdir, "ss_01_open_homepage",
                              f"goto {args.url}")
        log["steps"].append(rec)
        if blocked:
            log["stopped"] = "blocking captcha on landing page; no interaction performed"
            rc = 2
            return rc

        box = page.locator('textarea[name="q"], input[name="q"]').first
        box.wait_for(timeout=15000)
        box.fill(args.query)
        page.wait_for_timeout(800)
        rec, blocked = _step(page, detector, outdir, "ss_02_query_typed",
                              f"type query '{args.query}' into search box")
        log["steps"].append(rec)
        if blocked:
            log["stopped"] = "blocking captcha after typing; aborting"
            rc = 2
            return rc

        box.press("Enter")
        page.wait_for_load_state("load", timeout=20000)
        try:
            page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass
        rec, blocked = _step(page, detector, outdir, "ss_03_search_results",
                              "submit search (Enter)")
        log["steps"].append(rec)
        if blocked:
            log["stopped"] = "blocking captcha on results page; not opening any link"
            rc = 2
            return rc

        first = page.locator('div#search a:has(h3)').first
        first.wait_for(timeout=15000)
        href = first.get_attribute("href")
        text = (first.inner_text() or "").strip()[:120]
        first.click()
        page.wait_for_load_state("load", timeout=20000)
        try:
            page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass
        rec, blocked = _step(
            page, detector, outdir, "ss_04_first_link_opened",
            "click first result link",
            extra={"clicked_href": href, "clicked_text": text})
        log["steps"].append(rec)
        if blocked:
            log["note"] = ("blocking captcha on destination page; "
                           "task navigation done, captcha left untouched")
            rc = 2
            return rc

        log["status"] = "completed: no blocking captcha encountered"
        return rc
    except Exception as e:
        log["error"] = f"{type(e).__name__}: {e}"
        print(f"[error] {type(e).__name__}: {e}")
        return 1
    finally:
        log["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (outdir / "actions.json").write_text(json.dumps(log, indent=1))
        print(f"[log] actions.json written to {outdir / 'actions.json'}")
        browser.close()


if __name__ == "__main__":
    raise SystemExit(main())
