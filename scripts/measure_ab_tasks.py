#!/usr/bin/env python3
"""Measure GT boxes + behaviors for the 12-task A/B experiment.

All measurements at viewport 1280x720 (the experiment viewport). DOM rects
via getBoundingClientRect; shadow DOM pierced via JS; iframe via frame
locator; canvas buttons computed from the canvas rect + drawn coordinates
(read from test_1.html's own draw script). No models, no OCR -- rerun on
the server before the campaign (fonts shift pixels cross-platform).

Writes data/ab_tasks.json: [{task_id, start_url (portable file:// rel),
goal, expected_action, expected_target_keywords, gt (xyxy), check}].
Checkers reuse WebArenaEnvironment.check_success types.
"""

import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

VW, VH = 1280, 720
log = logging.getLogger("measure_ab")


def rect(page, selector):
    box = page.locator(selector).first.bounding_box()
    if not box:
        raise RuntimeError(f"no box for {selector}")
    return [box["x"], box["y"], box["x"] + box["width"],
            box["y"] + box["height"]]


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    from playwright.sync_api import sync_playwright
    import os

    tasks = []
    with sync_playwright() as p:
        # System Chrome where present (local Mac), else bundled Playwright
        # chromium (server). Same page, same viewport either way.
        try:
            browser = p.chromium.launch(
                channel="chrome", headless=True,
                args=["--no-sandbox"] if os.geteuid() == 0 else [])
        except Exception:
            log.info("System Chrome unavailable; using bundled chromium.")
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox"] if os.geteuid() == 0 else [])
        pg = browser.new_page(viewport={"width": VW, "height": VH})
        root = REPO.resolve().as_uri()

        def goto(rel):
            pg.goto(f"{root}/{rel}")
            pg.wait_for_timeout(400)

        # ---- minishop (5): behaviors verified in page JS ----
        goto("test_pages/minishop/index.html")
        tasks.append(("t01_shop_search_type", "test_pages/minishop/index.html",
                      "Type 'laptop' into the search box", "type", ["laptop"],
                      rect(pg, "#search-input"),
                      {"type": "input_value", "selector": "#search-input",
                       "value": "laptop"}, "laptop"))
        tasks.append(("t02_shop_open_laptop", "test_pages/minishop/index.html",
                      "Open the Laptop Pro product page", "click",
                      ["laptop"], rect(pg, "#link-laptop"),
                      {"type": "url_contains", "value": "product"}, None))
        goto("test_pages/minishop/product.html")
        tasks.append(("t03_shop_add_cart", "test_pages/minishop/product.html",
                      "Add the laptop to the cart", "click", ["cart"],
                      rect(pg, "#add-to-cart"),
                      {"type": "element_text_contains",
                       "selector": "#cart-status",
                       "value": "Added to cart"}, None))
        tasks.append(("t04_shop_view_cart", "test_pages/minishop/product.html",
                      "Open the shopping cart", "click", ["cart"],
                      rect(pg, "#view-cart"),
                      {"type": "url_contains", "value": "cart"}, None))
        goto("test_pages/minishop/index.html")
        tasks.append(("t05_shop_open_phone", "test_pages/minishop/index.html",
                      "Open the Phone Max product page", "click", ["phone"],
                      rect(pg, "#link-phone"),
                      {"type": "url_contains", "value": "product"}, None))

        # ---- shadow GT via a TEMP open-mode copy (closed roots are
        # intentionally unpierceable; shadow mode does not affect layout,
        # so coordinates transfer pixel-identically; copy deleted after).
        tmpdir = Path(tempfile.mkdtemp(prefix="ab_shadow_"))
        try:
            tmp = tmpdir / "test_1_open.html"
            tmp.write_text((REPO / "test_1.html").read_text().replace(
                'attachShadow({mode:"closed"})',
                'attachShadow({mode:"open"})'))
            pg.goto(tmp.resolve().as_uri())
            pg.wait_for_timeout(400)
            sh = pg.evaluate(
                "() => { const r = document.querySelector("
                "'#closed-shadow-host').shadowRoot.querySelector("
                "'button').getBoundingClientRect(); return [r.x, r.y, "
                "r.width, r.height]; }")
            shadow_gt = [sh[0], sh[1], sh[0] + sh[2], sh[1] + sh[3]]
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        # ---- test_1.html (6) on the REAL page ----
        goto("test_1.html")
        pg.wait_for_timeout(1500)  # dynamic Continue appears after ~1s
        tasks.append(("t06_test1_login", "test_1.html",
                      "Click the Login button", "click", ["login"],
                      rect(pg, "#login-button"), None, None))
        tasks.append(("t07_test1_search_type", "test_1.html",
                      "Type 'headphones' into the search box", "type",
                      ["headphones"], rect(pg, "#search-input"),
                      {"type": "input_value", "selector": "#search-input",
                       "value": "headphones"}, "headphones"))
        # canvas Add to Cart: drawn at (18,68,125,38) in 620x230 space
        cb = pg.locator("#uiCanvas").first.bounding_box()
        sx, sy = cb["width"] / 620.0, cb["height"] / 230.0
        tasks.append(("t08_test1_canvas_addtocart", "test_1.html",
                      "Click the Add to Cart button on the canvas panel",
                      "click", ["cart"],
                      [cb["x"] + 18 * sx, cb["y"] + 68 * sy,
                       cb["x"] + (18 + 125) * sx, cb["y"] + (68 + 38) * sy],
                      None, None))
        tasks.append(("t09_test1_shadow_upgrade", "test_1.html",
                      "Click the Upgrade plan button", "click", ["upgrade"],
                      [round(float(v), 1) for v in shadow_gt],
                      None, None))
        frame = next(f for f in pg.frames
                     if f.url == "about:srcdoc")
        fb = frame.locator('button:has-text("Approve")').first.bounding_box()
        tasks.append(("t10_test1_iframe_approve", "test_1.html",
                      "Click the Approve button in the embedded app",
                      "click", ["approve"],
                      [fb["x"], fb["y"], fb["x"] + fb["width"],
                       fb["y"] + fb["height"]], None, None))
        tasks.append(("t11_test1_dynamic_continue", "test_1.html",
                      "Click the Continue button", "click", ["continue"],
                      rect(pg, "#dynamic-continue"), None, None))

        # ---- test_page.html (1): anchor nav, hash checker ----
        goto("test_pages/test_page.html")
        tasks.append(("t12_testpage_nav_products", "test_pages/test_page.html",
                      "Open the Products section", "click", ["products"],
                      rect(pg, "#nav-products"),
                      {"type": "url_contains", "value": "#products"}, None))
        browser.close()

    out = []
    for (tid, rel, goal, act, kws, gt, check, tval) in tasks:
        out.append({
            "task_id": tid, "start_url": f"file://{rel}", "goal": goal,
            "expected_action": act, "expected_target_keywords": kws,
            "gt": [round(float(v), 1) for v in gt], "check": check,
            "type_value": tval, "viewport": [VW, VH]})
    path = REPO / "data" / "ab_tasks.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(out)} tasks to {path}")
    for t in out:
        print(f"  {t['task_id']}: {t['goal'][:48]:48s} gt={t['gt']} "
              f"check={bool(t['check'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
