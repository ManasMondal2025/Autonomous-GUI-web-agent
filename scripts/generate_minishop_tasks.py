#!/usr/bin/env python3
"""Generate the 5-task MiniShop benchmark.

Opens each page in a real browser at the fixed viewport, reads live DOM
bounding boxes, and bakes the true element centers into per-task
mock_responses. Re-run if the pages' CSS changes.

Output: data/webarena_minishop.json
"""

import logging  # stdlib first
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser.browser_manager import BrowserManager
from browser.dom_extractor import DOMExtractor


def center_of(elements, key, value):
    el = next(e for e in elements if e.get(key) == value)
    bb = el["bbox"]
    return int(bb["x"] + bb["width"] / 2), int(bb["y"] + bb["height"] / 2)


def click(x, y, reason):
    return json.dumps({"action": "click", "x": x, "y": y, "reason": reason})


def main():
    logging.basicConfig(level=logging.WARNING)
    root = Path(__file__).resolve().parents[1]
    shop = root / "test_pages" / "minishop"
    base = f"file://{shop}"

    browser = BrowserManager(headless=True, viewport_width=1280, viewport_height=720)
    try:
        browser.launch()
        browser.new_page()
        ex = DOMExtractor()

        browser.navigate(f"{base}/index.html")
        home = ex.extract(browser.get_page())
        sx, sy = center_of(home, "placeholder", "Search products...")
        lx, ly = center_of(home, "text", "Laptop Pro")
        nx, ny = center_of(home, "text", "Cart (0)")

        browser.navigate(f"{base}/product.html?id=laptop")
        prod = ex.extract(browser.get_page())
        ax, ay = center_of(prod, "text", "Add to Cart")
        vx, vy = center_of(prod, "text", "View Cart")
    finally:
        browser.close()

    done = json.dumps({"action": "done", "reason": "task complete"})
    tasks = [
        {"task_id": "shop1_search_type",
         "start_url": f"{base}/index.html",
         "goal": "Type 'laptop' into the search box",
         "check": {"type": "input_value", "selector": "#search-input", "value": "laptop"},
         "mock_responses": [
             json.dumps({"action": "type", "x": sx, "y": sy,
                         "value": "laptop", "reason": "type query"}), done]},
        {"task_id": "shop2_open_product",
         "start_url": f"{base}/index.html",
         "goal": "Open the Laptop Pro product page",
         "check": {"type": "url_contains", "value": "product.html"},
         "mock_responses": [click(lx, ly, "open product"), done]},
        {"task_id": "shop3_add_to_cart",
         "start_url": f"{base}/product.html?id=laptop",
         "goal": "Add Laptop Pro to the cart",
         "check": {"type": "element_text_contains", "selector": "#cart-status",
                   "value": "Added to cart"},
         "mock_responses": [click(ax, ay, "add to cart"), done]},
        {"task_id": "shop4_view_cart",
         "start_url": f"{base}/product.html?id=laptop",
         "goal": "Go to the cart page",
         "check": {"type": "url_contains", "value": "cart.html"},
         "mock_responses": [click(vx, vy, "view cart"), done]},
        {"task_id": "shop5_cart_total",
         "start_url": f"{base}/index.html",
         "goal": "Go to the cart page and confirm the $999 total is shown",
         "check": {"type": "all_of", "checks": [
             {"type": "url_contains", "value": "cart.html"},
             {"type": "element_text_contains", "selector": "#cart-total",
              "value": "$999"},
         ]},
         "mock_responses": [click(nx, ny, "open cart from nav"), done]},
    ]
    out = root / "data" / "webarena_minishop.json"
    out.write_text(json.dumps(tasks, indent=2))
    print(f"Wrote {len(tasks)} tasks to {out}")


if __name__ == "__main__":
    main()
