#!/usr/bin/env python3
"""Generate a synthetic ScreenSpot-style sample from the local test page.

GT boxes come from live DOM extraction (getBoundingClientRect), so they are
real measurements, not hand-written. Used only as a smoke test for the eval
harness — NOT as a substitute for the real ScreenSpot benchmark.
"""

import logging
import sys
import json
from pathlib import Path
from PIL import Image
import io

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser.browser_manager import BrowserManager
from browser.dom_extractor import DOMExtractor


def main():
    logging.basicConfig(level=logging.INFO)
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "data" / "screenspot" / "smoke"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    browser = BrowserManager(headless=True, viewport_width=1280, viewport_height=720)
    try:
        browser.launch()
        browser.new_page()
        browser.navigate(f"file://{root / 'test_pages' / 'test_page.html'}")
        shot = browser.screenshot()
        (out_dir / "images" / "test_page.png").write_bytes(shot)
        elements = DOMExtractor().extract(browser.get_page())

        samples = []
        for el in elements[:12]:
            bb = el["bbox"]
            text = (el.get("text") or el.get("ariaLabel") or el.get("placeholder") or el["tag"])[:40]
            samples.append({
                "img_path": "images/test_page.png",
                "instruction": f"Click the {el['tag']} labeled '{text}'",
                "bbox": [bb["x"], bb["y"], bb["x"] + bb["width"], bb["y"] + bb["height"]],
                "platform": "web",
                "elem_type": "text" if el["tag"] in ("a", "button", "label") else "widget",
                "img_size": [1280, 720],
            })
        with open(out_dir / "smoke.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"Wrote {len(samples)} smoke samples to {out_dir / 'smoke.jsonl'}")
    finally:
        browser.close()


if __name__ == "__main__":
    main()
