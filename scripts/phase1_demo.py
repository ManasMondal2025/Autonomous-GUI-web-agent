#!/usr/bin/env python3
"""Phase 1 demo: Browser + Screenshot + DOM extraction."""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser.browser_manager import BrowserManager
from browser.dom_extractor import DOMExtractor
from browser.observer import Observer


def main():
    project_root = Path(__file__).resolve().parents[1]
    test_page = project_root / "test_pages" / "test_page.html"
    output_dir = project_root / "data" / "phase1_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 1 Demo: Browser + Screenshot + DOM Extraction")
    print("=" * 60)

    browser = BrowserManager(
        headless=True,
        viewport_width=1280,
        viewport_height=720,
        timeout_ms=15000,
    )

    try:
        print("\n[1/5] Launching browser...")
        browser.launch()
        print("  ✓ Browser launched")

        print("\n[2/5] Creating new page...")
        browser.new_page()
        print(f"  ✓ Viewport: {browser.get_viewport_size()}")

        print(f"\n[3/5] Navigating to test page...")
        browser.navigate(f"file://{test_page}")
        print(f"  ✓ URL: {browser.get_url()}")

        print("\n[4/5] Capturing screenshot...")
        screenshot_path = output_dir / "screenshot.png"
        browser.screenshot(path=screenshot_path)
        print(f"  ✓ Screenshot saved: {screenshot_path}")
        print(f"  ✓ File size: {screenshot_path.stat().st_size} bytes")

        print("\n[5/5] Extracting DOM elements...")
        observer = Observer(browser, save_dir=output_dir)
        observation = observer.observe(save_screenshot=True)

        print(f"  ✓ Found {len(observation['dom_elements'])} interactive elements")
        print(f"  ✓ Screenshot size: {observation['screenshot_size']}")
        print(f"  ✓ URL: {observation['url']}")

        print("\n--- Interactive Elements ---")
        for el in observation['dom_elements'][:10]:
            bbox = el['bbox']
            text = (el.get('text', '') or el.get('ariaLabel', '') or el.get('placeholder', ''))[:30]
            print(
                f"  [{el['id']:2d}] {el['tag']:10s} "
                f"pos=({bbox['x']:.0f},{bbox['y']:.0f}) "
                f"size=({bbox['width']:.0f}x{bbox['height']:.0f}) "
                f"text='{text}'"
            )
        if len(observation['dom_elements']) > 10:
            print(f"  ... and {len(observation['dom_elements']) - 10} more")

        elements_path = output_dir / "dom_elements.json"
        with open(elements_path, "w") as f:
            json.dump(
                [el for el in observation['dom_elements']],
                f,
                indent=2,
            )
        print(f"\n✓ DOM elements saved to: {elements_path}")

        viewport = browser.get_viewport_size()
        obs_size = observation['screenshot_size']
        if viewport['width'] == obs_size['width'] and viewport['height'] == obs_size['height']:
            print(f"\n✓ Coordinate alignment verified: viewport == screenshot")
        else:
            print(f"\n✗ WARNING: Mismatch! viewport={viewport} vs screenshot={obs_size}")

        print("\n" + "=" * 60)
        print("Phase 1 complete!")
        print("=" * 60)
        print(f"\nOutputs saved to: {output_dir}")
        print(f"  - screenshot.png")
        print(f"  - dom_elements.json")
        print(f"  - step_001.png + step_001.json (from Observer)")

    finally:
        browser.close()
        print("\nBrowser closed.")


if __name__ == "__main__":
    main()
