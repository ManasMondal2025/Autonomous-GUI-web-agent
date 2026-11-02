#!/usr/bin/env python3
"""Phase 2 demo: SoM overlay + both grounders (interchangeable) on the test page."""

import sys
import json
from pathlib import Path
from PIL import Image
import io

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser.browser_manager import BrowserManager
from browser.dom_extractor import DOMExtractor
from grounding.som_overlay import overlay_screenshot, save_mapping, get_click_coordinates
from grounding.coordinate_grounder import CoordinateGrounder
from grounding.som_grounder import SoMGrounder
from models.base_vlm import MockVLM


def main():
    project_root = Path(__file__).resolve().parents[1]
    test_page = project_root / "test_pages" / "test_page.html"
    output_dir = project_root / "data" / "phase2_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 2 Demo: SoM Overlay + Interchangeable Grounders")
    print("=" * 60)

    browser = BrowserManager(headless=True, viewport_width=1280, viewport_height=720)
    try:
        browser.launch()
        browser.new_page()
        browser.navigate(f"file://{test_page}")
        screenshot_bytes = browser.screenshot()
        image = Image.open(io.BytesIO(screenshot_bytes))
        print(f"\n[1/5] Screenshot: {image.size}")

        extractor = DOMExtractor(only_visible=True, only_in_viewport=True)
        elements = extractor.extract(browser.get_page())
        print(f"[2/5] Extracted {len(elements)} clickable elements (center on-screen)")

        # SoM overlay
        annotated, mapping = overlay_screenshot(image, elements)
        assert annotated.size == image.size, "Coordinate invariant violated!"
        ann_path = output_dir / "som_annotated.png"
        annotated.save(ann_path)
        save_mapping(mapping, output_dir / "som_mapping.json")
        print(f"[3/5] SoM overlay saved: {ann_path} ({len(mapping)} markers)")
        print(f"      Mapping saved: som_mapping.json")

        # Show first few markers
        for mid in sorted(mapping.keys(), key=int)[:6]:
            info = mapping[mid]
            print(f"      [{mid}] <{info['tag']}> '{info['text'][:30]}' center={info['center']}")

        # SoM grounder with mock VLM (pretend the model picks marker 0)
        first_id = elements[0]["id"]
        som_vlm = MockVLM(responses=[
            json.dumps({"action": "click", "target_id": first_id, "reason": "first element"})
        ])
        som_grounder = SoMGrounder(vlm=som_vlm)
        som_pred = som_grounder.ground(image, "click the first element", elements)
        som_action = som_grounder.to_action(som_pred)
        print(f"\n[4/5] SoMGrounder → action={som_action['action_type']} "
              f"target_id={som_action['target_id']} "
              f"xy=({som_action['x']:.0f},{som_action['y']:.0f})")

        # Coordinate grounder with mock VLM (same target, raw coords)
        cx, cy = get_click_coordinates(mapping, first_id)
        coord_vlm = MockVLM(responses=[
            json.dumps({"action": "click", "x": int(cx), "y": int(cy), "reason": "same target"})
        ])
        coord_grounder = CoordinateGrounder(vlm=coord_vlm)
        coord_pred = coord_grounder.ground(image, "click the first element", elements)
        coord_action = coord_grounder.to_action(coord_pred)
        print(f"[5/5] CoordinateGrounder → action={coord_action['action_type']} "
              f"xy=({coord_action['x']},{coord_action['y']})")

        # Both resolve to the same point → proves interchangeability
        assert abs(som_action["x"] - coord_action["x"]) < 2
        assert abs(som_action["y"] - coord_action["y"]) < 2
        print("\n✓ Both grounders resolve to the same click point — interchangeable.")

        with open(output_dir / "grounder_outputs.json", "w") as f:
            json.dump({
                "som": {k: v for k, v in som_action.items() if k != "raw_prediction"},
                "coordinate": {k: v for k, v in coord_action.items() if k != "raw_prediction"},
            }, f, indent=2, default=str)

        print("=" * 60)
        print("Phase 2 complete!")
        print(f"Outputs in: {output_dir}")
    finally:
        browser.close()


if __name__ == "__main__":
    main()
