#!/usr/bin/env python3
"""Standalone CAPTCHA detection demo (no agent/HITL imports).

Examples:
  python scripts/captcha_demo.py --url file://test_pages/captcha_fixtures/recaptcha.html
  python scripts/captcha_demo.py --screenshot data/screenshots/step_001.png
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser.browser_manager import BrowserManager
from captcha.config import CaptchaConfig
from captcha.detector import CaptchaDetector
from utils.config import load_yaml_config

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _load_config(path: str | None) -> CaptchaConfig:
    if not path:
        return CaptchaConfig()
    data = load_yaml_config(Path(path))
    section = data.get("captcha_detection", data)
    return CaptchaConfig.from_dict(section)


def _print_result(result) -> None:
    d = result.to_dict()
    print("CAPTCHA DETECTION")
    print("-----------------")
    print(f"Detected: {'YES' if d['detected'] else 'NO'}")
    print(f"Blocking: {'YES' if d['blocking'] else 'NO'}")
    print(f"Type: {d['captcha_type'] or '-'}")
    print(f"Provider: {d['provider'] or '-'}")
    print(f"Confidence: {d['confidence']:.2f}")
    print("")
    print("Signals:")
    if not d["signals"]:
        print("  none")
    else:
        for s in d["signals"]:
            print(f"  \u2713 {s}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Standalone CAPTCHA detector demo")
    ap.add_argument("--url", default=None, help="URL or file:// page to inspect")
    ap.add_argument("--screenshot", default=None, help="Screenshot-only mode (PNG path)")
    ap.add_argument("--config", default=None, help="YAML config with captcha_detection section")
    ap.add_argument("--enable-ocr", action="store_true", help="Enable OCR fallback (needs paddleocr)")
    args = ap.parse_args()

    if not args.url and not args.screenshot:
        ap.error("provide --url and/or --screenshot")

    config = _load_config(args.config)
    if args.enable_ocr:
        config.ocr_detection = True

    ocr_backend = None
    if config.ocr_detection:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
            from browser.ocr import PaddleOCRBackend

            ocr_backend = PaddleOCRBackend()
        except Exception as e:
            print(f"[CAPTCHA] OCR requested but unavailable ({e}); continuing without OCR")

    detector = CaptchaDetector(config=config, ocr_backend=ocr_backend)

    if args.screenshot and not args.url:
        from captcha.layers import ocr_adapter, text

        shot = Path(args.screenshot).read_bytes()
        # Screenshot-only: OCR text path when available, else deterministic no-evidence.
        if ocr_backend is not None:
            info = ocr_adapter.detect_ocr(shot, ocr_backend)
            print("CAPTCHA DETECTION (screenshot-only, OCR)")
            print("-----------------")
            print(f"OCR signals: {info['signals'] or 'none'}")
            print(f"OCR text: {(info['ocr_text_excerpt'] or '-')[:300]}")
            return 0
        print("CAPTCHA DETECTION (screenshot-only, no OCR backend)")
        print("-----------------")
        print("Detected: NO (insufficient evidence without DOM/OCR)")
        return 0

    url = args.url
    if url.startswith("file://"):
        raw = url[len("file://"):]
        p = Path(raw)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        url = f"file://{p}"
    elif Path(url).exists():
        url = f"file://{Path(url).resolve()}"

    browser = BrowserManager(headless=True)
    try:
        browser.launch()
        page = browser.new_page()
        page.goto(url, wait_until="load")
        try:
            page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass
        result = detector.detect(page)
        _print_result(result)
        if args.screenshot:
            Path(args.screenshot).parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=args.screenshot)
            print(f"\nScreenshot saved to: {args.screenshot}")
        return 0
    finally:
        browser.close()


if __name__ == "__main__":
    raise SystemExit(main())
