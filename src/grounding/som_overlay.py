"""Set-of-Mark (SoM) prompting utilities and overlay generation.

SoM (Yang et al., 2023) turns grounding from continuous coordinate prediction
into discrete classification by overlaying numbered markers on each candidate
element. The model picks a number; the system looks up the bbox.

Critical invariant: the overlay MUST NOT shift the underlying coordinate
system. Markers are drawn on a copy of the screenshot so click coordinates
remain valid in the original image space.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
import io

logger = logging.getLogger(__name__)


DEFAULT_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",
    "#1ABC9C", "#E67E22", "#34495E", "#16A085", "#27AE60",
    "#2980B9", "#8E44AD", "#F1C40F", "#D35400", "#C0392B",
    "#7F8C8D", "#2C3E50", "#BDC3C7", "#7D3C98", "#48C9B0",
]

# Distinct background/foreground combinations for high contrast
MARKER_STYLES = [
    {"bg": "#E74C3C", "fg": "#FFFFFF"},
    {"bg": "#3498DB", "fg": "#FFFFFF"},
    {"bg": "#2ECC71", "fg": "#FFFFFF"},
    {"bg": "#F39C12", "fg": "#000000"},
    {"bg": "#9B59B6", "fg": "#FFFFFF"},
    {"bg": "#1ABC9C", "fg": "#000000"},
    {"bg": "#E67E22", "fg": "#FFFFFF"},
    {"bg": "#34495E", "fg": "#FFFFFF"},
    {"bg": "#16A085", "fg": "#FFFFFF"},
    {"bg": "#27AE60", "fg": "#FFFFFF"},
]


def load_default_font(size: int = 12) -> ImageFont.ImageFont:
    """Load a default font, falling back to PIL default if not available."""
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in font_candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


class SoMOverlay:
    """Generates Set-of-Mark annotated screenshots with stable coordinate mapping."""

    def __init__(
        self,
        marker_size: int = 18,
        border_width: int = 2,
        show_bbox: bool = True,
        show_text_label: bool = True,
        max_label_length: int = 30,
        font: Optional[ImageFont.ImageFont] = None,
    ):
        self.marker_size = marker_size
        self.border_width = border_width
        self.show_bbox = show_bbox
        self.show_text_label = show_text_label
        self.max_label_length = max_label_length
        self.font = font or load_default_font(14)
        self.label_font = font or load_default_font(11)

    def overlay(
        self,
        screenshot: Image.Image,
        elements: List[Dict[str, Any]],
    ) -> Tuple[Image.Image, Dict[str, Dict[str, Any]]]:
        """Overlay numbered markers on a screenshot.

        Args:
            screenshot: Original PIL Image.
            elements: List of DOM element dicts (must contain 'id' and 'bbox').

        Returns:
            (annotated_image, mapping) where mapping[marker_id] contains:
                - element_id, bbox (x,y,w,h), center (cx,cy), text, tag, etc.
            The annotated image has identical dimensions to the input; pixel
            coordinates in `mapping` refer to the ORIGINAL screenshot space
            (and therefore to the annotated image, which is a pixel-aligned copy).
        """
        if screenshot.mode != "RGB":
            screenshot = screenshot.convert("RGB")

        annotated = screenshot.copy()
        draw = ImageDraw.Draw(annotated)
        mapping: Dict[str, Dict[str, Any]] = {}

        for element in elements:
            marker_id = element["id"]
            bbox = element["bbox"]
            x = int(round(bbox["x"]))
            y = int(round(bbox["y"]))
            w = int(round(bbox["width"]))
            h = int(round(bbox["height"]))

            if w <= 0 or h <= 0:
                continue

            style = MARKER_STYLES[marker_id % len(MARKER_STYLES)]
            marker_color = style["bg"]
            text_color = style["fg"]

            if self.show_bbox:
                for offset in range(self.border_width):
                    draw.rectangle(
                        [x - offset, y - offset, x + w + offset, y + h + offset],
                        outline=marker_color,
                        width=1,
                    )

            marker_x, marker_y = self._compute_marker_position(x, y, w, h, annotated.size)
            marker_radius = self.marker_size // 2

            draw.ellipse(
                [
                    marker_x - marker_radius,
                    marker_y - marker_radius,
                    marker_x + marker_radius,
                    marker_y + marker_radius,
                ],
                fill=marker_color,
                outline="#000000",
                width=1,
            )

            label = str(marker_id)
            self._draw_centered_text(
                draw, label, marker_x, marker_y, text_color, self.font
            )

            text_content = self._extract_text(element)
            label_position = None
            if self.show_text_label and text_content:
                label_position = self._draw_text_label(
                    draw, text_content, marker_x, marker_y,
                    marker_color, annotated.size,
                )

            center_x = x + w / 2.0
            center_y = y + h / 2.0

            mapping[str(marker_id)] = {
                "element_id": element.get("id"),
                "tag": element.get("tag"),
                "type": element.get("type", ""),
                "role": element.get("role", ""),
                "text": text_content,
                "aria_label": element.get("ariaLabel", ""),
                "placeholder": element.get("placeholder", ""),
                "bbox": {
                    "x": x, "y": y, "width": w, "height": h,
                },
                "bbox_array": [x, y, x + w, y + h],
                "center": [center_x, center_y],
                "marker_position": [marker_x, marker_y],
                "label_position": label_position,
            }

        return annotated, mapping

    def _compute_marker_position(
        self, x: int, y: int, w: int, h: int,
        img_size: Tuple[int, int],
    ) -> Tuple[int, int]:
        """Place the marker at the top-left corner of the bbox, but keep it on-canvas."""
        img_w, img_h = img_size
        marker_x = max(self.marker_size, x)
        marker_y = max(self.marker_size, y)
        marker_x = min(marker_x, img_w - self.marker_size)
        marker_y = min(marker_y, img_h - self.marker_size)
        return marker_x, marker_y

    def _draw_centered_text(
        self, draw: ImageDraw.ImageDraw, text: str,
        cx: int, cy: int, color: str, font: ImageFont.ImageFont,
    ) -> None:
        """Draw text centered at (cx, cy)."""
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = draw.textsize(text, font=font)

        draw.text(
            (cx - text_w / 2 - bbox[0] if 'bbox' in locals() else cx - text_w / 2,
             cy - text_h / 2 - bbox[1] if 'bbox' in locals() else cy - text_h / 2),
            text, fill=color, font=font,
        )

    def _draw_text_label(
        self, draw: ImageDraw.ImageDraw, text: str,
        marker_x: int, marker_y: int, bg_color: str,
        img_size: Tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        """Draw a small text label below the marker."""
        img_w, img_h = img_size
        truncated = text[: self.max_label_length] + ("..." if len(text) > self.max_label_length else "")

        try:
            text_bbox = draw.textbbox((0, 0), truncated, font=self.label_font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except AttributeError:
            text_w, text_h = draw.textsize(truncated, font=self.label_font)

        padding = 3
        label_x = marker_x
        label_y = marker_y + self.marker_size // 2 + 2

        if label_x + text_w + 2 * padding > img_w:
            label_x = img_w - text_w - 2 * padding
        if label_y + text_h + 2 * padding > img_h:
            label_y = marker_y - self.marker_size // 2 - text_h - 2 * padding - 2

        draw.rectangle(
            [label_x, label_y, label_x + text_w + 2 * padding, label_y + text_h + 2 * padding],
            fill=bg_color,
        )
        draw.text(
            (label_x + padding, label_y + padding),
            truncated, fill="#FFFFFF", font=self.label_font,
        )
        return [label_x, label_y]

    def _extract_text(self, element: Dict[str, Any]) -> str:
        """Extract a human-readable text description for an element."""
        for key in ("label", "ariaLabel", "text", "placeholder", "title", "name", "htmlId", "value"):
            val = element.get(key, "")
            if val and str(val).strip():
                return str(val).strip()
        return element.get("tag", "")


def overlay_screenshot(
    screenshot: Image.Image,
    elements: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
    **kwargs,
) -> Tuple[Image.Image, Dict[str, Dict[str, Any]]]:
    """Convenience function: create a SoM overlay and optionally save it.

    Returns the annotated image (same dimensions as input) and the marker mapping.
    """
    som = SoMOverlay(**kwargs)
    annotated, mapping = som.overlay(screenshot, elements)
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(output_path)
    return annotated, mapping


def save_mapping(mapping: Dict[str, Dict[str, Any]], path: Path) -> None:
    """Save the marker mapping to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for marker_id, info in mapping.items():
        entry = dict(info)
        for key, value in entry.items():
            if isinstance(value, tuple):
                entry[key] = list(value)
        serializable[marker_id] = entry
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_mapping(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load a marker mapping from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def get_click_coordinates(
    mapping: Dict[str, Dict[str, Any]], marker_id: int,
) -> Tuple[float, float]:
    """Resolve a marker ID to click coordinates (center of bbox)."""
    key = str(marker_id)
    if key not in mapping:
        raise KeyError(f"Marker ID {marker_id} not found in mapping")
    center = mapping[key]["center"]
    return (center[0], center[1])


def verify_coordinate_invariant(
    original: Image.Image, annotated: Image.Image, tolerance: int = 2,
) -> bool:
    """Check that the annotated image has the same dimensions as the original.

    This is a critical sanity check: the SoM overlay must not resize or
    transform the screenshot, otherwise click coordinates become invalid.
    """
    if original.size != annotated.size:
        return False
    return True
