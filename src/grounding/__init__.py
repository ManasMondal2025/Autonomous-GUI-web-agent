"""Grounding strategies package."""

from .base_grounder import BaseGrounder
from .som_overlay import SoMOverlay, overlay_screenshot, get_click_coordinates

__all__ = ["BaseGrounder", "SoMOverlay", "overlay_screenshot", "get_click_coordinates"]
