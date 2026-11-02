"""DOM extractor for identifying interactive elements in web pages."""

import logging
from typing import List, Dict, Any, Optional
from playwright.sync_api import Page

logger = logging.getLogger(__name__)


class DOMExtractor:
    """Extracts interactive DOM elements with bounding boxes."""

    INTERACTIVE_TAGS = {
        "button", "a", "input", "textarea", "select", "label",
        "details", "summary", "option", "form",
    }

    INPUT_TYPES = {
        "button", "submit", "reset", "checkbox", "radio",
        "text", "email", "password", "search", "tel", "url", "number",
    }

    INTERACTIVE_ROLES = {
        "button", "link", "textbox", "checkbox", "radio",
        "combobox", "menuitem", "tab", "switch", "slider",
        "spinbutton", "searchbox", "option", "listbox",
    }

    def __init__(
        self,
        only_visible: bool = True,
        only_in_viewport: bool = True,
        min_width: int = 5,
        min_height: int = 5,
    ):
        self.only_visible = only_visible
        self.only_in_viewport = only_in_viewport
        self.min_width = min_width
        self.min_height = min_height
        self._element_counter = 0

    def extract(self, page: Page) -> List[Dict[str, Any]]:
        """Extract interactive elements from the page."""
        self._element_counter = 0

        try:
            viewport = page.viewport_size
            scroll_x = page.evaluate("window.scrollX || window.pageXOffset || 0")
            scroll_y = page.evaluate("window.scrollY || window.pageYOffset || 0")
        except Exception as e:
            logger.error(f"Failed to get viewport/scroll info: {e}")
            return []

        selector = self._build_selector()
        elements_data = page.evaluate(
            """
            (selector) => {
                const results = [];
                const allElements = document.querySelectorAll(selector);
                let id = 0;
                for (const el of allElements) {
                    const rect = el.getBoundingClientRect();
                    const style = window.getComputedStyle(el);

                    let labelText = '';
                    try {
                        if (el.labels && el.labels.length > 0) {
                            labelText = Array.from(el.labels).map(l => (l.innerText || l.textContent || '').trim()).join(' ');
                        } else if (el.id) {
                            const lbl = document.querySelector(`label[for="${el.id}"]`);
                            if (lbl) labelText = (lbl.innerText || lbl.textContent || '').trim();
                        }
                        if (!labelText && el.tagName.toLowerCase() !== 'button') {
                            const parent = el.closest('label, div, p, td');
                            if (parent) {
                                if (parent.tagName.toLowerCase() === 'label') {
                                    labelText = (parent.innerText || parent.textContent || '').trim();
                                } else {
                                    const lbl = parent.querySelector('label');
                                    if (lbl) labelText = (lbl.innerText || lbl.textContent || '').trim();
                                }
                            }
                        }
                        if (!labelText && el.closest('td')) {
                            const td = el.closest('td');
                            const tr = td.closest('tr');
                            const table = tr ? tr.closest('table') : null;
                            if (table) {
                                const colIdx = Array.from(tr.children).indexOf(td);
                                const th = table.querySelectorAll('th')[colIdx];
                                if (th) labelText = (th.innerText || th.textContent || '').trim();
                            }
                        }
                    } catch (e) {}

                    results.push({
                        id: id++,
                        htmlId: el.id || el.getAttribute('id') || '',
                        label: labelText,
                        tag: el.tagName.toLowerCase(),
                        type: el.getAttribute('type') || '',
                        role: el.getAttribute('role') || el.getAttribute('aria-role') || '',
                        text: (el.innerText || el.textContent || '').trim().substring(0, 200),
                        ariaLabel: el.getAttribute('aria-label') || '',
                        placeholder: el.getAttribute('placeholder') || '',
                        title: el.getAttribute('title') || '',
                        value: el.value || '',
                        name: el.getAttribute('name') || '',
                        href: el.getAttribute('href') || '',
                        bbox: {
                            x: rect.x,
                            y: rect.y,
                            width: rect.width,
                            height: rect.height,
                        },
                        visible: style.display !== 'none' &&
                                 style.visibility !== 'hidden' &&
                                 parseFloat(style.opacity) > 0,
                        enabled: !el.disabled,
                        interactable: !el.disabled && style.pointerEvents !== 'none',
                        zIndex: parseInt(style.zIndex) || 0,
                    });
                }
                return results;
            }
            """,
            selector,
        )

        filtered_elements = self._filter_elements(
            elements_data, viewport, scroll_x, scroll_y
        )

        indexed_elements = []
        for idx, el in enumerate(filtered_elements):
            el["id"] = idx
            indexed_elements.append(el)

        logger.info(f"Extracted {len(indexed_elements)} interactive elements")
        return indexed_elements

    def _build_selector(self) -> str:
        """Build a CSS selector for interactive elements."""
        tag_selectors = ", ".join(self.INTERACTIVE_TAGS)
        role_selectors = ", ".join(
            [f"[role='{r}']" for r in self.INTERACTIVE_ROLES]
        )
        return f"{tag_selectors}, {role_selectors}, [onclick], [tabindex]"

    def _filter_elements(
        self,
        elements: List[Dict[str, Any]],
        viewport: Dict[str, int],
        scroll_x: int,
        scroll_y: int,
    ) -> List[Dict[str, Any]]:
        """Filter elements based on visibility, size, and position."""
        filtered = []
        for el in elements:
            if self.only_visible and not el.get("visible", False):
                continue
            if not el.get("enabled", False):
                continue
            if not el.get("interactable", False):
                continue

            bbox = el.get("bbox", {})
            w = bbox.get("width", 0)
            h = bbox.get("height", 0)

            if w < self.min_width or h < self.min_height:
                continue

            if self.only_in_viewport:
                x, y = bbox.get("x", 0), bbox.get("y", 0)
                if (x + w <= 0 or y + h <= 0 or
                    x >= viewport["width"] or y >= viewport["height"]):
                    continue
                # The click point is the bbox center — it must be on-screen.
                # An element whose top edge is visible but whose center is
                # below the fold (e.g. tall form at page bottom) is not
                # clickable without scrolling, so exclude it.
                cx = x + w / 2.0
                cy = y + h / 2.0
                if not (0 <= cx < viewport["width"] and 0 <= cy < viewport["height"]):
                    continue

            filtered.append(el)

        filtered.sort(key=lambda e: (
            -(e.get("bbox", {}).get("y", 0)),
            e.get("bbox", {}).get("x", 0)
        ))

        return filtered

    def get_element_center(self, element: Dict[str, Any]) -> tuple:
        """Get the center coordinates of an element."""
        bbox = element.get("bbox", {})
        x = bbox.get("x", 0) + bbox.get("width", 0) / 2
        y = bbox.get("y", 0) + bbox.get("height", 0) / 2
        return (x, y)

    def get_element_at_point(
        self, elements: List[Dict[str, Any]], x: float, y: float
    ) -> Optional[Dict[str, Any]]:
        """Find the topmost element at a given point."""
        candidates = []
        for el in elements:
            bbox = el.get("bbox", {})
            ex, ey = bbox.get("x", 0), bbox.get("y", 0)
            ew, eh = bbox.get("width", 0), bbox.get("height", 0)
            if ex <= x <= ex + ew and ey <= y <= ey + eh:
                candidates.append(el)

        if not candidates:
            return None

        candidates.sort(key=lambda e: e.get("zIndex", 0), reverse=True)
        return candidates[0]
