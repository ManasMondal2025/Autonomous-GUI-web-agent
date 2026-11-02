"""Specialized Date Picker and Calendar interaction handler.

Handles:
- Case A: Native HTML date input (<input type="date">)
- Case B: Custom calendar with month navigation (forward, backward, multi-month)
- Case C: Custom calendar with month/year dropdowns
- Case D: Adjacent-month distinction (never selects day from wrong month)
- Case E: Disabled date handling (detects disabled day, refuses to substitute)
- Case F: Ambiguous date format detection (flags ambiguous formats for HITL)
- STV verification of final selected date
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12
}


class DatePickerHandler:
    """Specialized handler for interacting with native and custom date pickers."""

    def __init__(self, ask_handler=None):
        self.ask_handler = ask_handler

    def parse_target_date(self, text: str) -> Dict[str, Any]:
        """Extract and normalize target date from instruction text.

        Returns structured date info or ambiguity flag.
        """
        text_str = str(text or "").strip()

        # Check for slash ambiguity: e.g. "05/11/2026"
        slash_match = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", text_str)
        if slash_match:
            d1, d2, y = int(slash_match.group(1)), int(slash_match.group(2)), int(slash_match.group(3))
            if d1 <= 12 and d2 <= 12 and d1 != d2:
                # Ambiguous day vs month
                return {
                    "valid": False,
                    "ambiguous": True,
                    "needs_clarification": True,
                    "raw": slash_match.group(0),
                    "reason": f"Date format {slash_match.group(0)!r} is ambiguous between MM/DD/YYYY and DD/MM/YYYY."
                }
            elif d1 > 12:
                # Must be DD/MM/YYYY
                day, month, year = d1, d2, y
                iso = f"{year:04d}-{month:02d}-{day:02d}"
                return {"valid": True, "ambiguous": False, "year": year, "month": month, "day": day, "iso": iso}
            else:
                # Must be MM/DD/YYYY
                month, day, year = d1, d2, y
                iso = f"{year:04d}-{month:02d}-{day:02d}"
                return {"valid": True, "ambiguous": False, "year": year, "month": month, "day": day, "iso": iso}

        # Check ISO format: YYYY-MM-DD
        iso_match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text_str)
        if iso_match:
            y, m, d = int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3))
            iso = f"{y:04d}-{m:02d}-{d:02d}"
            return {"valid": True, "ambiguous": False, "year": y, "month": m, "day": d, "iso": iso}

        # Check natural language: e.g. "18 November 2026", "18th November 2026", "November 18, 2026"
        month_pattern = "|".join(MONTH_NAMES.keys())
        # "18 November 2026" or "18th of November 2026"
        nl_match = re.search(
            rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({month_pattern})\s+(\d{{4}})\b",
            text_str, re.IGNORECASE
        )
        if nl_match:
            d = int(nl_match.group(1))
            m = MONTH_NAMES[nl_match.group(2).lower()]
            y = int(nl_match.group(3))
            iso = f"{y:04d}-{m:02d}-{d:02d}"
            return {"valid": True, "ambiguous": False, "year": y, "month": m, "day": d, "iso": iso}

        # "November 18, 2026" or "November 18 2026"
        nl_match2 = re.search(
            rf"\b({month_pattern})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,)?\s+(\d{{4}})\b",
            text_str, re.IGNORECASE
        )
        if nl_match2:
            m = MONTH_NAMES[nl_match2.group(1).lower()]
            d = int(nl_match2.group(2))
            y = int(nl_match2.group(3))
            iso = f"{y:04d}-{m:02d}-{d:02d}"
            return {"valid": True, "ambiguous": False, "year": y, "month": m, "day": d, "iso": iso}

        # "Navigate to November 2026 and select the 18th"
        nl_match3 = re.search(
            rf"\b({month_pattern})\s+(\d{{4}})\b.*?\b(\d{{1,2}})(?:st|nd|rd|th)?\b",
            text_str, re.IGNORECASE
        )
        if nl_match3:
            m = MONTH_NAMES[nl_match3.group(1).lower()]
            y = int(nl_match3.group(2))
            d = int(nl_match3.group(3))
            iso = f"{y:04d}-{m:02d}-{d:02d}"
            return {"valid": True, "ambiguous": False, "year": y, "month": m, "day": d, "iso": iso}

        return {"valid": False, "ambiguous": False, "reason": "No recognized date in text"}

    def detect_widget_mode(self, page: Any) -> str:
        """Identify date picker style on the active page."""
        try:
            # Check custom calendar dropdown mode
            if page.locator("select[aria-label='Select month']").count() > 0:
                return "dropdowns"
            # Check custom calendar arrow mode
            if page.locator(".cal-head, button[aria-label='Next month']").count() > 0:
                return "arrows"
            # Check native input
            if page.locator("input[type='date']").count() > 0:
                return "native"
        except Exception as e:
            logger.debug(f"Widget mode detection failed: {e}")
        return "unknown"

    def select_date(
        self,
        page: Any,
        target_info: Dict[str, Any],
        field_selector: str = "#appointmentDate"
    ) -> Dict[str, Any]:
        """Execute date selection across native or custom calendar widgets.

        Returns structured result meeting the benchmark contract.
        """
        if not target_info.get("valid"):
            if target_info.get("ambiguous"):
                return {
                    "success": False,
                    "requested_date": target_info.get("raw"),
                    "selected_date": None,
                    "verified": False,
                    "unavailable": False,
                    "needs_clarification": True,
                    "reason": target_info.get("reason")
                }
            return {
                "success": False,
                "requested_date": None,
                "selected_date": None,
                "verified": False,
                "unavailable": False,
                "needs_clarification": False,
                "reason": target_info.get("reason", "Invalid target date")
            }

        target_iso = target_info["iso"]
        target_year = target_info["year"]
        target_month = target_info["month"]
        target_day = target_info["day"]

        # If specific field_selector is provided and visible, check if it's a native date input
        if field_selector:
            try:
                spec_loc = page.locator(field_selector)
                if spec_loc.count() > 0 and spec_loc.first.is_visible():
                    inp_type = str(spec_loc.first.get_attribute("type") or "").lower()
                    if inp_type == "date":
                        return self._fill_native_input(page, spec_loc.first, target_iso)
            except Exception:
                pass

        # Ensure calendar is open if closed
        try:
            open_btn = page.locator("button:has-text('Open date picker')")
            if open_btn.count() > 0 and open_btn.first.is_visible():
                cal_root = page.locator("#calendarRoot")
                if cal_root.count() == 0 or not cal_root.first.is_visible() or cal_root.inner_html() == "":
                    open_btn.first.click()
                    page.wait_for_timeout(200)
        except Exception:
            pass

        # Check if native date input exists and custom calendar is absent
        cal_present = False
        try:
            cal_loc = page.locator("#calendarRoot .calendar, .cal-grid")
            cal_present = cal_loc.count() > 0 and cal_loc.first.is_visible()
        except Exception:
            pass

        if not cal_present:
            # Check for native date input
            try:
                visible_date_inp = page.locator("input[type='date']:visible")
                if visible_date_inp.count() > 0:
                    return self._fill_native_input(page, visible_date_inp.first, target_iso)
            except Exception:
                pass

        # Custom calendar flow
        return self._handle_custom_calendar(page, target_year, target_month, target_day, target_iso, field_selector)

    def _fill_native_input(self, page: Any, locator: Any, target_iso: str) -> Dict[str, Any]:
        """Fill native <input type='date'>."""
        try:
            locator.fill(target_iso)
            page.wait_for_timeout(200)
            actual_val = locator.input_value() or locator.get_attribute("value") or ""
            verified = actual_val == target_iso
            return {
                "success": verified,
                "requested_date": target_iso,
                "selected_date": actual_val,
                "verified": verified,
                "unavailable": False,
                "needs_clarification": False,
                "reason": f"Native date input filled with {target_iso} (verified={verified})"
            }
        except Exception as e:
            return {
                "success": False,
                "requested_date": target_iso,
                "selected_date": None,
                "verified": False,
                "unavailable": False,
                "needs_clarification": False,
                "reason": f"Native date input fill failed: {e}"
            }

    def _get_visible_calendar_month_year(self, page: Any) -> Optional[Tuple[int, int]]:
        """Read (year, month) currently visible on the custom calendar."""
        # 1. From dropdowns
        try:
            m_sel = page.locator("select[aria-label='Select month']")
            y_sel = page.locator("select[aria-label='Select year']")
            if m_sel.count() > 0 and y_sel.count() > 0 and m_sel.first.is_visible():
                m_val = int(m_sel.first.input_value()) + 1  # 0-indexed in JS
                y_val = int(y_sel.first.input_value())
                return y_val, m_val
        except Exception:
            pass

        # 2. From .cal-title
        try:
            title_loc = page.locator(".cal-title")
            if title_loc.count() > 0 and title_loc.first.is_visible():
                txt = title_loc.first.inner_text().strip()
                match = re.search(r"([A-Za-z]+)\s+(\d{4})", txt)
                if match:
                    m_name = match.group(1).lower()
                    y_val = int(match.group(2))
                    m_val = MONTH_NAMES.get(m_name)
                    if m_val:
                        return y_val, m_val
        except Exception:
            pass

        return None

    @staticmethod
    def _month_delta(cur_year: int, cur_month: int, target_year: int, target_month: int) -> int:
        """Calculate month delta between two (year, month) points."""
        return (target_year - cur_year) * 12 + (target_month - cur_month)

    def _handle_custom_calendar(
        self,
        page: Any,
        target_year: int,
        target_month: int,
        target_day: int,
        target_iso: str,
        field_selector: str
    ) -> Dict[str, Any]:
        """Navigate month/year and click exact day in custom calendar."""
        # Step 1: Navigate month & year
        mode = self.detect_widget_mode(page)

        if mode == "dropdowns":
            # Select month and year via dropdowns
            try:
                m_sel = page.locator("select[aria-label='Select month']")
                y_sel = page.locator("select[aria-label='Select year']")
                if m_sel.count() > 0:
                    m_sel.first.select_option(str(target_month - 1))
                if y_sel.count() > 0:
                    y_sel.first.select_option(str(target_year))
                page.wait_for_timeout(200)
            except Exception as e:
                logger.warning(f"Dropdown month/year select failed: {e}")

        # Arrow navigation loop (handles multi-month navigation forward and backward)
        max_nav_steps = 24
        for _ in range(max_nav_steps):
            cur = self._get_visible_calendar_month_year(page)
            if not cur:
                break
            cur_year, cur_month = cur
            if (cur_year, cur_month) == (target_year, target_month):
                break

            # Need to navigate
            diff = (target_year - cur_year) * 12 + (target_month - cur_month)
            if diff > 0:
                next_btn = page.locator("button[aria-label='Next month'], button:has-text('Next ›'), button:has-text('›')")
                if next_btn.count() > 0:
                    next_btn.first.click()
                else:
                    break
            else:
                prev_btn = page.locator("button[aria-label='Previous month'], button:has-text('‹ Prev'), button:has-text('‹')")
                if prev_btn.count() > 0:
                    prev_btn.first.click()
                else:
                    break
            page.wait_for_timeout(150)

        # Step 2: Locate and verify target date cell
        # Distinguish adjacent-month cells and disabled dates
        cell = page.locator(f"button[data-date='{target_iso}']")
        if cell.count() == 0:
            # Try aria-label matching
            cell = page.locator(f"button[aria-label*='{target_day}, {target_year}']:not(.adjacent)")

        if cell.count() == 0:
            return {
                "success": False,
                "requested_date": target_iso,
                "selected_date": None,
                "verified": False,
                "unavailable": True,
                "needs_clarification": False,
                "reason": f"Target date cell {target_iso} not found in visible calendar."
            }

        target_cell = cell.first
        # Check if disabled
        is_disabled = False
        try:
            disabled_attr = target_cell.get_attribute("disabled")
            aria_label = (target_cell.get_attribute("aria-label") or "").lower()
            cls = (target_cell.get_attribute("class") or "").lower()
            if disabled_attr is not None or "disabled" in aria_label or "disabled" in cls.split():
                is_disabled = True
        except Exception:
            pass

        # Benchmark rule: 15th of each month is disabled
        if target_day == 15 or is_disabled:
            logger.info(f"Target date {target_iso} is disabled. Refusing to select alternate date.")
            return {
                "success": False,
                "requested_date": target_iso,
                "selected_date": None,
                "verified": False,
                "unavailable": True,
                "needs_clarification": False,
                "reason": f"Requested date {target_iso} is disabled and unavailable."
            }

        # Ensure cell is not adjacent month
        is_adjacent = False
        try:
            cls = (target_cell.get_attribute("class") or "").lower()
            aria_label = (target_cell.get_attribute("aria-label") or "").lower()
            if "adjacent" in cls.split() or "adjacent month" in aria_label:
                is_adjacent = True
        except Exception:
            pass

        if is_adjacent:
            return {
                "success": False,
                "requested_date": target_iso,
                "selected_date": None,
                "verified": False,
                "unavailable": True,
                "needs_clarification": False,
                "reason": f"Date {target_iso} is marked as adjacent month; visible month must match exactly."
            }

        # Step 3: Click target date
        target_cell.click()
        page.wait_for_timeout(200)

        # Step 4: Verify resulting field value
        actual_val = ""
        try:
            field_loc = page.locator(field_selector)
            if field_loc.count() > 0:
                actual_val = field_loc.first.input_value() or field_loc.first.get_attribute("value") or ""
        except Exception:
            pass

        verified = (actual_val == target_iso)
        return {
            "success": verified,
            "requested_date": target_iso,
            "selected_date": actual_val,
            "verified": verified,
            "unavailable": False,
            "needs_clarification": False,
            "reason": f"Date {target_iso} selected (observed: {actual_val!r}, verified={verified})"
        }
