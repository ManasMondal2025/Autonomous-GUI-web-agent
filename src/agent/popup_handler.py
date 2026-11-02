"""Popup, modal, consent banner, and dialog dismissal handler.

Provides robust detection, classification, safe dismissal planning,
and verification for:
- Type A: Cookie consent banners (privacy-preserving choice by default)
- Type B: Blocking modals (safe close/cancel buttons, never destructive actions)
- Type C: Delayed popups (detected upon fresh observation)
- Type D: Non-blocking promotions (left untouched unless instructed)
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TYPE_CONSENT_BANNER = "consent_banner"
TYPE_BLOCKING_MODAL = "blocking_modal"
TYPE_NON_BLOCKING_PROMO = "non_blocking_promotion"
TYPE_NONE = "none"


class PopupHandler:
    """Specialized handler for popups, modals, consent banners, and dialogs."""

    def __init__(self, ask_handler=None):
        self.ask_handler = ask_handler
        self.dismissal_attempts: Dict[str, int] = {}
        self.max_attempts = 3

    def detect_popup(
        self,
        observation: Dict[str, Any],
        page: Optional[Any] = None,
        task_text: str = ""
    ) -> Dict[str, Any]:
        """Detect and classify any active popup or dialog on the page.

        Returns a dictionary describing the popup type, blocking status,
        and candidate actions.
        """
        dom_elements = observation.get("dom_elements", []) or []
        visible_text = str(observation.get("visible_text", "") or "").lower()

        # 1. Check live page state if available (deterministic DOM check)
        if page is not None:
            try:
                page_popup = self._detect_via_page(page, task_text)
                if page_popup.get("detected"):
                    return page_popup
            except Exception as e:
                logger.debug(f"Live page popup check failed: {e}")

        # 2. Check DOM elements from observation
        return self._detect_via_dom(dom_elements, visible_text, task_text)

    def _detect_via_page(self, page: Any, task_text: str) -> Dict[str, Any]:
        """Query live Playwright page for visible dialogs."""
        # Check Cookie banner
        cookie_sel = "#cookieBanner, #customizeBanner, .cookie-banner, [aria-label*='cookie' i]"
        for sel in cookie_sel.split(", "):
            loc = page.locator(sel)
            if loc.count() > 0:
                first = loc.first
                try:
                    if first.is_visible() and not self._has_hidden_class(first):
                        return self._classify_consent_banner(first, task_text)
                except Exception:
                    pass

        # Check Blocking Modal
        modal_sel = "#blockingModal, .modal-backdrop, [aria-modal='true'], dialog[open], [role='dialog'].modal"
        for sel in modal_sel.split(", "):
            loc = page.locator(sel)
            if loc.count() > 0:
                first = loc.first
                try:
                    if first.is_visible() and not self._has_hidden_class(first):
                        return self._classify_blocking_modal(first, task_text)
                except Exception:
                    pass

        # Check Non-blocking Promotion
        promo_sel = "#promo, [role='complementary'], .promo-panel, .promotion"
        for sel in promo_sel.split(", "):
            loc = page.locator(sel)
            if loc.count() > 0:
                first = loc.first
                try:
                    if first.is_visible() and not self._has_hidden_class(first):
                        return self._classify_promotion(first, task_text)
                except Exception:
                    pass

        return {"detected": False, "popup_type": TYPE_NONE, "is_blocking": False}

    def _has_hidden_class(self, locator: Any) -> bool:
        """Check if locator has class 'hidden' or display:none style."""
        try:
            cls = locator.get_attribute("class") or ""
            if "hidden" in cls.split():
                return True
            style = locator.get_attribute("style") or ""
            if "display: none" in style.lower() or "display:none" in style.lower():
                return True
            aria_hidden = locator.get_attribute("aria-hidden")
            if aria_hidden == "true":
                return True
        except Exception:
            pass
        return False

    def _classify_consent_banner(self, banner_loc: Any, task_text: str) -> Dict[str, Any]:
        """Build classification and safe dismissal plan for consent banner."""
        banner_id = "cookie_banner"
        try:
            banner_id = banner_loc.get_attribute("id") or "cookie_banner"
        except Exception:
            pass

        # Check task text for explicit preference
        task_lower = task_text.lower()
        prefer_accept = "accept all" in task_lower or ("accept" in task_lower and "cookie" in task_lower and "reject" not in task_lower)
        prefer_customize = "customize" in task_lower

        # Find buttons inside the banner
        buttons = []
        try:
            btn_locs = banner_loc.locator("button")
            for i in range(btn_locs.count()):
                b = btn_locs.nth(i)
                txt = b.inner_text().strip()
                aria = b.get_attribute("aria-label") or ""
                cls = b.get_attribute("class") or ""
                buttons.append({"text": txt, "aria": aria, "class": cls, "index": i})
        except Exception:
            pass

        # Select target button
        action = None
        target_name = ""

        if prefer_accept:
            target_name = "Accept all"
        elif prefer_customize and banner_id == "cookieBanner":
            target_name = "Customize"
        else:
            # Privacy-preserving default:
            # 1. Reject optional / Reject all / Necessary only
            # 2. Customize if no direct reject
            # 3. Close (x)
            reject_candidates = [
                b for b in buttons if any(
                    k in b["text"].lower() or k in b["aria"].lower()
                    for k in ("reject optional", "reject all", "reject", "necessary only", "decline")
                )
            ]
            if reject_candidates:
                target_name = reject_candidates[0]["text"]
            elif banner_id == "customizeBanner":
                # Inside customize banner: save settings (optional analytics unchecked)
                target_name = "Save settings"
            else:
                target_name = "Reject optional cookies"

        return {
            "detected": True,
            "popup_type": TYPE_CONSENT_BANNER,
            "is_blocking": True,
            "banner_id": banner_id,
            "preferred_button": target_name,
            "safe_to_dismiss": True,
            "reason": f"Consent banner detected ({banner_id}). Planned privacy-preserving action: '{target_name}'",
            "needs_hitl": False
        }

    def _classify_blocking_modal(self, modal_loc: Any, task_text: str) -> Dict[str, Any]:
        """Build classification and safe dismissal plan for blocking modal."""
        modal_id = "blocking_modal"
        text_content = ""
        try:
            modal_id = modal_loc.get_attribute("id") or "blocking_modal"
            text_content = modal_loc.inner_text() or ""
        except Exception:
            pass

        # Check for consequential/destructive dialog needing user confirmation
        consequential_phrases = (
            "delete your account", "permanently delete", "confirm payment",
            "are you sure you want to delete", "transfer funds"
        )
        if any(p in text_content.lower() for p in consequential_phrases):
            return {
                "detected": True,
                "popup_type": TYPE_BLOCKING_MODAL,
                "is_blocking": True,
                "modal_id": modal_id,
                "safe_to_dismiss": False,
                "needs_hitl": True,
                "question": f"A consequential dialog requires your authorization: '{text_content[:200]}'. How should the agent proceed?",
                "reason": "Consequential dialog detected; routing to human-in-the-loop"
            }

        # Safe dismiss targets: Cancel and continue, Close, Cancel, Dismiss, ×
        preferred_button = "Cancel and continue"
        try:
            btn_locs = modal_loc.locator("button")
            for i in range(btn_locs.count()):
                b = btn_locs.nth(i)
                txt = (b.inner_text() or "").strip()
                aria = (b.get_attribute("aria-label") or "").strip()
                combined = f"{txt} {aria}".lower()
                if "cancel and continue" in combined:
                    preferred_button = "Cancel and continue"
                    break
                elif "close" in combined or txt == "×":
                    preferred_button = txt or aria or "Close"
                elif "cancel" in combined:
                    preferred_button = "Cancel"
        except Exception:
            pass

        return {
            "detected": True,
            "popup_type": TYPE_BLOCKING_MODAL,
            "is_blocking": True,
            "modal_id": modal_id,
            "preferred_button": preferred_button,
            "safe_to_dismiss": True,
            "reason": f"Blocking modal detected ({modal_id}). Safe dismissal button: '{preferred_button}'",
            "needs_hitl": False
        }

    def _classify_promotion(self, promo_loc: Any, task_text: str) -> Dict[str, Any]:
        """Classify non-blocking promotion banner."""
        task_lower = task_text.lower()
        # Only dismiss if task explicitly requested dismissing promotions
        must_dismiss = "close promo" in task_lower or "dismiss promo" in task_lower or "close promotion" in task_lower
        return {
            "detected": True,
            "popup_type": TYPE_NON_BLOCKING_PROMO,
            "is_blocking": False,
            "preferred_button": "Close promotion" if must_dismiss else None,
            "safe_to_dismiss": True,
            "should_dismiss": must_dismiss,
            "reason": "Non-blocking promotion detected; does not obstruct workflow. Preserving without dismissal.",
            "needs_hitl": False
        }

    def _detect_via_dom(
        self,
        dom_elements: List[Dict[str, Any]],
        visible_text: str,
        task_text: str
    ) -> Dict[str, Any]:
        """Fallback detection using extracted DOM elements list."""
        for el in dom_elements:
            tag = str(el.get("tag", "")).lower()
            role = str(el.get("role", "")).lower()
            text = str(el.get("text", "")).lower()
            aria = str(el.get("ariaLabel", "")).lower()
            ident = f"{text} {aria}".lower()

            # Check Cookie banner
            if "cookie" in ident or "consent" in ident or el.get("id") == "cookieBanner":
                target = "Reject optional cookies"
                if "accept all" in task_text.lower():
                    target = "Accept all"
                return {
                    "detected": True,
                    "popup_type": TYPE_CONSENT_BANNER,
                    "is_blocking": True,
                    "preferred_button": target,
                    "safe_to_dismiss": True,
                    "reason": "Consent banner detected via DOM elements",
                    "needs_hitl": False
                }

            # Check Blocking modal
            if role in ("dialog", "alertdialog") or el.get("id") == "blockingModal" or "session notice" in text:
                consequential = ("delete your account", "permanently delete", "confirm payment", "are you sure you want to delete", "transfer funds")
                combined = f"{visible_text} {text}".lower()
                if any(p in combined for p in consequential):
                    return {
                        "detected": True,
                        "popup_type": TYPE_BLOCKING_MODAL,
                        "is_blocking": True,
                        "safe_to_dismiss": False,
                        "needs_hitl": True,
                        "question": "A consequential dialog requires your authorization. How should the agent proceed?",
                        "reason": "Consequential dialog detected via DOM requiring authorization; routing to human-in-the-loop"
                    }
                return {
                    "detected": True,
                    "popup_type": TYPE_BLOCKING_MODAL,
                    "is_blocking": True,
                    "preferred_button": "Cancel and continue",
                    "safe_to_dismiss": True,
                    "reason": "Blocking modal detected via DOM elements",
                    "needs_hitl": False
                }

            # Check promotion
            if role == "complementary" or el.get("id") == "promo" or "demo promotion" in text:
                must_dismiss = any(k in task_text.lower() for k in ("close promo", "dismiss promo", "close promotion"))
                return {
                    "detected": True,
                    "popup_type": TYPE_NON_BLOCKING_PROMO,
                    "is_blocking": False,
                    "preferred_button": "Close promotion" if must_dismiss else None,
                    "should_dismiss": must_dismiss,
                    "reason": "Non-blocking promotion detected via DOM elements",
                    "needs_hitl": False
                }

        return {"detected": False, "popup_type": TYPE_NONE, "is_blocking": False}

    def dismiss(
        self,
        page: Any,
        popup_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute dismissal on live page and verify outcome.

        Returns structured action result.
        """
        ptype = popup_info.get("popup_type")
        btn_target = popup_info.get("preferred_button")
        banner_id = popup_info.get("banner_id", "")

        if not popup_info.get("detected"):
            return {"success": True, "action_taken": "none", "reason": "No popup detected"}

        if ptype == TYPE_NON_BLOCKING_PROMO and not popup_info.get("should_dismiss"):
            return {
                "success": True,
                "action_taken": "skipped",
                "popup_type": ptype,
                "reason": "Non-blocking promotion skipped without dismissal"
            }

        if popup_info.get("needs_hitl"):
            return {
                "success": False,
                "action_taken": "ask_user",
                "popup_type": ptype,
                "question": popup_info.get("question"),
                "reason": "Consequential popup requires user confirmation"
            }

        # Check repeated dismissal attempt budget
        key = f"{ptype}_{banner_id or btn_target}"
        attempts = self.dismissal_attempts.get(key, 0)
        if attempts >= self.max_attempts:
            return {
                "success": False,
                "action_taken": "exhausted",
                "popup_type": ptype,
                "reason": f"Dismissal attempts exhausted ({self.max_attempts}) for {key}"
            }
        self.dismissal_attempts[key] = attempts + 1

        # Locate and click target button
        try:
            clicked = False
            # Try by text inside dialog container
            if banner_id:
                container = page.locator(f"#{banner_id}")
                if container.count() > 0:
                    btn = container.locator(f"button:has-text('{btn_target}')")
                    if btn.count() > 0 and btn.first.is_visible():
                        btn.first.click()
                        clicked = True

            if not clicked:
                # Try global button locator with exact text or aria-label
                btn = page.locator(f"button:has-text('{btn_target}')")
                if btn.count() > 0 and btn.first.is_visible():
                    btn.first.click()
                    clicked = True
                else:
                    btn_aria = page.locator(f"button[aria-label*='{btn_target}' i]")
                    if btn_aria.count() > 0 and btn_aria.first.is_visible():
                        btn_aria.first.click()
                        clicked = True

            if not clicked:
                # Fallback: close button (x)
                close_btn = page.locator(".close, button[aria-label*='Close' i], button:has-text('×')")
                for i in range(close_btn.count()):
                    cb = close_btn.nth(i)
                    if cb.is_visible():
                        cb.click()
                        clicked = True
                        break

            if not clicked:
                return {
                    "success": False,
                    "action_taken": "not_found",
                    "popup_type": ptype,
                    "reason": f"Could not find dismissal button '{btn_target}'"
                }

            # Brief settle
            try:
                page.wait_for_timeout(300)
            except Exception:
                pass

            # Verification: is container no longer visible?
            verified = True
            if banner_id:
                loc = page.locator(f"#{banner_id}")
                if loc.count() > 0 and loc.first.is_visible() and not self._has_hidden_class(loc.first):
                    verified = False

            return {
                "success": verified,
                "action_taken": "dismissed",
                "popup_type": ptype,
                "button_clicked": btn_target,
                "verified": verified,
                "reason": f"Dismissed {ptype} via '{btn_target}' (verified={verified})"
            }
        except Exception as e:
            logger.error(f"Popup dismissal failed: {e}")
            return {
                "success": False,
                "action_taken": "error",
                "popup_type": ptype,
                "error": str(e),
                "reason": f"Dismissal execution threw error: {e}"
            }
