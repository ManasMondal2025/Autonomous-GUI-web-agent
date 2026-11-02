"""Versioned Skill Registry for agent skills."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from .models import SkillDefinition
from .builtins import BUILTIN_SKILLS

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Versioned registry for registering, discovering, and selecting reusable agent skills."""

    def __init__(self, load_builtins: bool = True):
        self._skills: Dict[str, Dict[int, SkillDefinition]] = {}
        if load_builtins:
            for s in BUILTIN_SKILLS:
                self.register(s)

    def register(self, skill: SkillDefinition) -> None:
        """Register a new version of a skill."""
        if skill.name not in self._skills:
            self._skills[skill.name] = {}
        self._skills[skill.name][skill.version] = skill
        logger.debug(f"Registered skill {skill.name!r} v{skill.version}")

    def register_all(self, skills: List[SkillDefinition]) -> None:
        """Register a list of skills."""
        for s in skills:
            self.register(s)

    def get_skill(self, name: str, version: Optional[int] = None) -> Optional[SkillDefinition]:
        """Retrieve a registered skill by name and optional version (defaults to latest)."""
        versions = self._skills.get(name)
        if not versions:
            return None
        if version is not None:
            return versions.get(version)
        latest_v = max(versions.keys())
        return versions[latest_v]

    def list_skills(self) -> List[SkillDefinition]:
        """List the latest version of all registered skills."""
        out = []
        for name in sorted(self._skills.keys()):
            latest_v = max(self._skills[name].keys())
            out.append(self._skills[name][latest_v])
        return out

    def find_applicable_skills(
        self,
        task_text: str,
        observation: Dict[str, Any],
        page: Any = None,
        subgoal_intent: Optional[str] = None,
    ) -> List[SkillDefinition]:
        """Discover applicable skills by evaluating preconditions against actual interface state.

        Critical requirement: does NOT choose skills based merely on keyword matching.
        Evaluates whether the interface actually contains the relevant component.
        """
        applicable: List[SkillDefinition] = []
        task_lower = f"{task_text} {subgoal_intent or ''}".lower()
        dom = observation.get("dom_elements", []) or []
        visible_text = str(observation.get("visible_text", "") or "").lower()

        for skill in self.list_skills():
            if not self._matches_task_scope(skill, task_lower):
                continue

            # Verify actual interface preconditions
            preconditions_met = self._evaluate_preconditions(skill, observation, page)
            if preconditions_met:
                applicable.append(skill)
            else:
                logger.debug(
                    f"Skill {skill.name!r} matched task intent but failed live interface preconditions."
                )

        return applicable

    def _matches_task_scope(self, skill: SkillDefinition, task_lower: str) -> bool:
        if skill.name == "date_picker":
            return any(w in task_lower for w in ("date", "calendar", "appointment date", "month", "year", "2026", "day"))
        elif skill.name == "dropdown_selection":
            return any(w in task_lower for w in ("dropdown", "select option", "choose from list", "picker", "select"))
        elif skill.name == "asynchronous_waiting":
            return any(w in task_lower for w in ("wait", "slow", "ajax", "async", "single submit", "loading"))
        elif skill.name == "pagination":
            return any(w in task_lower for w in ("page", "next", "previous", "pagination", "table rows"))
        elif skill.name == "form_filling":
            return any(w in task_lower for w in ("fill", "form", "profile", "name", "email", "enter details"))
        return False

    def _evaluate_preconditions(
        self,
        skill: SkillDefinition,
        observation: Dict[str, Any],
        page: Any = None,
    ) -> bool:
        """Inspect live DOM / observation to confirm preconditions are satisfied."""
        dom = observation.get("dom_elements", []) or []
        visible_text = str(observation.get("visible_text", "") or "").lower()

        if skill.name == "date_picker":
            # Live DOM inspection: Must have date input or calendar element
            if page is not None:
                try:
                    cal = page.locator("#calendarRoot, input[type='date'], #appointmentDate, #combinedDate, .calendar, button:has-text('Open date picker')")
                    return cal.count() > 0 and cal.first.is_visible()
                except Exception:
                    return False
            # Offline observation check (only interactable DOM elements, never mere visible text)
            return any(
                "calendar" in str(el.get("id", "")).lower()
                or "date" in str(el.get("id", "")).lower()
                or (el.get("tag") == "input" and el.get("type") == "date")
                for el in dom
            )

        elif skill.name == "dropdown_selection":
            if page is not None:
                try:
                    sels = page.locator("select, [role='combobox'], [role='listbox']")
                    return sels.count() > 0 and sels.first.is_visible()
                except Exception:
                    return False
            return any(el.get("tag") == "select" or el.get("role") in ("combobox", "listbox") for el in dom)

        elif skill.name == "asynchronous_waiting":
            if page is not None:
                try:
                    btn = page.locator("button:has-text('Submit once'), button:has-text('Search'), #slowSubmitButton, [aria-live]")
                    return btn.count() > 0 and btn.first.is_visible()
                except Exception:
                    return False
            return any(el.get("role") == "status" or "status" in str(el.get("id", "")) for el in dom)

        elif skill.name == "pagination":
            if page is not None:
                try:
                    pag = page.locator(".pagination, button:has-text('Next'), button:has-text('Previous'), [aria-label*='page']")
                    return pag.count() > 0 and pag.first.is_visible()
                except Exception:
                    return False
            return any("next" in str(el.get("text", "")).lower() for el in dom)

        elif skill.name == "form_filling":
            if page is not None:
                try:
                    inputs = page.locator("input:not([type='hidden']), textarea")
                    return inputs.count() >= 2
                except Exception:
                    return False
            input_count = sum(1 for el in dom if el.get("tag") in ("input", "textarea"))
            return input_count >= 2

        return False

