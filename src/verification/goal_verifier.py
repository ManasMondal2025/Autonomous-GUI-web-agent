"""GoalVerifier: has the user's goal already been achieved?

General-purpose (no task-specific hard-coding). Two modes:

- heuristic (default, free): grounds lexical entities from the goal
  (quoted strings, capitalized phrases, content keywords) in the page's
  visible text / input values / URL, and requires evidence of progress
  (URL change toward the goal, input change, or DOM change) so pages that
  merely mention the entities don't trigger premature termination.
- vlm: asks an attached VLM judge with the current screenshot.
- both: achieved if either mode says so.

Known limitation: the heuristic is lexical. Goals whose success is not
visible as text/URL change (e.g. "book the cheapest flight", file
downloads) need mode="vlm".
"""

import logging
import re
from typing import Dict, Any, Optional, List
from PIL import Image
import io

logger = logging.getLogger(__name__)

STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "with", "into", "from", "that",
    "this", "then", "than", "click", "type", "open", "close", "press",
    "enter", "select", "choose", "find", "search", "look", "add", "go",
    "navigate", "visit", "view", "show", "confirm", "verify", "check",
    "complete", "finish", "submit", "fill", "page", "screen",
    "site", "website", "tab", "button", "link", "field", "box", "form",
    "please", "using", "use", "your", "you", "its", "item", "task",
}

PAGE_WORDS = {"page", "screen", "site", "website", "view", "tab"}

_QUOTED_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')
_CAP_PHRASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def extract_entities(goal: str) -> List[str]:
    """Quoted spans + capitalized phrases (minus leading word & stopwords)."""
    entities: List[str] = []
    for m in _QUOTED_RE.finditer(goal):
        span = m.group(1) or m.group(2)
        if span and span.strip():
            entities.append(span.strip())
    first_word = (goal.strip().split() or [""])[0].strip(",.':\"")
    for m in _CAP_PHRASE_RE.finditer(goal):
        phrase = m.group(1)
        words = phrase.split()
        # Strip a leading action verb glued to the phrase ("Add Laptop Pro").
        if words and words[0].lower() in STOPWORDS:
            words = words[1:]
        if not words:
            continue
        phrase = " ".join(words)
        if phrase == first_word:
            continue
        if all(w.lower() in STOPWORDS for w in words):
            continue
        if phrase not in entities:
            entities.append(phrase)
    return entities


def extract_keywords(goal: str) -> List[str]:
    """Lowercase content words (len>2, not stopwords)."""
    words = [w.lower() for w in _WORD_RE.findall(goal)]
    seen, out = set(), []
    for w in words:
        if len(w) > 2 and w not in STOPWORDS and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def page_text(observation: Dict[str, Any]) -> str:
    """Searchable text: full visible text + DOM fields + input values + URL."""
    parts = [observation.get("url", ""), observation.get("visible_text", "")]
    for el in observation.get("dom_elements", []) or []:
        for key in ("text", "ariaLabel", "placeholder", "title", "value", "name"):
            v = el.get(key, "")
            if v:
                parts.append(str(v))
    return "\n".join(p for p in parts if p)


def entity_hit(entity: str, text: str) -> bool:
    """Exact phrase match, falling back to all-words-present (whitespace-proof)."""
    e = entity.lower()
    if e in text:
        return True
    words = [w for w in _WORD_RE.findall(e) if len(w) > 1]
    return bool(words) and all(w in text for w in words)


def input_values(observation: Dict[str, Any]) -> str:
    vals = []
    for el in observation.get("dom_elements", []) or []:
        if el.get("tag") in ("input", "textarea", "select") and el.get("value"):
            vals.append(str(el["value"]))
    return "\n".join(vals)


class GoalVerifier:
    def __init__(self, mode: str = "heuristic", vlm: Optional[Any] = None,
                 keyword_threshold: float = 0.5):
        if mode not in ("heuristic", "vlm", "both"):
            raise ValueError(f"Unknown GoalVerifier mode: {mode}")
        self.mode = mode
        self.vlm = vlm
        self.keyword_threshold = keyword_threshold

    def check(
        self,
        goal: str,
        observation: Dict[str, Any],
        start_url: str = "",
        prev_observation: Optional[Dict[str, Any]] = None,
        verification: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return {"success", "achieved", "done", "reason", "signals"}."""
        result: Dict[str, Any] = {
            "success": True, "achieved": False, "done": False,
            "reason": "", "signals": {},
        }
        heuristic_hit = False
        if self.mode in ("heuristic", "both"):
            heuristic_hit = self._heuristic_check(
                goal, observation, start_url, prev_observation,
                verification, result["signals"])
        vlm_hit = False
        if self.mode in ("vlm", "both") and self.vlm is not None:
            vlm_hit = self._vlm_check(goal, observation, result["signals"])

        achieved = heuristic_hit or vlm_hit
        result["achieved"] = achieved
        result["done"] = achieved
        if achieved:
            result["reason"] = "Goal already achieved"
        else:
            result["reason"] = result["signals"].get("heuristic_reason",
                                                     "Goal not yet achieved")
        return result

    # -- heuristic ------------------------------------------------------
    def _heuristic_check(self, goal, obs, start_url, prev_obs, verification,
                         signals) -> bool:
        entities = extract_entities(goal)
        keywords = extract_keywords(goal)
        text = page_text(obs).lower()
        url = (obs.get("url", "") or "").lower()

        entity_hits = [e for e in entities if entity_hit(e, text)]
        signals["entities"] = entities
        signals["entity_hits"] = entity_hits
        kw_hits = [k for k in keywords if k in text]
        signals["keyword_coverage"] = (
            round(len(kw_hits) / len(keywords), 3) if keywords else 1.0)

        entities_ok = len(entity_hits) == len(entities)
        keywords_ok = signals["keyword_coverage"] >= self.keyword_threshold
        present = entities_ok and keywords_ok
        signals["present"] = present
        if not present:
            signals["heuristic_reason"] = (
                f"missing entities {[e for e in entities if e not in entity_hits]}, "
                f"keyword coverage {signals['keyword_coverage']}")
            return False

        url_changed = bool(start_url) and obs.get("url", "") != start_url
        signals["url_changed"] = url_changed
        goal_mentions_page = any(re.search(rf"\b{re.escape(pw)}\b", goal.lower()) for pw in PAGE_WORDS)
        slug_ok = True
        if goal_mentions_page and not url.startswith("file://"):
            url_path = url.split("?")[0]
            content_terms = [k for k in keywords if k not in PAGE_WORDS] + \
                [w.lower() for e in entities for w in _WORD_RE.findall(e)]
            slug_ok = any(t in url_path for t in content_terms if len(t) > 2)
            signals["url_slug_ok"] = slug_ok

        inputs_now = input_values(obs).lower()
        goal_terms = keywords + [e.lower() for e in entities]
        input_changed = any(t in inputs_now for t in goal_terms if t)
        signals["input_match"] = bool(input_changed)

        dom_changed = False
        if prev_obs is not None:
            before = {(e.get("tag"), (e.get("text") or "")[:80], e.get("value") or "")
                      for e in prev_obs.get("dom_elements", []) or []}
            after = {(e.get("tag"), (e.get("text") or "")[:80], e.get("value") or "")
                     for e in obs.get("dom_elements", []) or []}
            prev_txt = page_text(prev_obs).strip().lower()
            curr_txt = text.strip().lower()
            dom_changed = (before != after) or (prev_txt != curr_txt)
        elif verification is not None:
            dom_changed = bool(verification.get("checks", {}).get("dom_changed", False))
        signals["dom_changed"] = dom_changed

        if not slug_ok:
            signals["heuristic_reason"] = \
                "entities present but URL does not reflect the goal page"
            return False
        if url_changed or input_changed or dom_changed:
            return True
        # Present before AND after with zero change: still count as achieved
        # only if the page already reflects the goal at check time and the
        # action just executed succeeded (e.g. confirmation pages).
        if verification and verification.get("success"):
            return True
        signals["heuristic_reason"] = "entities present but no state progress detected"
        return False

    # -- vlm judge --------------------------------------------------------
    def _vlm_check(self, goal, obs, signals) -> bool:
        import json as _json
        try:
            img = Image.open(io.BytesIO(obs["screenshot_bytes"]))
        except Exception as e:
            signals["vlm_error"] = f"no screenshot: {e}"
            return False
        prompt = (
            f"Task goal: {goal}\n"
            "Look at this screenshot of the current browser state. "
            "Has the goal ALREADY been fully achieved? "
            'Reply ONLY with JSON: {"success": true/false, "reason": "..."}')
        try:
            resp = self.vlm.generate(img, prompt)
            m = re.search(r"\{.*\}", resp, re.DOTALL)
            data = _json.loads(m.group(0) if m else resp)
            signals["vlm_success"] = bool(data.get("success"))
            signals["vlm_reason"] = str(data.get("reason", ""))[:200]
            return bool(data.get("success"))
        except Exception as e:
            signals["vlm_error"] = str(e)[:200]
            return False


def classify_termination(step_goal_checks: List[Dict[str, Any]],
                         run_succeeded: bool) -> str:
    """Post-hoc label from per-step goal_check records.

    achieved_and_terminated: first achieved step is the final step (and success)
    achieved_but_continued: achieved at an earlier step, agent kept going
    not_achieved: never achieved
    """
    achieved_steps = [i + 1 for i, gc in enumerate(step_goal_checks)
                      if isinstance(gc, dict) and gc.get("achieved")]
    if not achieved_steps:
        return "not_achieved"
    if achieved_steps[0] == len(step_goal_checks) and run_succeeded:
        return "achieved_and_terminated"
    return "achieved_but_continued"
