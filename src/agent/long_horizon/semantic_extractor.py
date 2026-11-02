"""Semantic Fact Extractor — LLM-powered, domain-agnostic.

Replaces the hardcoded keyword regex approach in _extract_facts with a
local LLM (Ollama qwen2.5:7b or llama3) that:

1. Understands the INTENT of any natural-language prompt regardless of:
   - Poor grammar ("bro book train delhi tmrw")
   - Typos ("laptp", "blck")
   - Missing punctuation or structure
   - Domain-specific jargon

2. Extracts ALL facts mentioned in the prompt:
   - Names, dates, locations, amounts, quantities
   - Credentials (email, phone, card, CVV, PIN, password)
   - Identifiers (booking IDs, coupon codes, promo codes)
   - Preferences (seat class, meal preference, etc.)

3. Identifies what is MISSING and REQUIRED for the task to succeed:
   - Understands that "book a flight" without a date is incomplete
   - Understands that "pay online" without card details is incomplete
   - Generates a specific, friendly HITL question for each gap

4. Gracefully falls back to regex extraction if Ollama is unavailable.

Design:
  - Single Ollama HTTP call (no external API, no network required)
  - Returns structured JSON that merges into the existing TaskSpec pipeline
  - Timeout-safe: falls back to regex on any failure in < 30s
  - Model preference: qwen2.5:7b (better JSON) > llama3:latest (fallback)
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Ollama client (zero external dependencies — stdlib only)
# ──────────────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
PREFERRED_MODELS = ["qwen2.5:7b", "llama3:latest"]
_TIMEOUT_SECONDS = 60  # generous for cold model load (first call loads model into RAM)


def _ollama_generate(model: str, prompt: str, timeout: int = _TIMEOUT_SECONDS) -> str:
    """Call Ollama /api/generate and return the response text.

    Raises OSError / urllib.error.URLError on connection failure.
    """
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,   # deterministic
            "top_p": 1.0,
            "num_predict": 1024,
        }
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    return body.get("response", "")


def _available_model() -> Optional[str]:
    """Return the first available model from PREFERRED_MODELS, or None."""
    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        names = {m["name"] for m in data.get("models", [])}
        for model in PREFERRED_MODELS:
            if model in names:
                return model
    except Exception as exc:
        logger.debug(f"Ollama not reachable: {exc}")
    return None


def _extract_json_from_response(text: str) -> Dict[str, Any]:
    """Extract JSON object from LLM response (handles markdown code fences)."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # Find the first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found in response: {text[:200]!r}")
    raw = m.group(0)
    # Common LLM JSON quirks: trailing commas, single-quoted strings
    raw = re.sub(r",(\s*[}\]])", r"\1", raw)
    raw = raw.replace("\\'", "'")
    return json.loads(raw)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt template
# ──────────────────────────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """You are a precise information extractor for a web automation agent.

A user gave this task instruction (may be poorly written, have typos, missing grammar):
\"\"\"
{task}
\"\"\"

Your job:
1. Understand the INTENT even if the text is messy or grammatically wrong.
2. Extract ALL concrete values the user provided (names, dates, locations, emails, phones, amounts, quantities, codes, credentials, preferences, IDs, etc.).
3. Determine what information is REQUIRED to complete this task that was NOT provided.
4. For each missing required piece, write a clear, friendly question to ask the user.

Rules:
- Extract ONLY values explicitly stated. Do NOT invent or assume values.
- For "missing_fields", include ONLY things that are REQUIRED for this specific task (not optional).
- A task like "book a flight" REQUIRES: departure city, destination city, date, passenger name. Missing any of these means it cannot proceed.
- A task like "buy a laptop" REQUIRES: which laptop (if ambiguous). Does NOT require a shipping address yet.
- If the task is completely unclear (only 1-2 words with no actionable meaning), set "task_unclear": true.
- "intent" should be a single clear sentence describing what the user wants to do.

Reply with ONLY this JSON (no other text):
{{
  "intent": "one sentence describing what the user wants",
  "task_unclear": false,
  "extracted_facts": {{
    "name": "full name if mentioned",
    "email": "email if mentioned",
    "phone": "phone number if mentioned",
    "date": "date or date range if mentioned",
    "origin": "departure city/station if mentioned",
    "destination": "destination city/station if mentioned",
    "amount": "monetary amount if mentioned",
    "quantity": "number of items/tickets/passengers if mentioned",
    "coupon_code": "promo or coupon code if mentioned",
    "card_number": "payment card number if mentioned",
    "card_expiry": "card expiry if mentioned",
    "cvv": "CVV if mentioned",
    "password": "password if mentioned",
    "address": "full address if mentioned",
    "city": "city if mentioned",
    "state": "state/province if mentioned",
    "country": "country if mentioned",
    "pin_code": "postal/pin code if mentioned",
    "seat_class": "class like economy/business/first if mentioned",
    "product_name": "product name if mentioned",
    "size": "size if mentioned",
    "color": "color if mentioned",
    "notes": "any other specific values mentioned"
  }},
  "missing_fields": [
    {{
      "field": "field name",
      "reason": "why it is required for this specific task",
      "question": "the exact question to ask the user"
    }}
  ],
  "clarification_needed": "one question if the overall task intent is unclear, else empty string"
}}

IMPORTANT: Only include fields in "extracted_facts" if the user actually provided them. Set the value to null or omit the key if not provided. Do NOT add placeholder values."""


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class SemanticExtractionResult:
    """Result from the semantic extractor."""

    def __init__(
        self,
        intent: str,
        extracted_facts: Dict[str, Any],
        missing_fields: List[Dict[str, str]],
        clarification_needed: str,
        task_unclear: bool,
        model_used: Optional[str],
        fell_back_to_regex: bool = False,
    ):
        self.intent = intent
        self.extracted_facts = {k: v for k, v in extracted_facts.items()
                                 if v is not None and v != ""}
        self.missing_fields = missing_fields  # list of {field, reason, question}
        self.clarification_needed = clarification_needed
        self.task_unclear = task_unclear
        self.model_used = model_used
        self.fell_back_to_regex = fell_back_to_regex

    @property
    def first_missing_question(self) -> Optional[str]:
        """The clarification question to ask the user first (if any)."""
        if self.clarification_needed:
            return self.clarification_needed
        if self.missing_fields:
            return self.missing_fields[0]["question"]
        return None

    @property
    def has_gaps(self) -> bool:
        return bool(self.clarification_needed or self.missing_fields or self.task_unclear)

    def __repr__(self) -> str:
        return (f"SemanticExtractionResult("
                f"intent={self.intent!r}, "
                f"facts={list(self.extracted_facts.keys())}, "
                f"missing={[m['field'] for m in self.missing_fields]}, "
                f"model={self.model_used!r})")


class SemanticExtractor:
    """LLM-powered semantic fact extraction for any domain.

    Falls back to an empty result (allowing regex _extract_facts to run)
    if Ollama is unavailable or times out.

    Usage:
        extractor = SemanticExtractor()
        result = extractor.extract("bro book train delhi mumbai tmrw rahul")
        if result.has_gaps:
            print(result.first_missing_question)
        print(result.extracted_facts)
    """

    def __init__(self, model: Optional[str] = None, timeout: int = _TIMEOUT_SECONDS):
        self._preferred_model = model
        self._timeout = timeout
        self._cached_model: Optional[str] = None  # resolved on first call
        self._fail_count = 0          # consecutive failures
        self._max_fails = 2           # latch unavailable after this many

    def _resolve_model(self) -> Optional[str]:
        if self._cached_model:
            return self._cached_model
        if self._preferred_model:
            self._cached_model = self._preferred_model
            return self._cached_model
        model = _available_model()
        if model:
            self._cached_model = model
            logger.info(f"SemanticExtractor: using Ollama model {model!r}")
        return model

    def extract(self, task_text: str) -> SemanticExtractionResult:
        """Semantically extract facts and detect missing fields.

        Returns a SemanticExtractionResult. If Ollama is unavailable or
        fails repeatedly, returns an empty result (fell_back_to_regex=True)
        so the caller can fall back to regex _extract_facts.
        """
        if self._fail_count >= self._max_fails:
            return self._empty_result(task_text, fell_back=True)

        model = self._resolve_model()
        if model is None:
            logger.warning("SemanticExtractor: no Ollama model available; "
                           "falling back to regex extraction")
            self._fail_count += 1
            return self._empty_result(task_text, fell_back=True)

        prompt = _EXTRACTION_PROMPT.format(task=task_text.strip())

        try:
            raw = _ollama_generate(model, prompt, timeout=self._timeout)
            self._fail_count = 0  # reset on success
        except Exception as exc:
            self._fail_count += 1
            remaining = self._max_fails - self._fail_count
            logger.warning(
                f"SemanticExtractor: Ollama call failed ({exc}); "
                f"failure {self._fail_count}/{self._max_fails} "
                f"({'latching disabled' if remaining <= 0 else f'{remaining} retries left'}); "
                f"falling back to regex extraction")
            return self._empty_result(task_text, fell_back=True)

        try:
            data = _extract_json_from_response(raw)
        except Exception as exc:
            logger.warning(f"SemanticExtractor: JSON parse failed ({exc}); "
                           f"raw={raw[:300]!r}; falling back to regex")
            return self._empty_result(task_text, fell_back=True)

        # Parse the structured response
        extracted = data.get("extracted_facts", {}) or {}
        missing = data.get("missing_fields", []) or []
        clarif = data.get("clarification_needed", "") or ""
        unclear = bool(data.get("task_unclear", False))
        intent = data.get("intent", "") or ""

        # Validate missing_fields shape
        clean_missing = []
        for item in missing:
            if isinstance(item, dict) and "question" in item:
                clean_missing.append({
                    "field": str(item.get("field", "unknown")),
                    "reason": str(item.get("reason", "")),
                    "question": str(item["question"]),
                })

        return SemanticExtractionResult(
            intent=intent,
            extracted_facts=extracted,
            missing_fields=clean_missing,
            clarification_needed=clarif,
            task_unclear=unclear,
            model_used=model,
            fell_back_to_regex=False,
        )

    def _empty_result(self, task_text: str,
                      fell_back: bool = False) -> SemanticExtractionResult:
        return SemanticExtractionResult(
            intent="",
            extracted_facts={},
            missing_fields=[],
            clarification_needed="",
            task_unclear=False,
            model_used=None,
            fell_back_to_regex=fell_back,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Merge helper: combine LLM facts with regex facts
# ──────────────────────────────────────────────────────────────────────────────

def merge_facts(
    semantic: Dict[str, Any],
    regex: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge semantic LLM facts with deterministic regex facts.

    Regex facts win for high-confidence structured data (emails, phones,
    amounts, dates) because they match the exact string verbatim.
    Semantic facts fill in everything else (names, cities, intents, etc.)
    that regex cannot capture.
    """
    # Start with semantic (broad coverage)
    merged = {k: v for k, v in semantic.items() if v is not None and v != ""}

    # Overlay regex for high-precision fields (they are more reliable)
    _REGEX_WINS = {"email", "emails", "phone", "amount", "date", "quantity",
                   "quoted_values", "codes"}
    for key in _REGEX_WINS:
        if key in regex and regex[key]:
            merged[key] = regex[key]

    return merged
