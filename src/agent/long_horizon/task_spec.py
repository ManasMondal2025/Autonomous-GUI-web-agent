"""Task Compiler: natural-language request → validated TaskSpec.

Phase 1 of the upgrade. A TaskSpec is the authority on what was requested,
what constraints must be honoured, and what observable predicates confirm
completion. It is compiled ONCE per task and stored in TaskMemory so every
downstream component reads from the same source of truth.

Design rules:
  - Uses Pydantic (already in requirements.txt) for validation.
  - Never invents personal information or inferred preferences.
  - DOMAIN-AGNOSTIC: the compiler contains NO hardcoded domain defaults
    (no "quantity=1", no "delivery=Standard"). These assumptions are
    wrong on flight bookings, bank forms, government portals, etc.
  - SEMANTIC-FIRST: fact extraction uses a local LLM (qwen2.5:7b via
    Ollama) to understand ANY prompt regardless of grammar, typos, or
    domain. Regex extraction is used as a high-precision fallback for
    structured values (emails, phones, amounts, dates).
  - HITL-FIRST: when the LLM determines required information is missing
    (e.g. "book a flight" with no date), the compiler asks the user
    via ask_handler. It never guesses or silently skips.
  - If Ollama is unavailable, the system falls back gracefully to
    keyword-based detection without crashing.
  - Compilation failure is a typed result (never a silent pass).
  - Schema version field enables safe deserialization across upgrades.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Semantic extractor (optional — falls back to regex if Ollama unavailable)
try:
    from .semantic_extractor import SemanticExtractor, merge_facts
    _SEMANTIC_AVAILABLE = True
except ImportError:
    _SEMANTIC_AVAILABLE = False
    SemanticExtractor = None  # type: ignore[assignment,misc]
    def merge_facts(semantic, regex):  # type: ignore[misc]
        return regex

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"

# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────

class PermissionLevel(str, Enum):
    READ_ONLY = "read_only"          # navigate, observe, extract
    FORM_FILL = "form_fill"          # fill forms (no submission)
    SUBMIT    = "submit"             # submit forms, save changes
    PURCHASE  = "purchase"           # checkout, pay, confirm orders
    ADMIN     = "admin"              # delete, reset, admin actions


class MissingItemSeverity(str, Enum):
    MANDATORY = "mandatory"          # execution cannot proceed safely
    OPTIONAL  = "optional"           # can proceed with a disclosed default


class FactSource(str, Enum):
    USER_PROVIDED  = "user_provided"   # verbatim in the user request
    INFERRED       = "inferred"        # model-derived (must be disclosed)
    DEFAULT        = "default"         # allowlisted default (disclosed)
    STV_VERIFIED   = "stv_verified"    # verified by STV (post-execution)


# ──────────────────────────────────────────────────────────────────────────────
# Sub-models
# ──────────────────────────────────────────────────────────────────────────────

class CompletionPredicate(BaseModel):
    """One observable condition that must hold for the task to be complete."""
    predicate: str = Field(description="EQUALS|CONTAINS|VISIBLE|PAGE_IS|SELECTED|EXISTS|APPLIED")
    target: str = Field(default="", description="UI element or fact name")
    value: str = Field(default="", description="Expected value")
    required: bool = Field(default=True)

    @field_validator("predicate")
    @classmethod
    def check_predicate_vocab(cls, v: str) -> str:
        allowed = {"EQUALS", "CONTAINS", "NOT_CONTAINS", "VISIBLE",
                   "EXISTS", "PAGE_IS", "SELECTED", "APPLIED",
                   "GREATER_THAN", "LESS_THAN", "CHANGED"}
        upper = v.strip().upper()
        if upper not in allowed:
            raise ValueError(f"Unknown predicate {v!r}; allowed: {sorted(allowed)}")
        return upper


class MissingItem(BaseModel):
    """Information the compiler could not find but the task may need."""
    key: str
    description: str
    severity: MissingItemSeverity = MissingItemSeverity.OPTIONAL
    suggested_question: str = Field(
        default="", description="Ready-to-ask HITL question")


class DisclosedDefault(BaseModel):
    """A default value the compiler chose (never hidden from the user)."""
    key: str
    value: Any
    reason: str = Field(default="", description="Why this default was chosen")

    def __init__(self, **data: Any):
        if "rationale" in data and "reason" not in data:
            data["reason"] = data.pop("rationale")
        super().__init__(**data)


class ExplicitConstraint(BaseModel):
    """A constraint or preference stated explicitly by the user."""
    key: str
    value: Any
    source: FactSource = FactSource.USER_PROVIDED


# ──────────────────────────────────────────────────────────────────────────────
# Primary schema
# ──────────────────────────────────────────────────────────────────────────────

class TaskSpec(BaseModel):
    """Typed, validated task specification.

    This is the canonical view of what the user wants. All downstream
    planning and verification components read from here rather than from
    the raw natural-language string.
    """
    schema_version: str = Field(default=SCHEMA_VERSION)
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Core content
    original_request: str = Field(description="Verbatim user request")
    normalized_goal: str = Field(description="Compiler-normalised goal statement")

    # Constraints — only user-stated facts; inferred items are disclosed separately
    constraints: List[ExplicitConstraint] = Field(default_factory=list)
    preferences: List[ExplicitConstraint] = Field(default_factory=list)

    # Information provenance
    user_provided_facts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Values extracted verbatim from the user request")
    missing_information: List[MissingItem] = Field(default_factory=list)
    disclosed_defaults: List[DisclosedDefault] = Field(default_factory=list)

    # Completion / verification
    completion_criteria: List[CompletionPredicate] = Field(default_factory=list)

    # Safety
    max_permission: PermissionLevel = Field(default=PermissionLevel.SUBMIT)
    consequential_actions: List[str] = Field(
        default_factory=list,
        description="Actions that are externally irreversible (purchase, submit, etc.)")
    requires_confirmation: bool = Field(
        default=False,
        description="True when a consequential action requires explicit user confirmation")

    # Task-memory back-references (optional)
    task_memory_refs: List[str] = Field(default_factory=list)

    model_config = {"use_enum_values": True}

    @field_validator("normalized_goal")
    @classmethod
    def goal_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("normalized_goal must not be empty")
        return v.strip()

    @field_validator("original_request")
    @classmethod
    def request_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("original_request must not be empty")
        return v.strip()

    @model_validator(mode="after")
    def check_completion_criteria_testable(self) -> "TaskSpec":
        for crit in self.completion_criteria:
            # Only required predicates must be testable
            if crit.required and not crit.target and not crit.value:
                raise ValueError(
                    f"Completion predicate {crit.predicate!r} has neither "
                    f"target nor value — it is untestable")
        return self

    # ── helpers ──────────────────────────────────────────────────────────────

    def mandatory_missing(self) -> List[MissingItem]:
        return [m for m in self.missing_information
                if m.severity == MissingItemSeverity.MANDATORY]

    def is_executable(self) -> bool:
        """True when no mandatory information is still missing."""
        return not self.mandatory_missing()

    def to_json(self) -> Dict[str, Any]:
        return self.model_dump(mode="python")

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "TaskSpec":
        ver = data.get("schema_version", "unknown")
        if ver != SCHEMA_VERSION:
            logger.warning(
                f"TaskSpec schema version mismatch: stored={ver!r}, "
                f"current={SCHEMA_VERSION!r}; loading with best-effort.")
        return cls(**data)


# ──────────────────────────────────────────────────────────────────────────────
# Compiler result
# ──────────────────────────────────────────────────────────────────────────────

class CompilationResult:
    """Outcome of one compilation attempt."""
    __slots__ = ("spec", "error", "needs_clarification", "clarification_question")

    def __init__(
        self,
        spec: Optional[TaskSpec] = None,
        error: str = "",
        needs_clarification: bool = False,
        clarification_question: str = "",
    ) -> None:
        self.spec = spec
        self.error = error
        self.needs_clarification = needs_clarification
        self.clarification_question = clarification_question

    @property
    def ok(self) -> bool:
        return self.spec is not None and not self.error and not self.needs_clarification

    def __repr__(self) -> str:
        if self.ok:
            return f"CompilationResult(ok, task_id={self.spec.task_id})"  # type: ignore[union-attr]
        return (f"CompilationResult(error={self.error!r}, "
                f"clarification={self.needs_clarification})")


# ──────────────────────────────────────────────────────────────────────────────
# Extraction helpers (deterministic; no VLM calls)
# ──────────────────────────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d[\d\s\-()]{7,}\d)\b")
_AMOUNT_RE = re.compile(r"(?:[$£€₹]\s*[\d,]+(?:\.\d+)?|[\d,]+(?:\.\d+)?\s*(?:USD|EUR|GBP|INR))", re.I)
_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec"
_DATE_RE   = re.compile(rf"\b(?:\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}|\d{{4}}[/-]\d{{2}}[/-]\d{{2}}|(?:{_MONTHS})\s+\d{{1,2}},?\s+\d{{4}}|\d{{1,2}}\s+(?:{_MONTHS}),?\s+\d{{4}})\b", re.I)
_QUANTITY_RE = re.compile(r"\b(\d+)\s*(?:items?|units?|pcs?|pieces?|qty|x)\b", re.I)
_QUOTED_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')
_CODE_RE   = re.compile(r"\b([A-Z0-9]{4,}(?:[_-][A-Z0-9]+)*)\b")

_IRREVERSIBLE_VERBS = frozenset({
    "buy", "purchase", "checkout", "pay", "subscribe", "book",
    "order", "submit", "confirm", "delete", "cancel", "reset",
    "send", "transfer", "withdraw", "install",
})

_CONSEQUENTIAL_ACTIONS = frozenset({
    "checkout", "pay", "purchase", "submit", "confirm",
    "delete", "cancel", "reset", "book", "order",
})

_PERMISSION_MAP: Dict[str, PermissionLevel] = {
    "read": PermissionLevel.READ_ONLY,
    "view": PermissionLevel.READ_ONLY,
    "find": PermissionLevel.READ_ONLY,
    "extract": PermissionLevel.READ_ONLY,
    "fill": PermissionLevel.FORM_FILL,
    "type": PermissionLevel.FORM_FILL,
    "enter": PermissionLevel.FORM_FILL,
    "save": PermissionLevel.SUBMIT,
    "submit": PermissionLevel.SUBMIT,
    "search": PermissionLevel.SUBMIT,
    "add to cart": PermissionLevel.SUBMIT,
    "purchase": PermissionLevel.PURCHASE,
    "buy": PermissionLevel.PURCHASE,
    "checkout": PermissionLevel.PURCHASE,
    "pay": PermissionLevel.PURCHASE,
    "order": PermissionLevel.PURCHASE,
    "delete": PermissionLevel.ADMIN,
    "remove": PermissionLevel.ADMIN,
    "admin": PermissionLevel.ADMIN,
}


def _extract_facts(text: str) -> Dict[str, Any]:
    """Pull structured values from free text. Pure deterministic extraction.

    Domain-agnostic: extracts emails, phones, amounts, dates, quantities,
    quoted strings, and coupon/promo codes from ANY natural-language text.
    Works on shopping, travel, banking, government portals, etc.
    """
    facts: Dict[str, Any] = {}
    emails = _EMAIL_RE.findall(text)
    if emails:
        facts["email"] = emails[0]
        if len(emails) > 1:
            facts["emails"] = emails
    phones = _PHONE_RE.findall(text)
    if phones:
        facts["phone"] = phones[0]
    amounts = _AMOUNT_RE.findall(text)
    if amounts:
        facts["amount"] = amounts[0]
    dates = _DATE_RE.findall(text)
    if dates:
        facts["date"] = dates[0]
    qtys = _QUANTITY_RE.findall(text)
    if qtys:
        facts["quantity"] = qtys[0]
    name_m = re.search(r'\b(?:name|for|with)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text)
    if name_m:
        facts["name"] = name_m.group(1).strip()
    quoted = [m.group(1) or m.group(2) for m in _QUOTED_RE.finditer(text)]
    if quoted:
        facts["quoted_values"] = quoted
    # Coupon/promo codes (standalone uppercase alphanumeric tokens)
    codes = _CODE_RE.findall(text)
    if codes:
        # Filter out common false positives (HTTP, URL, HTML, etc.)
        _noise = {"HTTP", "HTTPS", "HTML", "CSS", "JSON", "XML",
                  "POST", "GET", "PUT", "API", "URL", "NULL"}
        codes = [c for c in codes if c not in _noise]
        if codes:
            facts["codes"] = codes
    return facts


# ──────────────────────────────────────────────────────────────────────────────
# Generic missing-field detection (domain-agnostic, HITL-first)
# ──────────────────────────────────────────────────────────────────────────────

# Map: field keywords the user might mention → fact keys that would satisfy them.
# If the user mentions a keyword but no matching fact was extracted, it is MANDATORY.
_FIELD_KEYWORD_MAP: Dict[str, Dict[str, str]] = {
    # key -> {"fact_key": what to look for in facts, "question": HITL prompt}
    "email":    {"fact_key": "email",
                 "question": "What email address should I use?"},
    "e-mail":   {"fact_key": "email",
                 "question": "What email address should I use?"},
    "phone":    {"fact_key": "phone",
                 "question": "What phone number should I use?"},
    "mobile":   {"fact_key": "phone",
                 "question": "What mobile number should I use?"},
    "address":  {"fact_key": "address",
                 "question": "What address should I enter?"},
    "name":     {"fact_key": "name",
                 "question": "What name should I use?"},
    "password": {"fact_key": "password",
                 "question": "What password should I enter?"},
    "card":     {"fact_key": "card_number",
                 "question": "What payment card number should I use?"},
    "cvv":      {"fact_key": "cvv",
                 "question": "What is the CVV for the payment card?"},
    "pin":      {"fact_key": "pin",
                 "question": "What PIN code should I enter?"},
    "date":     {"fact_key": "date",
                 "question": "What date should I select?"},
    "city":     {"fact_key": "city",
                 "question": "What city should I enter?"},
    "state":    {"fact_key": "state",
                 "question": "What state/province should I select?"},
    "country":  {"fact_key": "country",
                 "question": "What country should I select?"},
    "zip":      {"fact_key": "zip",
                 "question": "What ZIP/postal code should I enter?"},
    "coupon":   {"fact_key": "codes",
                 "question": "What coupon code should I apply?"},
    "promo":    {"fact_key": "codes",
                 "question": "What promo code should I apply?"},
}

# Action verbs that signal the user expects the agent to FILL a field
# (not just navigate past it). If these verbs appear alongside a field
# keyword, and no concrete value was extracted, it is MANDATORY.
_FILL_VERBS = frozenset({
    "fill", "enter", "type", "input", "write", "put", "set",
    "use", "apply", "provide", "submit", "with",
})


def _detect_mentioned_but_missing_fields(
    text: str, facts: Dict[str, Any]
) -> List[MissingItem]:
    """Detect fields the user mentions but does NOT provide a value for.

    Domain-agnostic: works on shopping, travel, banking, government, etc.
    Every detected gap becomes MANDATORY → triggers HITL, never guessed.
    """
    lower = text.lower()
    words = set(lower.split())
    has_fill_verb = bool(words & _FILL_VERBS)
    missing: List[MissingItem] = []
    seen_fact_keys: set = set()

    # Compound-term suppression: avoid false positives where a keyword
    # appears as part of a larger phrase that means something else.
    # e.g. "email address" should NOT trigger the standalone "address" field.
    _suppress: Dict[str, List[str]] = {
        "address":  ["email address", "e-mail address", "web address",
                     "ip address", "mac address", "url address"],
        "name":     ["file name", "filename", "user name", "username",
                     "domain name", "host name", "hostname"],
        "date":     ["update", "to date", "out of date", "up to date"],
        "state":    ["state of", "state-of-the-art", "current state"],
        "pin":      ["pinch", "pinned", "opinion"],
        "card":     ["discard"],
        "promo":    ["promotion", "promotions", "promotional"],
    }

    for keyword, meta in _FIELD_KEYWORD_MAP.items():
        if keyword not in lower:
            continue

        # Check compound suppressions: if the keyword only appears as
        # part of a larger phrase, skip it.
        if keyword in _suppress:
            suppressed = False
            for compound in _suppress[keyword]:
                if compound in lower:
                    # Only suppress if ALL occurrences of keyword are
                    # within compound phrases. Simple heuristic: if the
                    # compound is present and keyword count matches,
                    # it is a false positive.
                    remaining = lower.replace(compound, "")
                    if keyword not in remaining:
                        suppressed = True
                        break
            if suppressed:
                continue

        fact_key = meta["fact_key"]
        if fact_key in seen_fact_keys:
            continue  # avoid duplicate questions for email / e-mail
        seen_fact_keys.add(fact_key)

        # Check if the user already provided a concrete value
        if fact_key in facts and facts[fact_key]:
            continue

        # The user mentions this field. Did they also use a fill-verb?
        # Even without a fill-verb, if the prompt implies the agent
        # should act (permission >= FORM_FILL), it is still mandatory.
        if has_fill_verb or any(
            v in lower for v in ("fill", "enter", "type", f"with {keyword}",
                                  f"use {keyword}", f"my {keyword}",
                                  f"the {keyword}")):
            missing.append(MissingItem(
                key=fact_key,
                description=(
                    f"The request mentions '{keyword}' but no concrete "
                    f"value was provided"),
                severity=MissingItemSeverity.MANDATORY,
                suggested_question=meta["question"],
            ))

    return missing


def _infer_permission(text: str) -> PermissionLevel:
    """Infer minimum sufficient permission from goal text.

    Starts at READ_ONLY and escalates only when an escalating verb is
    detected. Falls back to SUBMIT (safe default) when no verbs match
    and the goal is ambiguous.
    """
    lower = text.lower()
    order = [PermissionLevel.READ_ONLY, PermissionLevel.FORM_FILL,
             PermissionLevel.SUBMIT, PermissionLevel.PURCHASE,
             PermissionLevel.ADMIN]

    # Detect all matching verbs
    matched: list = []
    for phrase, level in _PERMISSION_MAP.items():
        if phrase in lower:
            matched.append(level)

    if not matched:
        # No recognizable verbs: conservative SUBMIT (form interactions)
        return PermissionLevel.SUBMIT

    # Return the highest-privilege permission detected
    best = matched[0]
    for level in matched[1:]:
        if order.index(level) > order.index(best):
            best = level
    return best


def _infer_consequential(text: str) -> List[str]:
    lower = text.lower()
    return [v for v in _CONSEQUENTIAL_ACTIONS if v in lower]


def _requires_confirmation(text: str) -> bool:
    lower = text.lower()
    return any(v in lower for v in ("purchase", "buy", "pay", "checkout",
                                     "order", "delete", "cancel", "reset"))


def _build_completion_criteria(text: str, facts: Dict[str, Any]) -> List[CompletionPredicate]:
    """Deterministic completion predicates derived from the goal text.

    Domain-agnostic: does NOT hardcode page names like 'cart' or
    'confirmation'. Instead it derives predicates from the user's own
    words and extracted facts.
    """
    criteria: List[CompletionPredicate] = []
    lower = text.lower()

    # ── Fact-presence signals (domain-agnostic) ──────────────────────────
    # If the user provided a concrete email, verify it appears on the page.
    if facts.get("email"):
        criteria.append(CompletionPredicate(
            predicate="CONTAINS", target="page",
            value=facts["email"], required=False))

    # Quoted values (coupon codes, product names, etc.) should be visible.
    for q in facts.get("quoted_values", []):
        if len(q) >= 3:
            criteria.append(CompletionPredicate(
                predicate="VISIBLE", target="page",
                value=q, required=False))

    # Quantity, if explicitly stated, should appear.
    if facts.get("quantity"):
        criteria.append(CompletionPredicate(
            predicate="CONTAINS", target="page",
            value=str(facts["quantity"]), required=False))

    # ── Goal-derived signals ─────────────────────────────────────────────
    # Look for explicit success indicators in the user's own words.
    _success_signals = [
        ("confirm",       "VISIBLE", "confirmation"),
        ("success",       "VISIBLE", "success"),
        ("thank you",     "VISIBLE", "thank you"),
        ("order placed",  "VISIBLE", "order placed"),
        ("booking confirmed", "VISIBLE", "booking confirmed"),
        ("payment received", "VISIBLE", "payment received"),
        ("ticket booked", "VISIBLE", "ticket booked"),
    ]
    for phrase, pred, val in _success_signals:
        if phrase in lower:
            criteria.append(CompletionPredicate(
                predicate=pred, target="page", value=val, required=False))
            break  # one success signal is enough

    # ── Fallback: at least the page state changed ────────────────────────
    # This is always present as a baseline (required=True) so the STV
    # never vacuously passes.
    criteria.append(CompletionPredicate(
        predicate="CHANGED", target="page", value="", required=True))
    return criteria


# ──────────────────────────────────────────────────────────────────────────────
# Task Compiler
# ──────────────────────────────────────────────────────────────────────────────

MAX_COMPILE_RETRIES = 2


class TaskCompiler:
    """Transform a natural-language request into a validated TaskSpec.

    Uses a two-layer extraction pipeline:
    1. Semantic (LLM via Ollama qwen2.5:7b): understands intent, extracts
       names/cities/dates/credentials from ANY prompt regardless of grammar
       or domain. Identifies missing required fields and generates HITL
       questions.
    2. Regex (deterministic): overlaid for high-precision structured values
       (emails, phone numbers, amounts, dates, codes).

    If Ollama is unavailable, falls back to regex + keyword detection.
    Integrates with the existing ask_handler (HITL) for mandatory missing
    information. Never adds a second clarification mechanism.
    """

    def __init__(
        self,
        ask_handler=None,
        max_asks: int = 3,
        task_id: Optional[str] = None,
        use_semantic: bool = True,
        semantic_model: Optional[str] = None,
    ) -> None:
        self.ask_handler = ask_handler
        self.max_asks = max_asks
        self._ask_count = 0
        self._forced_task_id = task_id
        # Semantic extractor: LLM-powered, domain-agnostic
        self._semantic: Optional[Any] = None
        if use_semantic and _SEMANTIC_AVAILABLE:
            self._semantic = SemanticExtractor(model=semantic_model)
            logger.info("TaskCompiler: semantic extraction enabled "
                        "(Ollama qwen2.5:7b / llama3)")
        else:
            logger.info("TaskCompiler: semantic extraction disabled; "
                        "using regex-only extraction")

    # ── public ───────────────────────────────────────────────────────────────

    def compile(self, request: str) -> CompilationResult:
        """Compile *request* into a validated TaskSpec.

        Returns a CompilationResult. The caller must check .ok before
        passing .spec to the planner; an invalid spec never reaches execution.
        """
        if not request or not request.strip():
            return CompilationResult(error="empty or whitespace-only request")

        for attempt in range(MAX_COMPILE_RETRIES):
            try:
                result = self._compile_once(request)
            except Exception as exc:
                logger.warning(f"Compilation attempt {attempt+1} crashed: {exc}")
                if attempt < MAX_COMPILE_RETRIES - 1:
                    continue
                return CompilationResult(
                    error=f"compilation failed after {MAX_COMPILE_RETRIES} "
                          f"attempts: {exc}")

            if result.ok:
                return result

            # Clarification needed: route through existing HITL
            if result.needs_clarification:
                if self.ask_handler and self._ask_count < self.max_asks:
                    self._ask_count += 1
                    logger.info(f"HITL clarification "
                                f"({self._ask_count}/{self.max_asks}): "
                                f"{result.clarification_question!r}")
                    try:
                        answer = self.ask_handler(result.clarification_question)
                    except Exception as e:
                        return CompilationResult(
                            error=f"ask_handler failed: {e}",
                            needs_clarification=True,
                            clarification_question=result.clarification_question)
                    # Re-compile with the enriched request
                    request = f"{request}\n[User clarification]: {answer}"
                    continue
                # No handler or budget exhausted: return the clarification
                # result as-is so the controller can surface it properly
                return result

            # Hard error
            return result

        return CompilationResult(error="compile retry budget exhausted")

    # ── private ──────────────────────────────────────────────────────────────

    def _compile_once(self, request: str) -> CompilationResult:
        """Single compilation pass: semantic LLM → regex overlay → TaskSpec.

        Layer 1 (Semantic): local LLM extracts all facts and identifies
          missing required fields for ANY domain prompt.
        Layer 2 (Regex): overlaid for high-precision structured values.
        Layer 3 (Keyword): fallback missing-field detection when no LLM.
        """
        text = request.strip()

        # ── Layer 1: Regex extraction (always runs — fast, high precision) ────
        regex_facts = _extract_facts(text)

        # ── Layer 2: Semantic extraction (LLM, domain-agnostic) ─────────────
        semantic_result = None
        sem_missing: List[MissingItem] = []

        if self._semantic is not None:
            try:
                semantic_result = self._semantic.extract(text)
                logger.info(
                    f"Semantic extraction: intent={semantic_result.intent!r}, "
                    f"facts={list(semantic_result.extracted_facts.keys())}, "
                    f"missing={[m['field'] for m in semantic_result.missing_fields]}, "
                    f"model={semantic_result.model_used!r}, "
                    f"fallback={semantic_result.fell_back_to_regex}")
            except Exception as exc:
                logger.warning(f"Semantic extractor crashed: {exc}; "
                               f"continuing with regex-only")
                semantic_result = None

        # ── Merge facts: semantic broad coverage + regex precision ──────────
        if semantic_result and not semantic_result.fell_back_to_regex:
            facts = merge_facts(semantic_result.extracted_facts, regex_facts)
        else:
            facts = regex_facts

        # ── Build missing-field list ───────────────────────────────────────
        if semantic_result and not semantic_result.fell_back_to_regex:
            # LLM-identified missing fields: semantically aware
            for mf in semantic_result.missing_fields:
                sem_missing.append(MissingItem(
                    key=mf["field"],
                    description=mf.get("reason", mf["field"]),
                    severity=MissingItemSeverity.MANDATORY,
                    suggested_question=mf["question"],
                ))
            # Also check: task itself is completely unclear
            if semantic_result.task_unclear:
                q = (semantic_result.clarification_needed or
                     "Could you describe what you want me to do in more detail?")
                return CompilationResult(
                    spec=None,
                    needs_clarification=True,
                    clarification_question=q)
            # If LLM says a clarification is needed about the overall goal
            if semantic_result.clarification_needed:
                return CompilationResult(
                    spec=None,
                    needs_clarification=True,
                    clarification_question=semantic_result.clarification_needed)
        else:
            # Fallback: keyword-based missing detection
            sem_missing = _detect_mentioned_but_missing_fields(text, facts)

        # ── Permission, consequential, confirmation ────────────────────────
        permission = _infer_permission(text)
        consequential = _infer_consequential(text)
        needs_confirm = _requires_confirmation(text)
        completion = _build_completion_criteria(text, facts)

        # Build normalized goal
        normalized = " ".join(text.split())
        if len(normalized) > 600:
            normalized = normalized[:597] + "..."

        defaults: List[DisclosedDefault] = []
        constraints: List[ExplicitConstraint] = []
        for key, val in facts.items():
            constraints.append(ExplicitConstraint(
                key=key, value=val, source=FactSource.USER_PROVIDED))

        # Build the spec
        try:
            task_id = self._forced_task_id or str(uuid.uuid4())[:8]
            spec = TaskSpec(
                task_id=task_id,
                original_request=text,
                normalized_goal=normalized if normalized else text,
                constraints=constraints,
                user_provided_facts=facts,
                missing_information=sem_missing,
                disclosed_defaults=defaults,
                completion_criteria=completion,
                max_permission=permission,
                consequential_actions=consequential,
                requires_confirmation=needs_confirm,
            )
        except Exception as exc:
            return CompilationResult(error=f"TaskSpec validation failed: {exc}")

        # Check for mandatory gaps that need HITL
        mandatory = spec.mandatory_missing()
        if mandatory:
            q = mandatory[0].suggested_question or (
                f"I need more information: {mandatory[0].description}")
            return CompilationResult(
                spec=None,
                needs_clarification=True,
                clarification_question=q)

        return CompilationResult(spec=spec)


# ──────────────────────────────────────────────────────────────────────────────
# Validation gate (called by controller before passing spec to planner)
# ──────────────────────────────────────────────────────────────────────────────

class ContractViolation(Exception):
    """Raised when a SubgoalContract fails the pre-execution safety gate."""


def validate_spec_for_execution(spec: TaskSpec) -> List[str]:
    """Deterministic pre-execution checks. Returns list of violations."""
    violations: List[str] = []
    if not spec.normalized_goal.strip():
        violations.append("empty normalized_goal")
    if not spec.completion_criteria:
        violations.append("no completion criteria — task cannot be verified")
    for crit in spec.completion_criteria:
        if crit.required and not crit.target and not crit.value:
            violations.append(f"required predicate {crit.predicate} is untestable")
    if spec.mandatory_missing():
        keys = [m.key for m in spec.mandatory_missing()]
        violations.append(f"mandatory information still missing: {keys}")
    return violations


def spec_is_compatible_with_contract(
        spec: TaskSpec, contract: Dict[str, Any]) -> List[str]:
    """Check a SubgoalContract against its parent TaskSpec.

    Returns a list of violation strings (empty = OK).
    Enforced deterministically — the LLM cannot bypass this gate.
    """
    violations: List[str] = []
    goal_words = set(spec.normalized_goal.lower().split())

    # 1. The contract must reference the spec's goal
    obj = str(contract.get("objective", "") or "")
    if obj:
        obj_words = set(obj.lower().split())
        if not (obj_words & goal_words):
            violations.append(
                "subgoal objective shares no words with the TaskSpec goal")

    # 2. Permission check for consequential actions
    action_intent = str(contract.get("action_intent", "") or "").lower()
    for ca in _CONSEQUENTIAL_ACTIONS:
        if ca in action_intent:
            order = [PermissionLevel.READ_ONLY, PermissionLevel.FORM_FILL,
                     PermissionLevel.SUBMIT, PermissionLevel.PURCHASE,
                     PermissionLevel.ADMIN]
            needed = PermissionLevel.PURCHASE
            if order.index(spec.max_permission) < order.index(needed):
                violations.append(
                    f"consequential action {ca!r} exceeds authorized "
                    f"permission {spec.max_permission!r}")

    # 3. Action intent must not be empty
    if not action_intent:
        violations.append("contract has no action_intent")

    # 4. Postconditions must exist and be testable
    postconditions = contract.get("postconditions", []) or []
    if not postconditions:
        violations.append("contract has no postconditions — completion "
                          "cannot be verified")

    return violations
