"""Structured action schema using Pydantic.

The VLM's raw dict is validated here before anything touches the browser.
Malformed outputs → ValidationError → treated as failed step, never executed.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, model_validator


class ActionType(str, Enum):
    CLICK = "click"
    TYPE = "type"
    PRESS = "press"
    SCROLL = "scroll"
    WAIT = "wait"
    DONE = "done"
    FAIL = "fail"
    ASK_USER = "ask_user"


# Allowlist for the `press` action: keyboard keys the agent may send to the
# focused element. Deliberately tiny (form submission / dialog dismissal);
# arbitrary key chords stay out of the action space.
PRESS_KEYS = ("Enter", "Tab", "Escape")

_PRESS_KEY_ALIASES = {
    "enter": "Enter",
    "return": "Enter",
    "tab": "Tab",
    "escape": "Escape",
    "esc": "Escape",
}


def normalize_press_key(key: Any) -> str:
    """Canonicalize a press key (default Enter); raises ValueError if banned."""
    if key is None or (isinstance(key, str) and not key.strip()):
        return "Enter"
    canonical = _PRESS_KEY_ALIASES.get(str(key).strip().lower())
    if canonical is None:
        raise ValueError(
            f"press key {key!r} not allowed (choose from {list(PRESS_KEYS)})")
    return canonical


class AskUserAction(BaseModel):
    """Human-in-the-loop action: pause and ask the human a question.

    This model documents the VLM-visible contract for `ask_user`. The
    runtime `Action` below mirrors these fields (flat model, see below)
    so existing parsers/executors keep working without a Union refactor.
    """
    action_type: str = Field(default="ask_user", description="Must be 'ask_user'")
    question: str = Field(
        description="Exact question to ask the human (e.g. missing "
                    "credentials, ambiguous goal, personal info needed).")
    reason: str = Field(default="", description="Model's explanation")

    model_config = {"use_enum_values": True}


class Action(BaseModel):
    action_type: ActionType
    x: Optional[float] = Field(default=None, description="Click/type x in pixels")
    y: Optional[float] = Field(default=None, description="Click/type y in pixels")
    value: Optional[str] = Field(default=None, description="Text for type, or scroll dx,dy")
    key: Optional[str] = Field(
        default=None,
        description="Key for press (Enter/Tab/Escape, default Enter); "
                    "acts on the focused element, no coordinates needed")
    dx: Optional[float] = Field(default=None, description="Scroll delta x")
    dy: Optional[float] = Field(default=None, description="Scroll delta y")
    target_id: Optional[int] = Field(default=None, description="SoM marker id")
    target: Optional[str] = Field(default=None, description="Target element description or text")
    question: Optional[str] = Field(
        default=None,
        description="Exact question to ask the human; required iff "
                    "action_type is ask_user (HITL).")
    human_answer: Optional[str] = Field(
        default=None,
        description="Human's answer, injected by the controller after "
                    "prompting; never emitted by the VLM.")
    reason: str = Field(default="", description="Model's explanation")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    method: str = Field(default="unknown", description="coordinate | som")
    expected_transition: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="STV expectation set by the long-horizon controller "
                    "(must_change/must_be_true/must_remain_true/"
                    "must_not_happen). Ignored by the executor.")

    @field_validator("x", "y")
    @classmethod
    def _check_coords(cls, v):
        if v is not None and not (-10000 <= v <= 10000):
            raise ValueError(f"Coordinate {v} out of plausible range")
        return v

    @model_validator(mode="after")
    def _check_ask_user(self):
        # use_enum_values=True coerces to plain str; compare by value.
        is_ask = self.action_type == ActionType.ASK_USER.value \
            if isinstance(self.action_type, str) else self.action_type == ActionType.ASK_USER
        if is_ask:
            if not self.question or not str(self.question).strip():
                raise ValueError("ask_user requires a non-empty 'question'")
            if self.x is not None or self.y is not None \
                    or self.target_id is not None:
                raise ValueError(
                    "ask_user must not carry x/y/target_id coordinates")
        else:
            if self.question is not None:
                raise ValueError(
                    f"question is only valid for ask_user, got {self.action_type}")
            if self.human_answer is not None:
                raise ValueError(
                    "human_answer is only valid for ask_user")
        return self

    def requires_coordinates(self) -> bool:
        return self.action_type in (ActionType.CLICK, ActionType.TYPE)

    @field_validator("key")
    @classmethod
    def _check_key(cls, v):
        if v is None:
            return v
        return normalize_press_key(v)

    def is_terminal(self) -> bool:
        return self.action_type in (ActionType.DONE, ActionType.FAIL)

    def is_success(self) -> bool:
        return self.action_type == ActionType.DONE

    def is_ask_user(self) -> bool:
        at = self.action_type
        return at == ActionType.ASK_USER or at == ActionType.ASK_USER.value

    model_config = {"use_enum_values": True, "extra": "ignore"}


def parse_action(data: Dict[str, Any]) -> Action:
    """Validate a raw grounder dict into an Action.

    Accepts both grounder output styles:
      coordinate: {"action": "click", "x":.., "y":.., ...}
      som:        {"action_type": "click", "x":.., "y":.., "target_id":..}
    """
    normalized = dict(data)
    if "action" in normalized and "action_type" not in normalized:
        normalized["action_type"] = normalized.pop("action")
    # Normalize common VLM action type aliases
    raw_type = str(normalized.get("action_type") or "").lower()
    if raw_type in ("left_click", "click_coordinate", "mouse_click"):
        normalized["action_type"] = "click"
    elif raw_type in ("type_text", "fill", "input"):
        normalized["action_type"] = "type"
    elif raw_type in ("press_key", "key_press"):
        normalized["action_type"] = "press"
    if normalized.get("action_type") == "type" and "value" not in normalized and "text" in normalized:
        normalized["value"] = normalized["text"]
    # HITL leniency (real-VLM evidence): models may echo a `question` /
    # `human_answer` key on non-ask_user actions. Those keys are only
    # meaningful for ask_user, so strip them instead of failing the turn.
    if normalized.get("action_type") != "ask_user":
        normalized.pop("question", None)
        normalized.pop("human_answer", None)
    # scroll value like "0,500" or {"dx":..,"dy":..}
    if normalized.get("action_type") == "scroll" and "value" in normalized:
        v = normalized["value"]
        if isinstance(v, str) and "," in v:
            try:
                dx_s, dy_s = v.split(",", 1)
                normalized["dx"] = float(dx_s.strip())
                normalized["dy"] = float(dy_s.strip())
            except ValueError:
                pass
        elif isinstance(v, dict):
            normalized["dx"] = v.get("dx", normalized.get("dx"))
            normalized["dy"] = v.get("dy", normalized.get("dy"))
    # default scroll downward if unspecified
    if normalized.get("action_type") == "scroll":
        normalized.setdefault("dy", 500.0)
        normalized.setdefault("dx", 0.0)
    # press acts on the focused element; default key is Enter.
    if normalized.get("action_type") == "press":
        try:
            normalized["key"] = normalize_press_key(normalized.get("key"))
        except ValueError:
            pass  # leave for Pydantic to reject with a clear error
    return Action(**normalized)
