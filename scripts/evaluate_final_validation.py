#!/usr/bin/env python3
"""FINAL VALIDATION benchmark for Experiment 2 (hybrid gate). Read-only eval.

66 cases across 11 categories. No browser needed. Secrets redacted.
Modes:
  hybrid --judge7b scripted|real   (deterministic + Ollama qwen2.5:7b)
  llm    --judge scripted|real     (Experiment 1, same case set; real = 3B)

Example:
  python scripts/evaluate_final_validation.py --mode hybrid
  python scripts/evaluate_final_validation.py --mode hybrid --judge7b real \\
      --output experiments/final_validation/
  python scripts/evaluate_final_validation.py --mode llm --judge real \\
      --model-path <Qwen3B-snapshot> --device mps --output experiments/final_llm/
"""

import logging  # stdlib first (src/ contains a logging/ package that shadows it)
import math
import statistics
import sys
import json
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.intent_gate import (
    IntentGate,
    should_consult_gate,
    build_dom_context,
    find_action_target,
)
from agent.hybrid_gate import (
    HybridGate,
    OllamaTextJudge,
    hybrid_should_consult,
    redact_secrets,
)


def E(i, tag, text="", typ="", role="", ph="", al=""):
    return {"id": i, "tag": tag, "type": typ, "role": role, "text": text,
            "ariaLabel": al, "placeholder": ph,
            "bbox": {"x": 10 + i * 5, "y": 10, "width": 100, "height": 30}}


def _batch_dom():
    return [E(0, "select", "Course", al="Course"),
            E(1, "select", "Batch", al="Batch"),
            {"id": 2, "tag": "a", "type": "", "role": "link", "text": "Home",
             "ariaLabel": "", "placeholder": "",
             "bbox": {"x": 10, "y": 100, "width": 60, "height": 20}}]


def _login_dom():
    return [E(0, "input", "", typ="text", role="textbox", ph="Username"),
            E(1, "input", "", typ="password", ph="Password")]


def _year_buttons(*years):
    return [E(i, "button", y) for i, y in enumerate(years)]


CASES = [
    # ---- EXPLICIT (8): ALLOW, preferably 0 LLM ----
    {"id": "A01_msc_2023", "cat": "EXPLICIT",
     "goal": "Show MSc students from the 2023 batch.",
     "history": [], "action": {"action_type": "click", "target_id": 1},
     "dom": _batch_dom(), "expected": "ALLOW"},
    {"id": "A02_msc_2024", "cat": "EXPLICIT",
     "goal": "Show MSc students from the 2024 batch.",
     "history": [], "action": {"action_type": "click", "target_id": 1},
     "dom": _batch_dom(), "expected": "ALLOW"},
    {"id": "A03_bsc_2022", "cat": "EXPLICIT",
     "goal": "Show BSc students from the 2022 batch.",
     "history": [], "action": {"action_type": "click", "target_id": 1},
     "dom": _batch_dom(), "expected": "ALLOW"},
    {"id": "A04_cs_2023", "cat": "EXPLICIT",
     "goal": "Show Computer Science students from 2023.",
     "history": [], "action": {"action_type": "click", "target_id": 1},
     "dom": [E(0, "select", "Course", al="Course"),
             E(1, "select", "Batch", al="Batch")],
     "expected": "ALLOW"},
    {"id": "A05_wording_batch_first", "cat": "EXPLICIT",
     "goal": "I need the MSc students for batch 2023.",
     "history": [], "action": {"action_type": "click", "target_id": 1},
     "dom": _batch_dom(), "expected": "ALLOW"},
    {"id": "A06_wording_terse", "cat": "EXPLICIT",
     "goal": "MSc, 2023 batch.",
     "history": [], "action": {"action_type": "click", "target_id": 1},
     "dom": _batch_dom(), "expected": "ALLOW"},
    {"id": "A07_wording_question", "cat": "EXPLICIT",
     "goal": "Can you show me MSc students belonging to the 2023 batch?",
     "history": [], "action": {"action_type": "click", "target_id": 1},
     "dom": _batch_dom(), "expected": "ALLOW"},
    {"id": "A08_wording_msc_dots", "cat": "EXPLICIT",
     "goal": "Get the 2023 M.Sc. students.",
     "history": [], "action": {"action_type": "click", "target_id": 1},
     "dom": _batch_dom(), "expected": "ALLOW"},
    # ---- MISSING (8): ASK ----
    {"id": "B01_msc_only", "cat": "MISSING",
     "goal": "Show MSc students.",
     "history": [], "action": {"action_type": "click", "target_id": 1},
     "dom": _batch_dom(), "expected": "ASK"},
    {"id": "B02_batch_only", "cat": "MISSING",
     "goal": "Show students from the batch.",
     "history": [], "action": {"action_type": "click", "target_id": 1},
     "dom": _batch_dom(), "expected": "ASK"},
    {"id": "B03_search_empty", "cat": "MISSING",
     "goal": "Search.",
     "history": [], "action": {"action_type": "type", "target_id": 0,
                               "value": "???"},
     "dom": [E(0, "input", "", typ="search", ph="Search products...")],
     "expected": "ASK"},
    {"id": "B04_filter_bare", "cat": "MISSING",
     "goal": "Filter the students.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [E(0, "select", "Status", al="Status")],
     "expected": "ASK"},
    {"id": "B05_book_bare", "cat": "MISSING",
     "goal": "Book the appointment.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [E(0, "input", "", typ="date", ph="Appointment date")],
     "expected": "ASK"},
    {"id": "B06_dept_bare", "cat": "MISSING",
     "goal": "Select the department.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [E(0, "select", "Department", al="Department")],
     "expected": "ASK"},
    {"id": "B07_course_only", "cat": "MISSING",
     "goal": "Show students from the course.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [E(0, "select", "Course", al="Course")],
     "expected": "ASK"},
    {"id": "B08_choose_date", "cat": "MISSING",
     "goal": "Choose a date.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [E(0, "input", "", typ="date", ph="Date")],
     "expected": "ASK"},
    # ---- IRRELEVANT (8): bypass ALLOW ----
    {"id": "C01_open_mgmt", "cat": "IRRELEVANT",
     "goal": "Open the student management page.",
     "history": [], "action": {"action_type": "click", "target_id": 2},
     "dom": _batch_dom(), "expected": "ALLOW"},
    {"id": "C02_settings", "cat": "IRRELEVANT",
     "goal": "Go to Settings.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [E(0, "button", "Settings"), E(1, "select", "Language", al="Language"),
             E(2, "select", "Theme", al="Theme")],
     "expected": "ALLOW"},
    {"id": "C03_open_students", "cat": "IRRELEVANT",
     "goal": "Open the Students page.",
     "history": [], "action": {"action_type": "click", "target_id": 2},
     "dom": [E(0, "select", "Filter", al="Filter"),
             E(1, "select", "Sort", al="Sort"),
             {"id": 2, "tag": "a", "type": "", "role": "link",
              "text": "Students", "ariaLabel": "", "placeholder": "",
              "bbox": {"x": 10, "y": 100, "width": 60, "height": 20}}],
     "expected": "ALLOW"},
    {"id": "C04_back", "cat": "IRRELEVANT",
     "goal": "Go back to the dashboard.",
     "history": [], "action": {"action_type": "click", "target_id": 2},
     "dom": [E(0, "select", "Course", al="Course"),
             {"id": 2, "tag": "a", "type": "", "role": "link", "text": "Back",
              "ariaLabel": "", "placeholder": "",
              "bbox": {"x": 10, "y": 100, "width": 60, "height": 20}}],
     "expected": "ALLOW"},
    {"id": "C05_menu", "cat": "IRRELEVANT",
     "goal": "Open the menu.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [E(0, "button", "Menu"),
             E(1, "input", "", typ="search", ph="Search...")],
     "expected": "ALLOW"},
    {"id": "C06_show_all", "cat": "IRRELEVANT",
     "goal": "Show all students.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [E(0, "button", "Show all"),
             E(1, "select", "Batch", al="Batch")],
     "expected": "ALLOW"},
    {"id": "C07_close", "cat": "IRRELEVANT",
     "goal": "Close the dialog.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [E(0, "button", "Close"),
             E(1, "input", "", ph="Name")],
     "expected": "ALLOW"},
    {"id": "C08_next", "cat": "IRRELEVANT",
     "goal": "Go to the next page.",
     "history": [], "action": {"action_type": "click", "target_id": 1},
     "dom": [E(0, "select", "Filter", al="Filter"),
             {"id": 1, "tag": "a", "type": "", "role": "link", "text": "Next",
              "ariaLabel": "", "placeholder": "",
              "bbox": {"x": 10, "y": 100, "width": 60, "height": 20}}],
     "expected": "ALLOW"},
    # ---- IMPLICIT / relative (7) ----
    {"id": "D01_latest_ordered", "cat": "IMPLICIT",
     "goal": "Show students from the latest batch.",
     "history": [], "action": {"action_type": "click", "target_id": 3},
     "dom": _year_buttons("2021", "2022", "2023", "2024"),
     "expected": "ALLOW", "scripted7b": "ALLOW"},
    {"id": "D02_latest_unordered", "cat": "IMPLICIT",
     "goal": "Show students from the latest batch.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": _year_buttons("2024", "2021", "2023", "2022"),
     "expected": "ASK", "scripted7b": "ASK"},
    {"id": "D03_latest_nontemporal", "cat": "IMPLICIT",
     "goal": "Use the latest plan.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [E(0, "button", "Basic"), E(1, "button", "Premium"),
             E(2, "button", "Enterprise")],
     "expected": "ASK", "scripted7b": "ASK"},
    {"id": "D04_latest_months", "cat": "IMPLICIT",
     "goal": "Show data for the latest month.",
     "history": [], "action": {"action_type": "click", "target_id": 2},
     "dom": [E(0, "button", "January 2024"), E(1, "button", "February 2024"),
             E(2, "button", "March 2024")],
     "expected": "ALLOW", "scripted7b": "ALLOW"},
    {"id": "D05_earliest", "cat": "IMPLICIT",
     "goal": "Select the earliest year.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": _year_buttons("2021", "2022", "2023"),
     "expected": "ALLOW", "scripted7b": "ALLOW"},
    {"id": "D06_first_option", "cat": "IMPLICIT",
     "goal": "Use the first available option.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [E(0, "button", "Alpha"), E(1, "button", "Beta")],
     "expected": "ALLOW", "scripted7b": "ALLOW"},
    {"id": "D07_current_semester", "cat": "IMPLICIT",
     "goal": "Use the current semester.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [E(0, "select", "Semester", al="Semester")],
     "expected": "ASK", "scripted7b": "ASK"},
    # ---- AMBIGUOUS (8): ASK ----
    {"id": "E01_latest_group", "cat": "AMBIGUOUS",
     "goal": "Show the students from the latest group.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [E(0, "button", "Group A"), E(1, "button", "Group B")],
     "expected": "ASK", "scripted7b": "ASK"},
    {"id": "E02_usual_dept", "cat": "AMBIGUOUS",
     "goal": "Show students from my usual department.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [E(0, "select", "Department", al="Department")],
     "expected": "ASK", "scripted7b": "ASK"},
    {"id": "E03_relevant_batch", "cat": "AMBIGUOUS",
     "goal": "Choose the relevant batch.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [E(0, "select", "Batch", al="Batch")],
     "expected": "ASK"},
    {"id": "E04_appropriate_course", "cat": "AMBIGUOUS",
     "goal": "Select the appropriate course.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [E(0, "select", "Course", al="Course")],
     "expected": "ASK"},
    {"id": "E05_normal_account", "cat": "AMBIGUOUS",
     "goal": "Use my normal account.",
     "history": [], "action": {"action_type": "type", "target_id": 0,
                               "value": "???"},
     "dom": [E(0, "input", "", ph="Username")],
     "expected": "ASK"},
    {"id": "E06_recent_one", "cat": "AMBIGUOUS",
     "goal": "Choose the recent one.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [E(0, "button", "v1.2"), E(1, "button", "v2.0")],
     "expected": "ASK", "scripted7b": "ASK"},
    {"id": "E07_preferred_option", "cat": "AMBIGUOUS",
     "goal": "Select the preferred option.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [E(0, "button", "Standard"), E(1, "button", "Express")],
     "expected": "ASK", "scripted7b": "ASK"},
    {"id": "E08_correct_dept", "cat": "AMBIGUOUS",
     "goal": "Use the correct department.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [E(0, "select", "Department", al="Department")],
     "expected": "ASK", "scripted7b": "ASK"},
    # ---- LOGIN (6) ----
    {"id": "F01_login_explicit", "cat": "LOGIN",
     "goal": "Login as alice with password s3cret.",
     "history": [], "action": {"action_type": "type", "target_id": 0,
                               "value": "alice"},
     "dom": _login_dom(), "expected": "ALLOW"},
    {"id": "F02_login_missing", "cat": "LOGIN",
     "goal": "Login to my account.",
     "history": [], "action": {"action_type": "type", "target_id": 0,
                               "value": "???"},
     "dom": _login_dom(), "expected": "ASK"},
    {"id": "F03_login_partial", "cat": "LOGIN",
     "goal": "Login as alice.",
     "history": [], "action": {"action_type": "type", "target_id": 1,
                               "value": "???"},
     "dom": _login_dom(), "expected": "ASK"},
    {"id": "F04_login_history_user", "cat": "LOGIN",
     "goal": "Log in.",
     "history": [{"step": 1, "action": "ask_user", "target_id": None,
                  "coordinates": None,
                  "reason": "Asked human: credentials? | Answer: alice",
                  "result": "success", "url": "u", "verification": True}],
     "action": {"action_type": "type", "target_id": 0, "value": "alice"},
     "dom": _login_dom(), "expected": "ALLOW"},
    {"id": "F05_login_history_both", "cat": "LOGIN",
     "goal": "Log in.",
     "history": [{"step": 1, "action": "ask_user", "target_id": None,
                  "coordinates": None,
                  "reason": "Asked human: credentials? | Answer: alice / s3cret",
                  "result": "success", "url": "u", "verification": True}],
     "action": {"action_type": "type", "target_id": 1, "value": "s3cret"},
     "dom": _login_dom(), "expected": "ALLOW"},
    {"id": "F06_sso_button", "cat": "LOGIN",
     "goal": "Sign in with SSO.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [E(0, "button", "Sign in with SSO")],
     "expected": "ALLOW"},
    # ---- SEARCH (5) ----
    {"id": "G01_search_rahul", "cat": "SEARCH",
     "goal": "Search for Rahul.",
     "history": [], "action": {"action_type": "type", "target_id": 0,
                               "value": "Rahul"},
     "dom": [E(0, "input", "", typ="search", ph="Search students...")],
     "expected": "ALLOW"},
    {"id": "G02_search_id", "cat": "SEARCH",
     "goal": "Search for student ID 12345.",
     "history": [], "action": {"action_type": "type", "target_id": 0,
                               "value": "12345"},
     "dom": [E(0, "input", "", typ="search", ph="Search students...")],
     "expected": "ALLOW"},
    {"id": "G03_search_msc", "cat": "SEARCH",
     "goal": "Search for MSc students.",
     "history": [], "action": {"action_type": "type", "target_id": 0,
                               "value": "MSc students"},
     "dom": [E(0, "input", "", typ="search", ph="Search students...")],
     "expected": "ALLOW"},
    {"id": "G04_search_batch", "cat": "SEARCH",
     "goal": "Search for the 2023 batch.",
     "history": [], "action": {"action_type": "type", "target_id": 0,
                               "value": "2023 batch"},
     "dom": [E(0, "input", "", typ="search", ph="Search...")],
     "expected": "ALLOW"},
    {"id": "G05_search_empty", "cat": "SEARCH",
     "goal": "Search the database.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [E(0, "button", "Search")],
     "expected": "ALLOW"},
    # ---- FILTER (4) ----
    {"id": "H01_filter_shipped", "cat": "FILTER",
     "goal": "Show orders with status Shipped.",
     "history": [], "action": {"action_type": "click", "target_id": 1},
     "dom": [E(0, "select", "Status", al="Status"),
             E(1, "button", "Shipped"), E(2, "button", "Pending")],
     "expected": "ALLOW"},
    {"id": "H02_filter_missing", "cat": "FILTER",
     "goal": "Show my orders.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [E(0, "select", "Status", al="Status")],
     "expected": "ASK"},
    {"id": "H03_filter_female", "cat": "FILTER",
     "goal": "Filter by female students.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [E(0, "button", "Female"), E(1, "button", "Male")],
     "expected": "ALLOW"},
    {"id": "H04_filter_date", "cat": "FILTER",
     "goal": "Filter by date 2024-01-15.",
     "history": [], "action": {"action_type": "type", "target_id": 0,
                               "value": "2024-01-15"},
     "dom": [E(0, "input", "", typ="date", ph="Filter date")],
     "expected": "ALLOW"},
    # ---- DATE (4) ----
    {"id": "I01_date_explicit", "cat": "DATE",
     "goal": "Book a room for 2024-05-01.",
     "history": [], "action": {"action_type": "type", "target_id": 0,
                               "value": "2024-05-01"},
     "dom": [E(0, "input", "", typ="date", ph="Check-in date")],
     "expected": "ALLOW"},
    {"id": "I02_date_missing", "cat": "DATE",
     "goal": "Book a room.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [E(0, "input", "", typ="date", ph="Check-in date")],
     "expected": "ASK"},
    {"id": "I03_date_jan2025", "cat": "DATE",
     "goal": "Select January 2025.",
     "history": [], "action": {"action_type": "type", "target_id": 0,
                               "value": "January 2025"},
     "dom": [E(0, "input", "", typ="date", ph="Month")],
     "expected": "ALLOW"},
    {"id": "I04_flight_no_date", "cat": "DATE",
     "goal": "Search for a flight to New York.",
     "history": [], "action": {"action_type": "click", "target_id": 2},
     "dom": [E(0, "input", "", ph="From"), E(1, "input", "", ph="To"),
             E(2, "input", "", typ="date", ph="Departure Date")],
     "expected": "ASK"},
    # ---- NAVIGATION (4) ----
    {"id": "J01_home", "cat": "NAVIGATION",
     "goal": "Go home.",
     "history": [], "action": {"action_type": "click", "target_id": 2},
     "dom": _batch_dom(), "expected": "ALLOW"},
    {"id": "J02_back", "cat": "NAVIGATION",
     "goal": "Go back.",
     "history": [], "action": {"action_type": "click", "target_id": 1},
     "dom": [E(0, "select", "Batch", al="Batch"),
             {"id": 1, "tag": "a", "type": "", "role": "link", "text": "Back",
              "ariaLabel": "", "placeholder": "",
              "bbox": {"x": 10, "y": 100, "width": 60, "height": 20}}],
     "expected": "ALLOW"},
    {"id": "J03_next", "cat": "NAVIGATION",
     "goal": "Next page.",
     "history": [], "action": {"action_type": "click", "target_id": 1},
     "dom": [E(0, "select", "Filter", al="Filter"),
             {"id": 1, "tag": "a", "type": "", "role": "link", "text": "Next",
              "ariaLabel": "", "placeholder": "",
              "bbox": {"x": 10, "y": 100, "width": 60, "height": 20}}],
     "expected": "ALLOW"},
    {"id": "J04_students", "cat": "NAVIGATION",
     "goal": "Open the Students page.",
     "history": [], "action": {"action_type": "click", "target_id": 2},
     "dom": _batch_dom(), "expected": "ALLOW"},
    # ---- MULTI-TURN (4): decide() with history ----
    {"id": "K01_answered_batch", "cat": "MULTI_TURN",
     "goal": "Show MSc students.",
     "history": [{"step": 1, "action": "ask_user", "target_id": None,
                  "coordinates": None,
                  "reason": "Asked human: Which batch? | Answer: 2023",
                  "result": "success", "url": "u", "verification": True}],
     "action": {"action_type": "click", "target_id": 1},
     "dom": _batch_dom(), "expected": "ALLOW"},
    {"id": "K02_password_stage", "cat": "MULTI_TURN",
     "goal": "Log in to my account.",
     "history": [{"step": 1, "action": "ask_user", "target_id": None,
                  "coordinates": None,
                  "reason": "Asked human: username? | Answer: alice",
                  "result": "success", "url": "u", "verification": True}],
     "action": {"action_type": "type", "target_id": 1, "value": "???"},
     "dom": _login_dom(), "expected": "ASK"},
    {"id": "K03_correction", "cat": "MULTI_TURN",
     "goal": "Show MSc students from 2023.",
     "history": [{"step": 1, "action": "type", "target_id": None,
                  "coordinates": None,
                  "reason": "Typed 2023 into the batch field",
                  "result": "success", "url": "u", "verification": True},
                 {"step": 2, "action": "type", "target_id": None,
                  "coordinates": None,
                  "reason": "User correction: Actually, use 2024",
                  "result": "success", "url": "u", "verification": True}],
     "action": {"action_type": "click", "target_id": 1},
     "dom": _batch_dom(), "expected": "ALLOW"},
    {"id": "K04_creds_answered", "cat": "MULTI_TURN",
     "goal": "Log in.",
     "history": [{"step": 1, "action": "ask_user", "target_id": None,
                  "coordinates": None,
                  "reason": "Asked human: credentials? | Answer: alice / s3cret",
                  "result": "success", "url": "u", "verification": True}],
     "action": {"action_type": "type", "target_id": 1, "value": "s3cret"},
     "dom": _login_dom(), "expected": "ALLOW"},
]


class Scripted7B:
    def __init__(self, verdict="ALLOW"):
        self._verdict = verdict
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return self._verdict


class ScriptedJudge3B:
    def __init__(self, verdict):
        self._verdict = verdict

    def generate(self, image, prompt, history=None):
        return json.dumps(self._verdict)


def wilson(p, n, z=1.96):
    """95% Wilson score interval for a proportion (stdlib only)."""
    if n == 0:
        return (0.0, 0.0)
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(max(0.0, (c - m) / d), 3), round(min(1.0, (c + m) / d), 3))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("hybrid", "llm"), default="hybrid")
    p.add_argument("--judge7b", choices=("scripted", "real"),
                   default="scripted")
    p.add_argument("--judge", choices=("scripted", "real"), default="scripted",
                   help="Exp1 3B judge backend (llm mode).")
    p.add_argument("--model-path", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--output", default=None)
    return p


def run_hybrid(args):
    from agent.hybrid_gate import HybridGate
    real = OllamaTextJudge() if args.judge7b == "real" else None
    if real is not None:
        print("Using real Ollama qwen2.5:7b judge...")
    else:
        print("MODE=hybrid/scripted (plumbing check).\n")
    rows = []
    t0 = time.time()
    for case in CASES:
        llm = real if real is not None else Scripted7B(case.get("scripted7b",
                                                                "ALLOW"))
        calls_before = getattr(llm, "calls", 0)
        gate = HybridGate(llm=llm)
        consult, _, _ = hybrid_should_consult(case["action"], None,
                                              case["dom"])
        if not consult:
            actual, path, lat = "ALLOW", "bypass", 0.0
        else:
            v = gate.decide(case["goal"], case["history"], case["action"],
                            case["dom"])
            actual, path, lat = (v["decision"], v["path"],
                                 v.get("latency_s", 0.0))
        rows.append({
            "case": case["id"], "cat": case["cat"],
            "goal": redact_secrets(case["goal"]),
            "expected": case["expected"], "actual": actual,
            "correct": actual == case["expected"],
            "llm_called": getattr(llm, "calls", 0) > calls_before,
            "path": path, "latency_s": lat,
        })
    return rows, time.time() - t0


def run_llm(args):
    """Experiment 1 on the SAME case set (controlled comparison)."""
    if args.judge == "real":
        from models.model_factory import get_vlm
        print("Loading real 3B text judge...")
        real = get_vlm("qwen2.5-vl", model_path=args.model_path,
                       device=args.device, max_new_tokens=args.max_tokens)
    else:
        print("MODE=llm/scripted (plumbing check).\n")
        real = None
    rows = []
    t0 = time.time()
    for case in CASES:
        if real is not None:
            judge, path = real, "llm-judge"
        else:
            exp = case["expected"]
            judge = ScriptedJudge3B(
                {"decision": exp, "missing_field": "",
                 "question": "Q?" if exp == "ASK" else ""})
            path = "scripted"
        gate = IntentGate(text_llm=judge)
        consult, _, target = should_consult_gate(case["action"], None,
                                                 case["dom"])
        if not consult:
            actual, lat = "ALLOW", 0.0
            path = "bypass"
        else:
            from agent.hybrid_gate import HybridGate as _H
            target = find_action_target(case["action"], None, case["dom"])
            v = gate.validate_action_intent(
                case["goal"], case["history"], case["action"],
                build_dom_context(target, case["dom"]))
            actual, lat = v["decision"], v.get("latency_s", 0.0)
            path = "llm-judge"
        rows.append({"case": case["id"], "cat": case["cat"],
                     "goal": redact_secrets(case["goal"]),
                     "expected": case["expected"], "actual": actual,
                     "correct": actual == case["expected"],
                     "llm_called": consult and real is not None,
                     "path": path, "latency_s": lat})
    return rows, time.time() - t0


def report(rows, total_s, label, out_path=None):
    print(f"\n{label}: {len(rows)} cases")
    print(f"{'Case':22} | {'Cat':10} | {'Exp':5} | {'Act':5} | Result")
    print("-" * 66)
    for r in rows:
        flag = "OK" if r["correct"] else "WRONG"
        extra = " (7b)" if r.get("llm_called") else ""
        print(f"{r['case']:22} | {r['cat']:10} | {r['expected']:5} | "
              f"{r['actual']:5} | {flag}{extra}")
    print("-" * 66)
    tp = sum(1 for r in rows if r["expected"] == "ASK" and r["actual"] == "ASK")
    fp = sum(1 for r in rows if r["expected"] == "ALLOW" and r["actual"] == "ASK")
    fn = sum(1 for r in rows if r["expected"] == "ASK" and r["actual"] != "ASK")
    tn = len(rows) - tp - fp - fn
    n_ask = tp + fn
    n_allow = tn + fp
    lat = sorted(r.get("latency_s", 0.0) for r in rows)
    paths = [r.get("path", "") for r in rows]
    llm_n = sum(1 for r in rows if r.get("llm_called"))
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rep = {
        "total": len(rows),
        "confusion": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "hitl_recall": round(rec, 3),
        "hitl_precision": round(prec, 3),
        "false_ask": fp,
        "false_allow": fn,
        "false_ask_rate": round(fp / n_allow, 3) if n_allow else 0.0,
        "false_allow_rate": round(fn / n_ask, 3) if n_ask else 0.0,
        "recall_ci95": list(wilson(rec, tp + fn)),
        "precision_ci95": list(wilson(prec, tp + fp)),
        "mean_latency_s": round(sum(lat) / len(lat), 3) if lat else 0.0,
        "median_latency_s": round(statistics.median(lat), 3) if lat else 0.0,
        "p95_latency_s": round(
            statistics.quantiles(lat, n=100)[94], 3) if len(lat) >= 2 else 0.0,
        "llm_calls": llm_n,
        "pct_requiring_llm": round(llm_n / len(rows), 3),
        "deterministic_allow_pct": round(
            sum(1 for p in paths if p == "deterministic-allow") / len(rows), 3),
        "deterministic_ask_pct": round(
            sum(1 for p in paths if p == "deterministic-ask") / len(rows), 3),
        "bypass_pct": round(
            sum(1 for p in paths if p == "bypass") / len(rows), 3),
        "total_s": round(total_s, 1),
        "actions_per_s": round(len(rows) / total_s, 2) if total_s else 0.0,
        "rows": rows,
    }
    print(f"Confusion  TP={tp} FP={fp} / FN={fn} TN={tn}")
    print(f"Recall {rep['hitl_recall']} {rep['recall_ci95']} | "
          f"Precision {rep['hitl_precision']} {rep['precision_ci95']}")
    print(f"False ASK {fp} (rate {rep['false_ask_rate']}) | "
          f"False ALLOW {fn} (rate {rep['false_allow_rate']})")
    print(f"Latency mean {rep['mean_latency_s']}s median "
          f"{rep['median_latency_s']}s p95 {rep['p95_latency_s']}s | "
          f"LLM {llm_n}/{len(rows)} | det ALLOW/ASK "
          f"{rep['deterministic_allow_pct']}/{rep['deterministic_ask_pct']} "
          f"bypass {rep['bypass_pct']}")
    print(json.dumps(rep, indent=2))
    if out_path:
        out = Path(out_path)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "final_validation_report.json", "w") as f:
            json.dump(rep, f, indent=2)
        print(f"\nReport saved to {out / 'final_validation_report.json'}")
    return rep


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    if args.mode == "llm":
        rows, total = run_llm(args)
        report(rows, total, "Experiment 1 (llm gate, final set)", args.output)
    else:
        rows, total = run_hybrid(args)
        report(rows, total, "Experiment 2 (hybrid gate, final set)",
               args.output)


if __name__ == "__main__":
    main()
