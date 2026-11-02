#!/usr/bin/env python3
"""Evaluate the EXPERIMENTAL intent gates offline (no browser needed).

Modes:
  llm    = Experiment 1 (original 6 cases; scripted or real 3B judge).
  hybrid = Experiment 2 (22 cases across 10 categories; deterministic
           first, scripted or real qwen2.5:7b judge for AMBIGUOUS only).

Never stores secrets: goals/answers are redacted in reports.

Example:
  python scripts/evaluate_intent_gate.py --mode hybrid
  python scripts/evaluate_intent_gate.py --mode hybrid --judge7b real \\
      --output experiments/intent_gate/
"""

import logging  # stdlib first (src/ contains a logging/ package that shadows it)
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


def _el(i, tag, text="", typ="", role="", ph="", al=""):
    return {"id": i, "tag": tag, "type": typ, "role": role, "text": text,
            "ariaLabel": al, "placeholder": ph,
            "bbox": {"x": 10 + i * 5, "y": 10, "width": 100, "height": 30}}


def _batch_dom():
    return [_el(0, "select", "Course", al="Course"),
            _el(1, "select", "Batch", al="Batch"),
            {"id": 2, "tag": "a", "type": "", "role": "link", "text": "Home",
             "ariaLabel": "", "placeholder": "",
             "bbox": {"x": 10, "y": 100, "width": 60, "height": 20}}]


# -- Experiment 1 cases (preserved verbatim) ---------------------------------
EXP1_CASES = [
    {
        "name": "msc_missing_batch",
        "goal": "Show me all MSc students.",
        "history": [],
        "action": {"action_type": "click", "target_id": 1},
        "dom": _batch_dom(),
        "expected": "ASK",
        "scripted": {"decision": "ASK", "missing_field": "Batch",
                     "question": "Which batch would you like to view?"},
    },
    {
        "name": "msc_with_2023",
        "goal": "Show me all MSc students from the 2023 batch.",
        "history": [],
        "action": {"action_type": "click", "target_id": 1},
        "dom": _batch_dom(),
        "expected": "ALLOW",
        "scripted": {"decision": "ALLOW", "missing_field": "",
                     "question": ""},
    },
    {
        "name": "open_page_nav",
        "goal": "Open the student management page.",
        "history": [],
        "action": {"action_type": "click", "target_id": 2},
        "dom": _batch_dom(),
        "expected": "ALLOW",
        "scripted": {"decision": "ALLOW", "missing_field": "",
                     "question": ""},
    },
    {
        "name": "login_missing",
        "goal": "Log in to the portal.",
        "history": [],
        "action": {"action_type": "type", "target_id": 0, "value": "???"},
        "dom": [
            {"id": 0, "tag": "input", "type": "text", "role": "textbox",
             "text": "", "ariaLabel": "", "placeholder": "Username",
             "bbox": {"x": 1, "y": 1, "width": 10, "height": 10}},
            {"id": 1, "tag": "input", "type": "password", "role": "",
             "text": "", "ariaLabel": "", "placeholder": "Password",
             "bbox": {"x": 1, "y": 20, "width": 10, "height": 10}},
        ],
        "expected": "ASK",
        "scripted": {"decision": "ASK", "missing_field": "credentials",
                     "question": "Please provide username and password."},
    },
    {
        "name": "login_provided",
        "goal": "Log in as alice with password s3cret.",
        "history": [],
        "action": {"action_type": "type", "target_id": 0, "value": "alice"},
        "dom": [
            {"id": 0, "tag": "input", "type": "text", "role": "textbox",
             "text": "", "ariaLabel": "", "placeholder": "Username",
             "bbox": {"x": 1, "y": 1, "width": 10, "height": 10}},
            {"id": 1, "tag": "input", "type": "password", "role": "",
             "text": "", "ariaLabel": "", "placeholder": "Password",
             "bbox": {"x": 1, "y": 20, "width": 10, "height": 10}},
        ],
        "expected": "ALLOW",
        "scripted": {"decision": "ALLOW", "missing_field": "",
                     "question": ""},
    },
    {
        "name": "latest_batch_inferable",
        "goal": "Show students from the latest batch.",
        "history": [],
        "action": {"action_type": "click", "target_id": 3},
        "dom": [
            {"id": 0, "tag": "button", "type": "", "role": "",
             "text": "2021", "ariaLabel": "", "placeholder": "",
             "bbox": {"x": 10, "y": 10, "width": 60, "height": 30}},
            {"id": 1, "tag": "button", "type": "", "role": "",
             "text": "2022", "ariaLabel": "", "placeholder": "",
             "bbox": {"x": 80, "y": 10, "width": 60, "height": 30}},
            {"id": 2, "tag": "button", "type": "", "role": "",
             "text": "2023", "ariaLabel": "", "placeholder": "",
             "bbox": {"x": 150, "y": 10, "width": 60, "height": 30}},
            {"id": 3, "tag": "button", "type": "", "role": "",
             "text": "2024", "ariaLabel": "", "placeholder": "",
             "bbox": {"x": 220, "y": 10, "width": 60, "height": 30}},
        ],
        "expected": "ALLOW",
        "scripted": {"decision": "ALLOW", "missing_field": "",
                     "question": ""},
    },
]


# -- Experiment 2 cases (22 across 10 categories) -----------------------------
def _login_dom():
    return [
        {"id": 0, "tag": "input", "type": "text", "role": "textbox",
         "text": "", "ariaLabel": "", "placeholder": "Username",
         "bbox": {"x": 1, "y": 1, "width": 10, "height": 10}},
        {"id": 1, "tag": "input", "type": "password", "role": "",
         "text": "", "ariaLabel": "", "placeholder": "Password",
         "bbox": {"x": 1, "y": 20, "width": 10, "height": 10}},
    ]


HYBRID_CASES = [
    # EXPLICIT (deterministic ALLOW, 0 LLM)
    {"id": "exp_batch_explicit", "cat": "EXPLICIT",
     "goal": "Show me all MSc students from the 2023 batch.",
     "history": [], "action": {"action_type": "click", "target_id": 1},
     "dom": _batch_dom(), "field": "Batch", "expected": "ALLOW"},
    {"id": "exp_login_explicit", "cat": "LOGIN",
     "goal": "Login as alice with password s3cret.",
     "history": [], "action": {"action_type": "type", "target_id": 0,
                               "value": "alice"},
     "dom": _login_dom(), "field": "Username", "expected": "ALLOW"},
    {"id": "exp_search_explicit", "cat": "SEARCH",
     "goal": 'Search for "laptop".',
     "history": [], "action": {"action_type": "type", "target_id": 0,
                               "value": "laptop"},
     "dom": [_el(0, "input", "", typ="search", ph="Search products...")],
     "field": "Search", "expected": "ALLOW"},
    {"id": "exp_filter_explicit", "cat": "FILTER",
     "goal": "Show orders with status Shipped.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [_el(0, "select", "Status", al="Status"),
             _el(1, "button", "Shipped"), _el(2, "button", "Pending")],
     "field": "Status", "expected": "ALLOW"},
    {"id": "exp_date_explicit", "cat": "DATE",
     "goal": "Book a room for 2024-05-01.",
     "history": [], "action": {"action_type": "type", "target_id": 0,
                               "value": "2024-05-01"},
     "dom": [_el(0, "input", "", typ="date", ph="Check-in date")],
     "field": "Check-in date", "expected": "ALLOW"},
    {"id": "exp_option_named", "cat": "EXPLICIT",
     "goal": "Select the Blue color.",
     "history": [], "action": {"action_type": "click", "target_id": 1},
     "dom": [_el(0, "button", "Red"), _el(1, "button", "Blue"),
             _el(2, "button", "Green")],
     "field": "Blue", "expected": "ALLOW"},
    # MISSING (deterministic ASK, 0 LLM)
    {"id": "miss_batch", "cat": "MISSING",
     "goal": "Show me all MSc students.",
     "history": [], "action": {"action_type": "click", "target_id": 1},
     "dom": _batch_dom(), "field": "Batch", "expected": "ASK"},
    {"id": "miss_login", "cat": "LOGIN",
     "goal": "Log in to my account.",
     "history": [], "action": {"action_type": "type", "target_id": 0,
                               "value": "???"},
     "dom": _login_dom(), "field": "Username", "expected": "ASK"},
    {"id": "miss_address", "cat": "MISSING",
     "goal": "Check out and place my order.",
     "history": [], "action": {"action_type": "type", "target_id": 0,
                               "value": "???"},
     "dom": [_el(0, "input", "", ph="Shipping address"),
             _el(1, "button", "Place Order")],
     "field": "Shipping address", "expected": "ASK"},
    {"id": "miss_date", "cat": "DATE",
     "goal": "Search for a flight to New York.",
     "history": [], "action": {"action_type": "click", "target_id": 2},
     "dom": [_el(0, "input", "", ph="From"), _el(1, "input", "", ph="To"),
             _el(2, "input", "", typ="date", ph="Departure Date"),
             _el(3, "button", "Search Flights")],
     "field": "Departure Date", "expected": "ASK"},
    {"id": "miss_search_query", "cat": "SEARCH",
     "goal": "Search.",
     "history": [], "action": {"action_type": "type", "target_id": 0,
                               "value": "???"},
     "dom": [_el(0, "input", "", typ="search", ph="Search products...")],
     "field": "Search", "expected": "ASK"},
    {"id": "miss_filter", "cat": "FILTER",
     "goal": "Show my orders.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [_el(0, "select", "Status", al="Status")],
     "field": "Status", "expected": "ASK"},
    # IMPLICIT / AMBIGUOUS (7B judge)
    {"id": "impl_latest", "cat": "IMPLICIT",
     "goal": "Show students from the latest batch.",
     "history": [], "action": {"action_type": "click", "target_id": 3},
     "dom": [_el(0, "button", "2021"), _el(1, "button", "2022"),
             _el(2, "button", "2023"), _el(3, "button", "2024")],
     "field": "2024", "expected": "ALLOW", "scripted7b": "ALLOW"},
    {"id": "amb_usual_dept", "cat": "AMBIGUOUS",
     "goal": "Show students from my usual department.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [_el(0, "select", "Department", al="Department")],
     "field": "Department", "expected": "ASK", "scripted7b": "ASK"},
    {"id": "amb_color", "cat": "AMBIGUOUS",
     "goal": "Buy this shirt.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [_el(0, "button", "Red"), _el(1, "button", "Blue"),
             _el(2, "button", "Green")],
     "field": "Red", "expected": "ASK"},
    {"id": "amb_slot", "cat": "AMBIGUOUS",
     "goal": "Schedule the meeting.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [_el(0, "button", "Mon 9 AM"), _el(1, "button", "Tue 2 PM"),
             _el(2, "button", "Fri 11 AM")],
     "field": "Mon 9 AM", "expected": "ASK"},
    # IRRELEVANT / NAVIGATION (bypass ALLOW)
    {"id": "irr_nav", "cat": "IRRELEVANT_CONTROL",
     "goal": "Open the student management page.",
     "history": [], "action": {"action_type": "click", "target_id": 2},
     "dom": _batch_dom(), "field": "", "expected": "ALLOW"},
    {"id": "irr_single_button", "cat": "IRRELEVANT_CONTROL",
     "goal": "Open settings.",
     "history": [], "action": {"action_type": "click", "target_id": 0},
     "dom": [_el(0, "button", "Settings")],
     "field": "", "expected": "ALLOW"},
    {"id": "nav_home", "cat": "NAVIGATION",
     "goal": "Go back home.",
     "history": [], "action": {"action_type": "click", "target_id": 2},
     "dom": _batch_dom(), "field": "", "expected": "ALLOW"},
    # LOGIN via history + quoted batch
    {"id": "login_history", "cat": "LOGIN",
     "goal": "Log in.",
     "history": [{"step": 1, "action": "ask_user", "target_id": None,
                  "coordinates": None,
                  "reason": "Asked human: credentials? Answer: alice / s3cret",
                  "result": "success", "url": "http://t/",
                  "verification": True}],
     "action": {"action_type": "type", "target_id": 0, "value": "alice"},
     "dom": _login_dom(), "field": "Username", "expected": "ALLOW"},
    {"id": "exp_quoted_batch", "cat": "EXPLICIT",
     "goal": 'Show students in the "2024" batch.',
     "history": [], "action": {"action_type": "click", "target_id": 1},
     "dom": _batch_dom(), "field": "Batch", "expected": "ALLOW"},
    {"id": "search_form_date", "cat": "SEARCH",
     "goal": "Find flights to Boston.",
     "history": [], "action": {"action_type": "type", "target_id": 2,
                               "value": "???"},
     "dom": [_el(0, "input", "", ph="From"), _el(1, "input", "", ph="To"),
             _el(2, "input", "", typ="date", ph="Departure Date")],
     "field": "Departure Date", "expected": "ASK"},
]


class ScriptedJudge:
    def __init__(self, verdict):
        self._verdict = verdict

    def generate(self, image, prompt, history=None):
        return json.dumps(self._verdict)


class Scripted7B:
    """Per-case canned 7B verdict (plumbing only)."""

    def __init__(self, verdict="ALLOW"):
        self._verdict = verdict
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return self._verdict


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("llm", "hybrid"), default="hybrid")
    p.add_argument("--judge", choices=("scripted", "real"), default="scripted",
                   help="Exp1 3B judge backend.")
    p.add_argument("--judge7b", choices=("scripted", "real"),
                   default="scripted", help="Exp2 7B judge backend.")
    p.add_argument("--model-path", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--output", default=None)
    return p


def run_llm_mode(args):
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
    for case in EXP1_CASES:
        judge = real if real is not None else ScriptedJudge(case["scripted"])
        gate = IntentGate(text_llm=judge)
        consult, why, target = should_consult_gate(
            case["action"], None, case["dom"])
        if not consult:
            actual, lat, path = "ALLOW", 0.0, "bypass"
        else:
            target = find_action_target(case["action"], None, case["dom"])
            v = gate.validate_action_intent(
                case["goal"], case["history"], case["action"],
                build_dom_context(target, case["dom"]))
            actual, lat, path = (v["decision"], v.get("latency_s", 0.0),
                                 "llm-judge")
        rows.append({"case": case["name"], "expected": case["expected"],
                     "actual": actual, "correct": actual == case["expected"],
                     "latency_s": lat, "path": path})
    return rows, time.time() - t0


def run_hybrid_mode(args):
    from agent.hybrid_gate import HybridGate
    if args.judge7b == "real":
        print("Using real Ollama qwen2.5:7b judge...")
        real = OllamaTextJudge()
    else:
        print("MODE=hybrid/scripted (plumbing check).\n")
        real = None
    gate = HybridGate(llm=real)
    rows = []
    t0 = time.time()
    for case in HYBRID_CASES:
        if real is None:
            gate.llm = Scripted7B(case.get("scripted7b", "ALLOW"))
            gate.llm.calls = 0
        else:
            gate.llm = real
        calls_before = gate.llm_calls
        consult, why, target = hybrid_should_consult(
            case["action"], None, case["dom"])
        if not consult:
            actual, path, lat = "ALLOW", "bypass", 0.0
        else:
            v = gate.decide(case["goal"], case["history"],
                            case["action"], case["dom"])
            actual, path, lat = (v["decision"], v["path"],
                                 v.get("latency_s", 0.0))
        llm_called = gate.llm_calls > calls_before
        rows.append({
            "case": case["id"], "cat": case["cat"],
            "goal": redact_secrets(case["goal"]),
            "field": case.get("field", ""),
            "expected": case["expected"], "actual": actual,
            "correct": actual == case["expected"],
            "llm_called": llm_called, "path": path,
            "latency_s": lat,
        })
    total = time.time() - t0
    return rows, total


def report(rows, total_s, label, out_path=None):
    print(f"\n{label}: {len(rows)} cases")
    print(f"{'Case':22} | {'Cat':10} | {'Exp':5} | {'Act':5} | Result")
    print("-" * 66)
    for r in rows:
        name = r.get("case", r.get("id", "?"))
        flag = "OK" if r["correct"] else "WRONG"
        extra = " (7b)" if r.get("llm_called") else ""
        print(f"{name:22} | {r.get('cat', '-'):10} | {r['expected']:5} | "
              f"{r['actual']:5} | {flag}{extra}")
    print("-" * 66)
    tp = sum(1 for r in rows if r["expected"] == "ASK" and r["actual"] == "ASK")
    fp = sum(1 for r in rows if r["expected"] == "ALLOW" and r["actual"] == "ASK")
    fn = sum(1 for r in rows if r["expected"] == "ASK" and r["actual"] != "ASK")
    n_ask = sum(1 for r in rows if r["expected"] == "ASK")
    n_allow = len(rows) - n_ask
    lat = [r.get("latency_s", 0.0) for r in rows]
    llm_n = sum(1 for r in rows if r.get("llm_called"))
    det_allow = sum(1 for r in rows if r.get("path") == "deterministic-allow")
    det_ask = sum(1 for r in rows if r.get("path") == "deterministic-ask")
    rep = {
        "total": len(rows),
        "correct_allow": n_allow - fp,
        "correct_ask": tp,
        "false_allow": fn,
        "false_ask": fp,
        "false_ask_rate": round(fp / n_allow, 3) if n_allow else 0.0,
        "false_allow_rate": round(fn / n_ask, 3) if n_ask else 0.0,
        "hitl_recall": round(tp / (tp + fn), 3) if (tp + fn) else 0.0,
        "hitl_precision": round(tp / (tp + fp), 3) if (tp + fp) else 0.0,
        "avg_gate_latency_s": round(sum(lat) / len(lat), 2) if lat else 0.0,
        "median_gate_latency_s": round(statistics.median(lat), 2) if lat else 0.0,
        "qwen7b_calls": llm_n,
        "pct_requiring_7b": round(llm_n / len(rows), 3),
        "deterministic_allow_pct": round(det_allow / len(rows), 3),
        "deterministic_ask_pct": round(det_ask / len(rows), 3),
        "total_s": round(total_s, 1),
        "rows": rows,
    }
    print(f"Total: {rep['total']} | Correct ALLOW: {rep['correct_allow']}/{n_allow} "
          f"| Correct ASK: {tp}/{n_ask} | False ALLOW: {fn} | False ASK: {fp}")
    print(f"HITL recall: {rep['hitl_recall']} | precision: {rep['hitl_precision']} | "
          f"false-ASK rate: {rep['false_ask_rate']} | false-ALLOW rate: {rep['false_allow_rate']}")
    print(f"Avg latency: {rep['avg_gate_latency_s']}s | median: {rep['median_gate_latency_s']}s | "
          f"7B calls: {llm_n} ({rep['pct_requiring_7b']*100:.0f}%) | "
          f"det-ALLOW {rep['deterministic_allow_pct']*100:.0f}% / det-ASK {rep['deterministic_ask_pct']*100:.0f}%")
    print(json.dumps(rep, indent=2))
    if out_path:
        out = Path(out_path)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "intent_gate_report.json", "w") as f:
            json.dump(rep, f, indent=2)
        print(f"\nReport saved to {out / 'intent_gate_report.json'}")
    return rep


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    if args.mode == "llm":
        rows, total = run_llm_mode(args)
        report(rows, total, "Experiment 1 (llm gate)", args.output)
    else:
        rows, total = run_hybrid_mode(args)
        # Per-case latency for real runs is inside decide(); re-time here
        # only if latencies are zero (scripted mode).
        report(rows, total, "Experiment 2 (hybrid gate)", args.output)


if __name__ == "__main__":
    main()
