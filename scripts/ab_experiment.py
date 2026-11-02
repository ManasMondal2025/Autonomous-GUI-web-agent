#!/usr/bin/env python3
"""Controlled A/B: fused Qwen vs Qwen-planner + GUI-Actor-grounder.

Same 12 tasks, same fresh browser state per task, same Playwright
executor, same viewport, same max_steps, greedy (temperature-free)
decoding on both VLMs. Systems differ ONLY in the perception/decision
path (fused single call vs planner JSON + pointer grounding).

Systems:
  fused : CoordinateGrounder(Qwen).ground -> to_action -> executor
          (identical to --method coordinate behaviour).
  split : Qwen planner JSON {action,target,instruction[,text|direction]}
          (NO coordinates requested) -> GUI-Actor grounds instruction ->
          executor. Planner restricted to executor-supported actions
          (click/type/scroll/wait/done/fail; keypress excluded because
          ActionExecutor cannot execute it).

Mock doubles (--qwen mock / --grounder mock) validate the harness
without weights; real runs use --qwen qwen2.5-vl --grounder gui-actor.

Writes: <output>/ab_report.json + ab_report.md + ab_records.jsonl.
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

VIEWPORT = (1280, 720)
MAX_STEPS = 3
SETTLE_S = 1.5  # post-load settle (dynamic content); identical both systems

PLANNER_PROMPT = (
    "You are the planner of a GUI agent. Look at the screenshot. "
    "User goal: {goal}. Decide the single next action. Reply with ONLY "
    'a JSON object: {{"action": "click"|"type"|"scroll"|"wait"|"done"|"fail", '
    '"target": "<short noun phrase naming the UI element>", '
    '"instruction": "<concise location-first imperative that uniquely '
    'identifies the target by its visible text AND where it is on screen, '
    'e.g. \'Click inside the search box at the top of the page\'\">, '
    '"text": "<text to type, for type only>", '
    '"direction": "<up|down|left|right, for scroll only>"}}. '
    'The "instruction" key is REQUIRED on every action (the grounder '
    'locates purely from it, so phrase locations, not actions). '
    "Do NOT output coordinates. No other text.")

SCROLL_D = {"down": (0, 500), "up": (0, -500), "left": (-500, 0),
            "right": (500, 0)}

log = logging.getLogger("ab_experiment")


class MockPlannerVLM:
    """Harness test double: correct planner JSON from task GT metadata."""

    name = "mock-planner"

    def __init__(self):
        self.current = {}

    def generate(self, image, prompt, history=None):
        t = self.current
        d = {"action": t.get("expected_action", "click"),
             "target": " ".join(t.get("expected_target_keywords", ["thing"])),
             "instruction": f"{t.get('expected_action', 'click')} "
                            f"{' '.join(t.get('expected_target_keywords', ['thing']))}"}
        if t.get("type_value"):
            d["text"] = t["type_value"]
        return json.dumps(d)


class MockGrounderVLM:
    """Harness test double: GUI-Actor-shaped JSON at the GT center."""

    name = "mock-grounder"

    def __init__(self):
        self.current = {}

    def generate(self, image, prompt, history=None):
        gt = self.current.get("gt", [100, 100, 200, 200])
        cx, cy = (gt[0] + gt[2]) / 2.0, (gt[1] + gt[3]) / 2.0
        return json.dumps({"action": "click", "x": cx, "y": cy})


class MockFusedVLM:
    """Harness test double: CoordinateGrounder-shaped JSON at GT center."""

    name = "mock-fused"

    def __init__(self):
        self.current = {}

    def generate(self, image, prompt, history=None):
        t = self.current
        gt = t.get("gt", [100, 100, 200, 200])
        cx, cy = (gt[0] + gt[2]) / 2.0, (gt[1] + gt[3]) / 2.0
        d = {"action": t.get("expected_action", "click"), "x": cx, "y": cy,
             "reason": "mock"}
        if t.get("type_value"):
            d["value"] = t["type_value"]
        return json.dumps(d)


def parse_planner_json(text):
    """Parse planner JSON. Returns (plan_dict_or_None, err_or_None).

    The dict is returned even on schema failure (with the specific err)
    so the caller can retry or repair from partial fields. Error codes:
    no_json_object, json_parse_error, bad_action, missing_instruction.
    """
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None, "no_json_object"
    try:
        plan = json.loads(m.group(0))
    except Exception:
        return None, "json_parse_error"
    if not isinstance(plan, dict):
        return None, "bad_schema"
    if plan.get("action") not in (
            "click", "type", "scroll", "wait", "done", "fail"):
        return plan, "bad_action"
    if not plan.get("instruction"):
        return plan, "missing_instruction"
    return plan, None


def repair_instruction(plan):
    """Deterministic repair for a parsed plan missing only `instruction`.

    Builds it from the planner's own action/target/text (location-first).
    Returns the instruction string, or None if unrepairable."""
    if not isinstance(plan, dict):
        return None
    act = plan.get("action")
    tgt = (plan.get("target") or "").strip()
    if act == "type" and plan.get("text"):
        return f"Type '{plan['text']}' into {tgt or 'this element'}"
    if act == "scroll":
        return f"Scroll {plan.get('direction') or 'down'} to find {tgt}"
    if act in ("click", "wait", "done", "fail") and tgt:
        return f"{act.capitalize()} the {tgt}"
    return None


def planner_correct(plan, task):
    if plan.get("action") != task["expected_action"]:
        return False
    hay = f"{plan.get('target', '')} {plan.get('instruction', '')}".lower()
    return all(kw.lower() in hay
               for kw in task.get("expected_target_keywords", []))


def point_in_gt(x, y, gt):
    return gt[0] <= x <= gt[2] and gt[1] <= y <= gt[3]


# --------------------------------------------------------------------------
# Shared harness: browser, executor, checkers, step loop
# --------------------------------------------------------------------------

def resolve_url(rel: str) -> str:
    if rel.startswith("file://") and not rel.startswith("file:///"):
        return f"file://{(Path.cwd() / rel[len('file://'):]).resolve()}"
    return rel


def make_action(action_type, x=None, y=None, value=None, direction=None,
                reason=""):
    from actions.action_schema import Action
    dx = dy = None
    if action_type == "scroll" and direction in SCROLL_D:
        dx, dy = SCROLL_D[direction]
    return Action(action_type=action_type, x=x, y=y, value=value,
                  dx=dx, dy=dy, reason=reason or "")


def run_task(system, task, browser, executor, env, vlms, max_steps,
             vopts=None):
    """One task, one system. Returns (task_success, step_records)."""
    from PIL import Image
    import io

    browser.new_page()
    browser.navigate(resolve_url(task["start_url"]))
    browser.get_page().wait_for_timeout(int(SETTLE_S * 1000))
    env.current_task = task
    records, success, failure, lat = [], False, None, {}
    for step in range(1, max_steps + 1):
        rec = {"task_id": task["task_id"], "architecture": system,
               "step_number": step,
               "expected_action": task["expected_action"]}
        t0 = time.perf_counter()
        try:
            shot = browser.screenshot()
            image = Image.open(io.BytesIO(shot)).convert("RGB")
            v_rec = {"verifier_on": False, "scores": [],
                     "accepted_rank": None, "rejected": 0}
            t_verify = 0.0
            if system == "fused":
                pred = vlms["fused"].ground(
                    image, task["goal"], [], history=None)
                try:
                    act = vlms["fused"].to_action(pred)
                except Exception:
                    act = pred if isinstance(pred, dict) else {}
                # to_action() names it action_type; accept both shapes.
                action = act.get("action_type", act.get("action"))
                instr = task["goal"]
                qact, qtarg, qinstr = action, "(fused: goal text)", instr
                plan_ok = (action == task["expected_action"])
                lat_p, lat_g = 0.0, time.perf_counter() - t0
                x, y = act.get("x"), act.get("y")
                value = act.get("value")
            else:
                t1 = time.perf_counter()
                base_prompt = PLANNER_PROMPT.format(goal=task["goal"])
                plan_raw = vlms["planner"].generate(image, base_prompt)
                plan, err = parse_planner_json(plan_raw)
                retries, repaired = 0, False
                # Structural enforcement: one feedback retry on a missing
                # instruction key (Qwen-3B's type-task prior omits it), then
                # deterministic repair from the planner's own fields.
                while err == "missing_instruction" and retries < 1 \
                        and isinstance(plan, dict):
                    retries += 1
                    plan_raw = vlms["planner"].generate(
                        image, base_prompt + "\nYour previous reply "
                        "violated the schema: the REQUIRED \"instruction\" "
                        "key was missing. Reply with ONLY the corrected "
                        "JSON object.")
                    plan, err = parse_planner_json(plan_raw)
                if err == "missing_instruction" and isinstance(plan, dict):
                    synth = repair_instruction(plan)
                    if synth:
                        plan["instruction"] = synth
                        repaired, err = True, None
                lat_p = time.perf_counter() - t1
                rec.update({"planner_retries": retries,
                            "instruction_synthesized": repaired})
                if err:
                    rec.update({"qwen_action": None, "qwen_target": None,
                                "qwen_instruction": plan_raw[:200],
                                "planner_correct": False,
                                "grounder_coordinate": None,
                                "grounder_correct": False,
                                "execution_success": False,
                                "latency": {"planner": round(lat_p, 2)},
                                "failure_type": "PLANNER_ERROR",
                                "note": err})
                    records.append(rec)
                    failure, lat = "PLANNER_ERROR", {"planner": lat_p}
                    break
                qact, qtarg = plan.get("action"), plan.get("target", "")
                qinstr = plan.get("instruction", "")
                plan_ok = planner_correct(plan, task)
                t2 = time.perf_counter()
                v_rec = {"verifier_on": False, "scores": [],
                         "accepted_rank": None, "rejected": 0}
                t_verify = 0.0
                x = y = None
                gd = {}
                use_verifier = bool(
                    vopts and vopts.get("on")
                    and hasattr(vlms["grounder"], "generate_candidates"))
                v_rec["verifier_on"] = use_verifier
                if use_verifier:
                    # Gate: score ranked candidates, accept the first at or
                    # above threshold; a vetoed point is never executed.
                    cands = vlms["grounder"].generate_candidates(
                        image, qinstr, vopts.get("top_k", 3)) or []
                    verifier = vopts["verifier"]
                    thr = vopts.get("threshold", 0.8)
                    for rank, c in enumerate(cands):
                        t3 = time.perf_counter()
                        try:
                            s = verifier.verify(image, qinstr, c["x"],
                                                c["y"])
                        except Exception as ex:
                            log.warning("verifier failed: %s", ex)
                            s = None
                        t_verify += time.perf_counter() - t3
                        v_rec["scores"].append(
                            round(s, 4) if s is not None else None)
                        if s is not None and s >= thr:
                            x, y = c["x"], c["y"]
                            gd = {"action": qact, "x": x, "y": y}
                            v_rec["accepted_rank"] = rank
                            break
                    else:
                        v_rec["rejected"] = len(cands)
                else:
                    g_raw = vlms["grounder"].generate(image, qinstr)
                    try:
                        gd = json.loads(g_raw) \
                            if isinstance(g_raw, str) else dict(g_raw)
                    except Exception:
                        gd = {}
                    x, y = gd.get("x"), gd.get("y")
                lat_g = time.perf_counter() - t2
                value = plan.get("text")
                direction = plan.get("direction")
                action, instr = qact, qinstr
            g_ok = isinstance(x, (int, float)) and isinstance(
                y, (int, float)) and point_in_gt(x, y, task["gt"])
            exec_ok = False
            if action in ("click", "type") and (
                    not isinstance(x, (int, float))
                    or not isinstance(y, (int, float))):
                # Gate: never execute a vetoed/missing point (a click at
                # None must not become a click at origin). Terminal and
                # scroll/wait actions need no coordinates.
                log.info("Skipping execution: no coordinates for %s.",
                         action)
            else:
                try:
                    act_obj = make_action(
                        action, x=x, y=y,
                        value=value if action == "type" else None,
                        direction=direction if action == "scroll" else None,
                        reason=(instr or "")[:200])
                    res = executor.execute(act_obj)
                    exec_ok = bool(res.get("success"))
                except Exception as e:
                    exec_ok = False
                    log.warning("Execution failed %s step %d: %s",
                                task["task_id"], step, e)
            check_ok, check_reason = True, "no-state-check"
            if task.get("check"):
                try:
                    cr = env.check_success(task)
                    check_ok, check_reason = bool(cr.get("success")), \
                        cr.get("reason", "")
                except Exception as e:
                    check_ok, check_reason = False, f"check error: {e}"
            if not plan_ok:
                failure = "PLANNER_ERROR"
            elif not g_ok:
                failure = "GROUNDING_ERROR"
            elif not exec_ok:
                failure = "EXECUTION_ERROR"
            elif not check_ok:
                failure = "GROUNDING_ERROR"
                check_reason += " (gates passed but state unchanged)"
            else:
                failure = None
            rec.update({
                "qwen_action": qact, "qwen_target": qtarg,
                "qwen_instruction": qinstr,
                "planner_correct": bool(plan_ok),
                "grounder_coordinate": [x, y]
                if g_ok or isinstance(x, (int, float)) else None,
                "grounder_correct": bool(g_ok),
                "verifier_on": v_rec["verifier_on"] if system == "split"
                else False,
                "verifier_scores": v_rec["scores"],
                "verifier_accepted_rank": v_rec["accepted_rank"],
                "verifier_rejected": v_rec["rejected"],
                "execution_success": bool(exec_ok),
                "check_success": bool(check_ok),
                "check_reason": check_reason,
                "latency": {"planner": round(lat_p, 2),
                            "grounder": round(lat_g, 2),
                            "verifier": round(t_verify, 2),
                            "executor": round(time.perf_counter() - t0
                                              - lat_p - lat_g, 2)},
                "failure_type": failure})
            records.append(rec)
            lat = {"planner": lat_p, "grounder": lat_g}
            if failure is None or action in ("done", "fail"):
                break
        except Exception as e:
            log.error("ENVIRONMENT_ERROR %s step %d: %s",
                      task["task_id"], step, e)
            records.append({"task_id": task["task_id"],
                            "architecture": system, "step_number": step,
                            "failure_type": "ENVIRONMENT_ERROR",
                            "note": f"{type(e).__name__}: {e}",
                            "latency": {}})
            failure = "ENVIRONMENT_ERROR"
            break
    task_success = failure is None
    return task_success, records


def build_report(all_records, systems, lat_totals):
    import collections
    per_task, fails = {}, collections.Counter()
    for arch in systems:
        recs = [r for r in all_records if r["architecture"] == arch]
        by_task = {}
        for r in recs:
            by_task.setdefault(r["task_id"], []).append(r)
        pt = {}
        for tid, rs in sorted(by_task.items()):
            last = rs[-1]
            ok = last.get("failure_type") is None
            pt[tid] = {"success": ok, "steps": len(rs),
                       "failure": last.get("failure_type"),
                       "first_action_ok": rs[0].get("qwen_action")
                       == rs[0].get("expected_action"),
                       "first_planner_ok": bool(
                           rs[0].get("planner_correct")),
                       "first_grounder_ok": bool(
                           rs[0].get("grounder_correct")),
                       "latency": round(sum(
                           sum(v for v in r.get("latency", {}).values()
                               if isinstance(v, (int, float)))
                           for r in rs), 2)}
            if not ok:
                fails[(arch, last.get("failure_type"))] += 1
        per_task[arch] = pt

    def rate(arch, key):
        ts = per_task[arch]
        return round(sum(1 for v in ts.values() if v[key]) / max(1, len(ts))
                     * 100.0, 1)

    overall = {}
    for arch in systems:
        ts = per_task[arch]
        lat = lat_totals.get(arch, 0.0)
        overall[arch] = {
            "task_success": f"{sum(1 for v in ts.values() if v['success'])}"
                            f"/{len(ts)}",
            "action_success_pct": rate(arch, "first_action_ok"),
            "planner_success_pct": rate(arch, "first_planner_ok")
            if arch == "split" else None,
            "grounding_success_pct": rate(arch, "first_grounder_ok"),
            "avg_steps": round(sum(v["steps"] for v in ts.values())
                               / max(1, len(ts)), 2),
            "avg_latency": round(lat / max(1, len(ts)), 2)}
    # instruction analysis (split only): length buckets vs grounding hit
    instr_stats = []
    for tid, rs in sorted(per_task.get("split", {}).items()):
        pass
    split_rows = [r for r in all_records if r["architecture"] == "split"
                  and r.get("step_number") == 1]
    for r in split_rows:
        ins = r.get("qwen_instruction") or ""
        instr_stats.append({"task_id": r["task_id"],
                            "words": len(str(ins).split()),
                            "grounder_correct": bool(
                                r.get("grounder_correct"))})
    ver_rows = [r for r in all_records if r.get("verifier_on")]
    verifier_stats = None
    if ver_rows:
        acc = sum(1 for r in ver_rows if r.get("verifier_accepted_rank") == 0)
        rej = sum(1 for r in ver_rows if r.get("verifier_accepted_rank") is None
                  and r.get("verifier_scores"))
        verifier_stats = {
            "judged_steps": len(ver_rows),
            "accepted_top1": acc,
            "all_rejected": rej,
            "mean_attempts": round(sum(
                len(r.get("verifier_scores") or []) for r in ver_rows)
                / max(1, len(ver_rows)), 2),
        }
    return {"overall": overall, "per_task": per_task,
            "failures": {f"{a}:{f_}": c for (a, f_), c in fails.items()},
            "instruction_stats": instr_stats,
            "verifier": verifier_stats}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Controlled A/B: fused Qwen "
                                 "vs Qwen-planner + GUI-Actor grounder.")
    ap.add_argument("--tasks", default="data/ab_tasks.json")
    ap.add_argument("--systems", default="both",
                    choices=["fused", "split", "both"])
    ap.add_argument("--qwen", default="mock",
                    help="mock | qwen2.5-vl (planner, and fused VLM)")
    ap.add_argument("--qwen-path", default=None)
    ap.add_argument("--grounder", default="mock",
                    help="mock | gui-actor (split grounder)")
    ap.add_argument("--guiactor-path", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--output", default="./experiments/ab_out")
    ap.add_argument("--viewport", default="1280x720")
    ap.add_argument("--verifier", default="off", choices=["off", "on"],
                    help="split only: gate GUI-Actor candidates with the "
                         "GUI-Actor verifier (accept first at threshold, "
                         "else fail grounding). Default off (behavior "
                         "unchanged).")
    ap.add_argument("--verifier-path", default=None)
    ap.add_argument("--verifier-topk", type=int, default=3)
    ap.add_argument("--verify-threshold", type=float, default=0.8)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    vw, vh = (int(v) for v in args.viewport.lower().split("x"))
    systems = ["fused", "split"] if args.systems == "both" \
        else [args.systems]

    from models.model_factory import get_vlm
    from browser.browser_manager import BrowserManager
    from actions.action_executor import ActionExecutor
    from evaluation.webarena_env import WebArenaEnvironment
    from grounding.coordinate_grounder import CoordinateGrounder

    def _vlm(kind, path):
        if kind == "mock":
            return None  # replaced per-task below
        if kind == "gui-actor":
            return get_vlm("gui-actor", model_path=path, device=args.device)
        return get_vlm(kind, model_path=path, device=args.device)

    vlms_real = {"qwen": None if args.qwen == "mock" else _vlm(
                     args.qwen, args.qwen_path),
                 "grounder": None if args.grounder == "mock" else _vlm(
                     args.grounder, args.guiactor_path)}
    if vlms_real["qwen"] is None:
        vlms_real["qwen"] = MockFusedVLM()
        vlms_real["planner"] = MockPlannerVLM()
    else:
        vlms_real["planner"] = vlms_real["qwen"]
    if vlms_real["grounder"] is None:
        vlms_real["grounder"] = MockGrounderVLM()
    # In mock mode vlms_real["qwen"] IS the MockFusedVLM whose .current the
    # loop sets per task; reuse the same object (a fresh one would never
    # see the task and would ground the default box).
    fused_grounder = CoordinateGrounder(vlm=vlms_real["qwen"])

    # Verifier gate (split only; default off = behavior unchanged). Built
    # here so weights load once, not per task.
    verifier = None
    if args.verifier == "on":
        if not args.verifier_path:
            log.error("--verifier on requires --verifier-path.")
            return 2
        from models.gui_actor_verifier import GUIActorVerifier
        verifier = GUIActorVerifier(model_path=args.verifier_path,
                                    device=args.device)
        log.info("Verifier gate ON (top_k=%d, threshold=%.2f).",
                 args.verifier_topk, args.verify_threshold)
    vopts = {"on": verifier is not None, "verifier": verifier,
             "top_k": args.verifier_topk,
             "threshold": args.verify_threshold} \
        if verifier is not None else None

    tasks = json.load(open(args.tasks))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_records, lat_totals = [], {}
    browser = BrowserManager(headless=True, viewport_width=vw,
                             viewport_height=vh)
    browser.launch()
    try:
        for system in systems:
            env = WebArenaEnvironment(browser)
            executor = ActionExecutor(browser, viewport_width=vw,
                                      viewport_height=vh)
            lat_totals[system] = 0.0
            for t in tasks:
                for m in ("planner", "grounder", "qwen"):
                    mock = vlms_real.get(m)
                    if hasattr(mock, "current"):
                        mock.current = t
                if system == "fused":
                    per_task_vlms = {"fused": fused_grounder
                                     if args.qwen != "mock"
                                     else _FusedWithMock(vlms_real["qwen"])}
                else:
                    per_task_vlms = {"planner": vlms_real["planner"],
                                     "grounder": vlms_real["grounder"]}
                ok, recs = run_task(system, t, browser, executor, env,
                                    per_task_vlms, args.max_steps,
                                    vopts=vopts)
                lat_totals[system] += sum(
                    sum(v for v in r.get("latency", {}).values()
                        if isinstance(v, (int, float))) for r in recs)
                all_records.extend(recs)
                log.info("%s %s success=%s steps=%d", system,
                         t["task_id"], ok, len(recs))
    finally:
        browser.close()

    rep = build_report(all_records, systems, lat_totals)
    (out_dir / "ab_report.json").write_text(json.dumps(rep, indent=2))
    with open(out_dir / "ab_records.jsonl", "w") as f:
        for r in all_records:
            f.write(json.dumps(r, default=str) + "\n")
    (out_dir / "ab_report.md").write_text(render_markdown(rep, systems))
    print(json.dumps(rep["overall"], indent=2))
    return 0


class _FusedWithMock:
    """Adapts a MockFusedVLM (generate-only) to the grounder interface
    (ground + to_action) for harness validation without weights."""

    def __init__(self, mock):
        self._mock = mock

    def ground(self, image, instruction, dom_elements=None, history=None):
        import json as _json
        d = _json.loads(self._mock.generate(image, instruction, history))
        return {"action": d.get("action", "click"), "x": d.get("x"),
                "y": d.get("y"), "value": d.get("value"),
                "reason": "mock"}

    def to_action(self, pred):
        return {"action_type": pred.get("action", "click"),
                "x": pred.get("x"), "y": pred.get("y"),
                "value": pred.get("value"), "reason": pred.get("reason", "")}


def render_markdown(rep, systems):
    L = ["# A/B: fused Qwen vs Qwen-planner + GUI-Actor grounder", "",
         "## Overall"]
    for arch in systems:
        o = rep["overall"][arch]
        L.append(f"### {arch}")
        for k, v in o.items():
            L.append(f"- {k}: {v}")
        L.append("")
    L.append("## Per-task")
    L.append("| Task | " + " | ".join(systems) + " | Failure |")
    L.append("| --- | " + " | ".join(["---"] * len(systems)) + " | --- |")
    tids = sorted({t for arch in systems for t in rep["per_task"][arch]})
    for tid in tids:
        cells = [("OK" if rep["per_task"][a].get(tid, {}).get("success")
                  else "FAIL") for a in systems]
        fails = "; ".join(
            f"{a}={rep['per_task'][a].get(tid, {}).get('failure')}"
            for a in systems
            if not rep["per_task"][a].get(tid, {}).get("success"))
        L.append(f"| {tid} | " + " | ".join(cells) + f" | {fails} |")
    L += ["", "## Failures",
          *[f"- {k}: {v}" for k, v in sorted(rep["failures"].items())],
          "", "## Instruction stats (split, first steps)",
          *[f"- {r['task_id']}: {r['words']} words -> "
            f"grounder_correct={r['grounder_correct']}"
            for r in rep["instruction_stats"]],
          "", "## Conclusion"]
    fo, so = rep["overall"].get("fused"), rep["overall"].get("split")
    if fo and so:
        fs = int(fo["task_success"].split("/")[0])
        ss = int(so["task_success"].split("/")[0])
        if ss - fs >= 2:
            L.append("- Evidence favors the split architecture "
                     f"({ss} vs {fs} tasks).")
        elif fs - ss >= 2:
            L.append("- Evidence favors fused Qwen "
                     f"({fs} vs {ss} tasks).")
        else:
            L.append(f"- Inconclusive on task success ({fs} vs {ss}; "
                     f"N={fo['task_success'].split('/')[1]}, within noise).")
    L.append("- See failure/instruction sections for error origins.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()

