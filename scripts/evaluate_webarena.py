#!/usr/bin/env python3
"""Evaluate the agent on WebArena-style tasks.

Grounding is selected with ONE high-level flag:

  --method {coordinate,som,som_dom,cascade_som,cascade_som_dom}

Verification/recovery stay independent: --goal-mode, --no-verify, --no-recovery.
NOTE: in this script --no-verify disables the GoalVerifier (historical
behavior, preserved); use --goal-mode off for the same effect explicitly.

Example (smoke test on local test page):
  python scripts/evaluate_webarena.py --tasks data/webarena/smoke_tasks.json \\
      --method coordinate --vlm mock --output experiments/webarena_smoke/
"""

import logging  # stdlib first
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser.browser_manager import BrowserManager
from browser.observer import Observer
from grounding.methods import (resolve_method, build_proposer, build_grounder,
                               add_method_argument)
from actions.action_executor import ActionExecutor
from agent.controller import AgentController
from agent.memory import AgentMemory
from agent.recovery import RecoveryManager
from verification.verifier import StateVerifier
from verification.goal_verifier import GoalVerifier
from models.model_factory import get_vlm
from evaluation.webarena_env import WebArenaEnvironment, load_tasks
from evaluation.webarena_eval import WebArenaEvaluator

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "trajectory_logger",
    str(Path(__file__).resolve().parents[1] / "src" / "logging" / "trajectory_logger.py"))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TrajectoryLogger = _mod.TrajectoryLogger

GOAL_MODES = ("heuristic", "vlm", "both", "off")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True)
    p.add_argument("--base-url", default="")
    # Default coordinate preserves the historical default.
    add_method_argument(p, default="coordinate")
    p.add_argument("--vlm", default="mock")
    p.add_argument("--model-path", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--mock-responses", default=None)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--headless", dest="headless", action="store_true",
                   default=True)
    p.add_argument("--no-headless", dest="headless", action="store_false",
                   help="Run headed (debugging; needs display/Xvfb)")
    p.add_argument("--no-trajectories", action="store_true",
                   help="Disable per-task trajectory PNG/JSON logging")
    p.add_argument("--no-verify", action="store_true",
                   help="Disable GoalVerifier (historical behavior, preserved). "
                        "Prefer --goal-mode off.")
    p.add_argument("--goal-mode", choices=GOAL_MODES, default="heuristic",
                   help="GoalVerifier mode; 'off' disables goal-achievement early stop")
    p.add_argument("--no-recovery", action="store_true",
                   help="Disable recovery (zero all recovery budgets)")
    p.add_argument("--resume", action="store_true",
                   help="Skip tasks already in webarena_checkpoint.jsonl and "
                        "merge their records into the final report.")
    p.add_argument("--output", default="./experiments/webarena/")
    return p


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    method = resolve_method(args.method)
    goal_enabled = (args.goal_mode != "off") and (not args.no_verify)

    tasks = load_tasks(Path(args.tasks))
    browser = BrowserManager(headless=args.headless)
    try:
        browser.launch()
        env = WebArenaEnvironment(browser, base_url=args.base_url)

        def make_controller(task):
            if task.get("mock_responses"):
                responses = task["mock_responses"]
            elif args.mock_responses:
                responses = json.load(open(args.mock_responses))
            else:
                responses = [json.dumps({"action": "done", "reason": "smoke"})]
            vlm = get_vlm(args.vlm, responses=responses) if args.vlm == "mock" \
                else get_vlm(args.vlm, model_path=args.model_path,
                             device=args.device, max_new_tokens=args.max_tokens)
            grounder = build_grounder(method, vlm)
            if args.no_trajectories:
                traj = None
            else:
                traj = TrajectoryLogger(
                    Path(args.output) / "trajectories",
                    run_id=f"{task.get('task_id', 'task')}_{args.method}")
            if goal_enabled:
                goal_verifier = GoalVerifier(
                    mode=args.goal_mode,
                    vlm=vlm if args.goal_mode in ("vlm", "both") else None)
            else:
                goal_verifier = None
            if args.no_recovery:
                recovery = RecoveryManager(max_re_ground_attempts=0,
                                           max_recovery_attempts=0,
                                           max_execution_retries=0)
            else:
                recovery = RecoveryManager()
            return AgentController(
                browser, Observer(browser, proposer=build_proposer(method)),
                grounder, ActionExecutor(browser),
                verifier=StateVerifier(), memory=AgentMemory(),
                recovery=recovery, trajectory_logger=traj,
                max_steps=args.max_steps,
                task=task.get("goal", ""),
                goal_verifier=goal_verifier,
                enable_goal_check=goal_enabled)

        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "run_config.json", "w") as f:
            json.dump({
                "tasks": args.tasks,
                "method": method,
                "goal_mode": args.goal_mode,
                "goal_verification_enabled": goal_enabled,
                "recovery_enabled": not args.no_recovery,
                "vlm": args.vlm,
                "max_steps": args.max_steps,
                "resume": args.resume,
            }, f, indent=2)
        evaluator = WebArenaEvaluator(env, make_controller, output_dir=out_dir,
                                      resume=args.resume)
        summary = evaluator.evaluate(tasks)
        print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))
    finally:
        browser.close()


if __name__ == "__main__":
    main()
