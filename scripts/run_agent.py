#!/usr/bin/env python3
"""Run the GUI agent end-to-end.

Grounding is selected with ONE high-level flag:

  --method {coordinate,som,som_dom,cascade_som,cascade_som_dom}

Verification/recovery stay independent: --goal-mode, --no-verify, --no-recovery.

Examples:
  python scripts/run_agent.py --task "Type laptop in search then finish" \\
      --url "file://test_pages/test_page.html" --method coordinate --vlm mock
  python scripts/run_agent.py --task "Click the search button" \\
      --url "file://test_pages/test_page.html" --method som_dom --vlm mock
  python scripts/run_agent.py --task "Search minishop for laptop" \\
      --url "file://test_pages/minishop/index.html" --method coordinate \\
      --vlm mock --long-horizon --planning-horizon 3
"""

import logging  # noqa: F401  (import stdlib logging BEFORE src hits sys.path)
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser.browser_manager import BrowserManager
from browser.observer import Observer
from grounding.methods import resolve_method, build_proposer, build_grounder
from actions.action_executor import ActionExecutor
from agent.controller import AgentController
from agent.memory import AgentMemory
from agent.recovery import RecoveryManager
from agent.long_horizon.controller import LongHorizonController
from agent.long_horizon.experience_memory import ExperienceMemory
from verification.verifier import StateVerifier
from verification.goal_verifier import GoalVerifier
from models.model_factory import get_vlm

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "trajectory_logger",
    str(Path(__file__).resolve().parents[1] / "src" / "logging" / "trajectory_logger.py"),
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TrajectoryLogger = _mod.TrajectoryLogger

logger = logging.getLogger("run_agent")

GOAL_MODES = ("heuristic", "vlm", "both", "off")


def build_mock_responses(task: str):
    """Demo script: return a generic done so the mock run terminates gracefully."""
    return [json.dumps({"action": "done", "reason": "mock demo done"})]


def maybe_enable_captcha_gate(args, controller, name="controller"):
    """Attach the CAPTCHA safety gate when requested (default OFF).

    Flag `--captcha-gate` or env ENABLE_CAPTCHA_GATE=1. Uses default
    detector settings (DOM+frame+text+structural; OCR/VLM off). Never
    changes Planner/Grounder/STV/HITL themselves.
    """
    import os as _os
    if not getattr(args, "captcha_gate", False) and \
            _os.getenv("ENABLE_CAPTCHA_GATE", "").strip().lower() not in (
                "1", "true", "yes", "on"):
        return False
    from agent.captcha_gate import CaptchaGate
    controller.set_captcha_gate(CaptchaGate())
    controller.captcha_max_rounds = int(
        getattr(args, "captcha_max_rounds", 0) or 3)
    logger.info("[CAPTCHA-GATE] enabled=True on %s "
                "(blocking only -> existing HITL, reason=captcha_detected)",
                name)
    return True


def maybe_start_dashboard(args):
    """Build the dashboard (bridge + server) when --dashboard is set.

    Returns (bridge, server), or None when --dashboard is absent (callers
    gate on `is not None`, so a (None, None) tuple must NOT be returned).
    The agent keeps running with terminal input unless the returned
    bridge is attached to a controller.
    """
    if not getattr(args, "dashboard", False):
        return None
    from dashboard.hitl_bridge import DashboardHITLBridge
    from dashboard.server import DashboardServer
    bridge = DashboardHITLBridge(
        hitl_timeout_s=float(getattr(args, "hitl_timeout", 600.0) or 600.0))
    host = getattr(args, "dashboard_host", "127.0.0.1") or "127.0.0.1"
    port = int(getattr(args, "dashboard_port", 3000) or 3000)
    print(f"\n=== GUI Web Agent Dashboard ===\n"
          f"  URL (via SSH):  ssh -L {port}:localhost:{port} <server>\n"
          f"                  then open http://localhost:{port}\n"
          f"  noVNC upstream: {getattr(args, 'novnc_base', '')}\n")
    if host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning("Dashboard bound to %s (non-loopback): only do this "
                       "on trusted networks", host)
    server = DashboardServer(
        bridge, host=host,
        port=int(getattr(args, "dashboard_port", 3000) or 3000),
        novnc_base=getattr(args, "novnc_base",
                           "http://127.0.0.1:6080") or
        "http://127.0.0.1:6080")
    return bridge, server


def run_long_horizon(args, method, browser, observer, grounder, executor,
                     vlm, traj, goal_verifier, enable_goal_check,
                     dashboard=None):
    """Build and run the verified long-horizon controller.

    Detector/OCR backends are optional: construction never fails the run;
    StateExtractor degrades to text/DOM signatures when a backend is
    missing or errors (logged, never silent).
    """
    from agent.long_horizon.recovery import RecoveryPolicy

    if args.no_experience:
        experience = None
    else:
        try:
            experience = ExperienceMemory(Path(args.experience_path))
        except Exception as e:
            logger.warning(f"Experience store unavailable ({e}); "
                           f"continuing without experience")
            experience = None

    detector, ocr = None, None
    if args.stv_detector == "omniparser":
        try:
            from browser.proposers import OmniParserDetector
            detector = OmniParserDetector(device=args.device)
        except Exception as e:
            logger.warning(f"STV detector unavailable ({e}); "
                           f"continuing without visual detector")
    if args.stv_ocr == "paddle":
        try:
            from browser.ocr import PaddleOCRBackend
            ocr = PaddleOCRBackend()
        except Exception as e:
            logger.warning(f"STV OCR unavailable ({e}); "
                           f"continuing without OCR signatures")

    if args.no_recovery:
        recovery = RecoveryPolicy(
            recovery=RecoveryManager(max_re_ground_attempts=0,
                                     max_recovery_attempts=0,
                                     max_execution_retries=0),
            max_waits=0, max_retries=0, max_regrounds=0,
            max_local_repairs=0, max_local_replans=0,
            max_global_replans=0, max_rollbacks=0)
    else:
        recovery = None  # controller default (10 global recoveries)

    ckpt_dir = args.checkpoints_dir
    if ckpt_dir is None and traj is not None:
        ckpt_dir = str(Path(traj.path) / "checkpoints")
    controller = LongHorizonController(
        browser, observer, grounder, executor, vlm,
        memory=None, recovery=recovery, trajectory_logger=traj,
        max_steps=args.max_steps, task=args.task,
        goal_verifier=goal_verifier, enable_goal_check=enable_goal_check,
        planning_horizon=args.planning_horizon,
        experience_memory=experience,
        checkpoints_dir=Path(ckpt_dir) if ckpt_dir else None,
        enable_stv=not args.no_stv,
        enable_planning=not args.no_planning,
        enable_experience=not args.no_experience,
        enable_checkpoints=not args.no_checkpoints,
        risk=args.risk, detector=detector, ocr=ocr,
        enable_state_gate=not args.disable_state_gate,
        enable_micro_planner=not args.disable_micro_planner,
        enable_milestone_graph=not args.disable_milestone_graph,
        enable_progress_score=not args.disable_progress_score,
        enable_failure_classifier=not args.disable_failure_classifier,
        enable_task_compiler=not getattr(args, "no_task_compiler", False),
        action_chunking=getattr(args, "action_chunking", False),
        chunk_size=getattr(args, "chunk_size", 3),
        adaptive_chunking=getattr(args, "adaptive_chunking", False),
        adaptive_chunk_max_size=getattr(args, "adaptive_chunk_max", 3),
    )
    maybe_enable_captcha_gate(args, controller, "long-horizon controller")
    if dashboard is not None:
        bridge, server = dashboard
        server.attach_controller(controller)
        if args.planner_mode == "hierarchical":
            outcome = server.run_agent_foreground(
                lambda: controller.run_hierarchical(args.task), args.task)
        else:
            outcome = server.run_agent_foreground(
                lambda: controller.run(args.task), args.task)
        return outcome
    if args.planner_mode == "hierarchical":
        outcome = controller.run_hierarchical(args.task)
    else:
        outcome = controller.run(args.task)
    if traj is not None:
        traj.set_metadata({
            "method": method,
            "controller": "long_horizon",
            "action_chunking": getattr(args, "action_chunking", False),
            "chunk_size": getattr(args, "chunk_size", 3) if getattr(args, "action_chunking", False) else None,
            "adaptive_chunking": getattr(args, "adaptive_chunking", False),
            "adaptive_chunk_max": getattr(args, "adaptive_chunk_max", 3) if getattr(args, "adaptive_chunking", False) else None,
            # Additive: whether the terminal goal check was actually
            # available on the ACC path (None when ACC off).
            "adaptive_goal_verification": (
                bool(goal_verifier) and enable_goal_check
            ) if getattr(args, "adaptive_chunking", False) else None,
            "enable_stv": not args.no_stv,
            "enable_planning": not args.no_planning,
            "enable_experience": not args.no_experience,
            "enable_checkpoints": not args.no_checkpoints,
            "planning_horizon": args.planning_horizon,
            "risk": args.risk,
            "stv_detector": args.stv_detector,
            "stv_ocr": args.stv_ocr,
            "planner_mode": args.planner_mode,
            "state_gate_enabled": not args.disable_state_gate,
            "micro_planner_enabled": not args.disable_micro_planner,
            "milestone_graph_enabled": not args.disable_milestone_graph,
            "progress_score_enabled": not args.disable_progress_score,
            "failure_classifier_enabled":
                not args.disable_failure_classifier,
            "goal_mode": args.goal_mode,
            "goal_verification_enabled": enable_goal_check,
            "recovery_enabled": not args.no_recovery,
            "model": args.vlm,
            "model_path": args.model_path,
            "device": args.device,
            "max_tokens": args.max_tokens,
        })
    return outcome


def build_parser():
    from grounding.methods import add_method_argument
    parser = argparse.ArgumentParser(description="Run the GUI web agent")
    parser.add_argument("--task", required=True, help="Natural language task")
    parser.add_argument("--url", required=True, help="Start URL (http(s):// or file://)")
    # Default som_dom preserves the historical default: SoM grounding with
    # SoMGrounder's default include_dom_in_prompt=True.
    add_method_argument(parser, default="som_dom")
    parser.add_argument("--vlm", default="mock", help="mock | qwen2.5-vl | gui-actor")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--headless", dest="headless", action="store_true",
                        default=True, help="Run headless (default; kept for compat)")
    parser.add_argument("--no-headless", dest="headless", action="store_false",
                        help="Run headed (debugging; needs display/Xvfb)")
    parser.add_argument("--viewport-width", type=int, default=1280)
    parser.add_argument("--viewport-height", type=int, default=720)
    parser.add_argument("--trajectory-dir", default="./data/trajectories")
    parser.add_argument("--no-trajectories", action="store_true",
                        help="Disable trajectory PNG/JSON logging")
    parser.add_argument("--no-verify", action="store_true",
                        help="Disable StateVerifier (action-level verification)")
    parser.add_argument("--goal-mode", choices=GOAL_MODES, default="heuristic",
                        help="GoalVerifier mode; 'off' disables goal-achievement early stop")
    parser.add_argument("--no-recovery", action="store_true",
                        help="Disable recovery (zero all recovery budgets)")
    parser.add_argument("--mock-responses", default=None,
                        help="JSON file with list of mock VLM response strings")
    parser.add_argument("--human-mode", action="store_true",
                        help="Human-observable speed: slow glides, visible cursor holds, "
                             "keystroke typing. Default (flag absent) is machine mode: fast.")
    parser.add_argument("--browser-takeover", action="store_true", default=False,
                        help="noVNC takeover HITL: ask_user turns wait for YOUR "
                             "clicks/typing in the noVNC browser window "
                             "(auto-resume on page change) instead of "
                             "blocking on terminal input. Needs headed "
                             "browser (--no-headless) under Xvfb + x11vnc/noVNC "
                             "on the server. Also enabled via HITL_TAKEOVER=1.")
    parser.add_argument("--takeover-timeout", type=float, default=120.0,
                        help="Takeover wait budget in seconds before falling "
                             "back to terminal ask (default: 120).")
    parser.add_argument("--takeover-poll", type=float, default=2.0,
                        help="Takeover re-observe interval in seconds "
                             "(default: 2).")
    parser.add_argument("--ask-route", choices=("terminal", "takeover", "auto"),
                        default=None,
                        help="Per-question HITL transport (overrides "
                             "HITL_ASK_ROUTE env; default: terminal). 'auto' "
                             "sends credential-shaped asks to noVNC takeover "
                             "and everything else to terminal input. Legacy "
                             "--browser-takeover forces takeover for all.")
    parser.add_argument("--captcha-gate", action="store_true", default=False,
                        help="CAPTCHA safety gate: a BLOCKING challenge pauses "
                             "autonomous execution via the existing HITL path "
                             "(reason=captcha_detected) and re-detects after "
                             "you solve it in the browser/noVNC window. "
                             "Also enabled via ENABLE_CAPTCHA_GATE=1. "
                             "Non-blocking widgets never interrupt. Combine "
                             "with --browser-takeover (or --ask-route "
                             "takeover) so solving happens in noVNC.")
    parser.add_argument("--captcha-max-rounds", type=int, default=3,
                        help="Bounded human-solve/re-detect rounds per "
                             "CAPTCHA encounter (default: 3).")
    # Experimental Action Chunking (opt-in CLI feature)
    parser.add_argument("--action-chunking", action="store_true", default=False,
                        help="Experimental: execute several predictable browser actions "
                             "within a single planning cycle instead of invoking the "
                             "planner after every individual action.")
    parser.add_argument("--chunk-size", type=int, default=3,
                        help="Experimental: chunk size for action chunking (2-6, default: 3).")
    # Adaptive Chunk Controller (opt-in, extends --action-chunking; default
    # OFF so all existing behavior, including plain --action-chunking, is
    # byte-identical).
    parser.add_argument("--adaptive-chunking", action="store_true", default=False,
                        help="Experimental Adaptive Chunk Controller (ACC): "
                             "deterministically size each action chunk from "
                             "DOM stability, target validity, and recent "
                             "outcomes; enforce expected_transition contracts "
                             "and goal-verify terminal done (requires "
                             "--action-chunking). Default off.")
    parser.add_argument("--adaptive-chunk-max", type=int, default=3,
                        help="ACC hard cap on chunk size (1-6, default: 3; "
                             "values above 3 are experimental and validated "
                             "when --adaptive-chunking is on).")
    # Long-horizon verified control (leaves the baseline path untouched).
    parser.add_argument("--long-horizon", action="store_true",
                        help="Use the verified subgoal loop (plan -> act -> "
                             "STV-verify -> commit) instead of AgentController")
    parser.add_argument("--no-stv", action="store_true",
                        help="LH only: disable State Transition Verification "
                             "(ablation; nothing becomes VERIFIED)")
    parser.add_argument("--no-planning", action="store_true",
                        help="LH only: skip global/local planners (single-goal "
                             "reactive ablation)")
    parser.add_argument("--no-experience", action="store_true",
                        help="LH only: disable experience retrieval")
    parser.add_argument("--no-checkpoints", action="store_true",
                        help="LH only: disable checkpoint snapshots")
    parser.add_argument("--no-task-compiler", action="store_true",
                        help="LH only: disable TaskCompiler and SubgoalContract gate (ablation)")
    parser.add_argument("--planning-horizon", type=int, default=3,
                        help="LH only: local planner action horizon (0 disables "
                             "batch planning; post-recovery forces 1)")
    parser.add_argument("--risk", choices=("low", "normal", "high"),
                        default="normal",
                        help="LH only: planning risk posture")
    parser.add_argument("--experience-path",
                        default="./data/experience.jsonl",
                        help="LH only: JSONL experience store path")
    parser.add_argument("--checkpoints-dir", default=None,
                        help="LH only: checkpoint dir (default: "
                             "<trajectory-dir>/checkpoints)")
    parser.add_argument("--stv-detector", choices=("none", "omniparser"),
                        default="none",
                        help="LH only: visual detector for STV signatures "
                             "(failures degrade to text/DOM, never silent)")
    parser.add_argument("--stv-ocr", choices=("none", "paddle"),
                        default="none",
                        help="LH only: OCR backend for STV signatures")
    parser.add_argument("--planner-mode", choices=("global", "hierarchical"),
                        default="global",
                        help="LH only: legacy subgoal queue (global, "
                             "validated default) vs rolling-horizon "
                             "milestone loop (hierarchical, research)")
    parser.add_argument("--disable-state-gate", action="store_true",
                        help="LH ablation: skip pre-execution page/stage "
                             "compatibility gate")
    parser.add_argument("--disable-micro-planner", action="store_true",
                        help="LH ablation: hierarchical loop uses the legacy "
                             "local planner instead of the micro-planner")
    parser.add_argument("--disable-milestone-graph", action="store_true",
                        help="LH ablation: milestones run as a flat sequence "
                             "(dependencies dropped)")
    parser.add_argument("--disable-progress-score", action="store_true",
                        help="LH ablation: omit progress distance logging "
                             "and remaining_requirements from planner "
                             "context")
    parser.add_argument("--disable-failure-classifier", action="store_true",
                        help="LH ablation: record failures without "
                             "class/scope metadata; scope-based routing off")
    parser.add_argument("--intent-gate", action="store_true", default=False,
                        help="Legacy alias for '--intent-gate-mode llm' "
                             "(Experiment 1). If both flags are given, "
                             "--intent-gate-mode wins.")
    parser.add_argument("--intent-gate-mode", choices=("off", "llm", "hybrid"),
                        default=None,
                        help="Gate mode (overrides INTENT_GATE_MODE env; "
                             "default when omitted: hybrid). "
                             "'llm' = Experiment 1, 'hybrid' = Experiment 2.")
    parser.add_argument("--dashboard", action="store_true", default=False,
                        help="Serve the Agent Dashboard (noVNC + HITL chat on "
                             "one page) and route ask_user turns to the chat "
                             "instead of terminal input. Agent runs in a "
                             "background thread; open via SSH forwarding.")
    parser.add_argument("--dashboard-host", default="127.0.0.1",
                        help="Dashboard bind host (default 127.0.0.1; keep "
                             "loopback for SSH-only access).")
    parser.add_argument("--dashboard-port", type=int, default=3000,
                        help="Dashboard port (default 3000).")
    parser.add_argument("--novnc-base", default="http://127.0.0.1:6080",
                        help="Upstream noVNC/websockify base URL proxied at "
                             "/novnc/* (default http://127.0.0.1:6080).")
    parser.add_argument("--hitl-timeout", type=float, default=600.0,
                        help="Dashboard HITL wait budget in seconds before "
                             "recording an empty answer (default 600).")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.action_chunking:
        if args.chunk_size < 2 or args.chunk_size > 6:
            parser.error("--chunk-size must be between 2 and 6")
    if args.adaptive_chunking:
        if not args.action_chunking:
            parser.error("--adaptive-chunking requires --action-chunking")
        if args.adaptive_chunk_max < 1 or args.adaptive_chunk_max > 6:
            parser.error("--adaptive-chunk-max must be between 1 and 6")
        # Fail-closed design (spec §2): with ACC on, a terminal done is
        # only accepted after an independent goal verification. With
        # --goal-mode off there is no verifier, so every terminal done
        # would be rejected and the task could never honestly complete
        # through the done path. Refuse the combination up front.
        if args.goal_mode == "off":
            parser.error(
                "--adaptive-chunking requires an independent terminal "
                "goal check; --goal-mode off provides none, so unverified "
                "completions would all be rejected. Use "
                "--goal-mode heuristic|vlm|both.")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if args.action_chunking:
        logger.info("[ACTION_CHUNK] Experimental action chunking enabled (chunk_size=%d)",
                    args.chunk_size)
    if args.adaptive_chunking:
        logger.info("[ACC] Adaptive Chunk Controller enabled (hard_cap=%d)",
                    args.adaptive_chunk_max)

    method = resolve_method(args.method)

    # Resolve file:// relative paths against project root
    url = args.url
    if url.startswith("file://") and not url.startswith("file:///"):
        rel = url[len("file://"):]
        abs_path = (Path.cwd() / rel).resolve()
        url = f"file://{abs_path}"

    if args.vlm == "mock":
        if args.mock_responses:
            with open(args.mock_responses) as f:
                responses = json.load(f)
        else:
            responses = build_mock_responses(args.task)
        vlm = get_vlm("mock", responses=responses)
    else:
        vlm = get_vlm(args.vlm, model_path=args.model_path, device=args.device,
                      max_new_tokens=args.max_tokens)
    # Fail-fast (spec §2): with ACC on and a VLM-based goal mode, a missing
    # VLM means the terminal goal check is unavailable — every completion
    # would be rejected. Surface the misconfiguration before the run.
    if args.adaptive_chunking and args.goal_mode in ("vlm", "both") \
            and vlm is None:
        parser.error(
            "--adaptive-chunking with --goal-mode vlm|both requires a "
            "working VLM backend; none could be loaded, so the terminal "
            "goal check would be unavailable (fail-closed)")

    if args.human_mode:
        speed = {"slow_mo": 500, "move_steps": 80,
                 "cursor_pause": 1.5, "type_delay": 150}
    else:  # machine mode (default): fast
        speed = {"slow_mo": 0, "move_steps": 10,
                 "cursor_pause": 0.0, "type_delay": 20}
    browser = BrowserManager(headless=args.headless,
                             viewport_width=args.viewport_width,
                             viewport_height=args.viewport_height,
                             slow_mo_ms=speed["slow_mo"])
    try:
        browser.launch()
        browser.new_page()
        browser.navigate(url)

        proposer = build_proposer(method)
        if args.long_horizon and proposer is None:
            # LH + STV require DOM evidence (anchor, attribution, field
            # values, signatures). Coordinate grounding ignores DOM, so
            # attaching a DOM proposer changes grounding not at all --
            # it only fills observation dom_elements. Baseline
            # coordinate runs (non-LH) are untouched.
            from browser.proposers import DOMProposer
            proposer = DOMProposer()
            logger.info("LH mode: attached DOMProposer for observation "
                        "evidence (grounding stays coordinate)")
        observer = Observer(browser, proposer=proposer)
        grounder = build_grounder(method, vlm,
                                  viewport_width=args.viewport_width,
                                  viewport_height=args.viewport_height)
        executor = ActionExecutor(browser, viewport_width=args.viewport_width,
                                  viewport_height=args.viewport_height,
                                  type_delay_ms=speed["type_delay"],
                                  move_steps=speed["move_steps"],
                                  cursor_pause=speed["cursor_pause"])
        verifier = None if args.no_verify else StateVerifier()
        if args.goal_mode == "off":
            goal_verifier, enable_goal_check = None, False
        else:
            goal_verifier = GoalVerifier(
                mode=args.goal_mode,
                vlm=vlm if args.goal_mode in ("vlm", "both") else None)
            enable_goal_check = True
        traj = None if args.no_trajectories else TrajectoryLogger(
            Path(args.trajectory_dir))
        dashboard = maybe_start_dashboard(args)
        if args.long_horizon:
            outcome = run_long_horizon(
                args, method, browser, observer, grounder, executor, vlm,
                traj, goal_verifier, enable_goal_check,
                dashboard=dashboard)
        else:
            if args.no_recovery:
                recovery = RecoveryManager(max_re_ground_attempts=0,
                                           max_recovery_attempts=0,
                                           max_execution_retries=0)
            else:
                recovery = RecoveryManager()
            controller = AgentController(
                browser, observer, grounder, executor, verifier=verifier,
                memory=AgentMemory(), recovery=recovery,
                trajectory_logger=traj, max_steps=args.max_steps,
                task=args.task,
                goal_verifier=goal_verifier,
                enable_goal_check=enable_goal_check,
                action_chunking=args.action_chunking,
                chunk_size=args.chunk_size,
            )
            import os as _os
            if args.ask_route is not None:
                controller.ask_route = args.ask_route
            elif _os.getenv("HITL_ASK_ROUTE", "").strip():
                controller.ask_route = _os.getenv("HITL_ASK_ROUTE",
                                                  "terminal")
            if args.browser_takeover or _os.getenv("HITL_TAKEOVER", "") \
                    .strip().lower() in ("1", "true", "yes", "on"):
                controller.takeover_mode = True
                controller.takeover_timeout_s = args.takeover_timeout
                controller.takeover_poll_s = args.takeover_poll
                if args.headless:
                    logger.warning(
                        "Takeover HITL needs a headed browser "
                        " visible over noVNC; forcing --no-headless off "
                        "is recommended (continuing headless anyway).")
                logger.info(
                    "[HITL-TAKEOVER] enabled (timeout=%.0fs poll=%.1fs)",
                    args.takeover_timeout, args.takeover_poll)
            import os as _os
            # Precedence: explicit --intent-gate-mode beats legacy
            # --intent-gate beats env. Unset everything -> hybrid default.
            if args.intent_gate_mode is not None:
                _os.environ["INTENT_GATE_MODE"] = args.intent_gate_mode
                if args.intent_gate_mode == "llm":
                    _os.environ["ENABLE_EXPERIMENTAL_INTENT_GATE"] = "1"
            elif args.intent_gate:
                # Back-compat: legacy flag == Experiment 1 (llm).
                _os.environ["INTENT_GATE_MODE"] = "llm"
                _os.environ["ENABLE_EXPERIMENTAL_INTENT_GATE"] = "1"
            from agent.hybrid_gate import get_intent_gate_mode
            from agent.intent_gate import IntentGate
            _mode = get_intent_gate_mode()
            if _mode == "hybrid":
                from agent.hybrid_gate import HybridGate, OllamaTextJudge
                controller.set_hybrid_gate(HybridGate(llm=OllamaTextJudge()))
                logger.info("[HYBRID-GATE] enabled=True mode=hybrid (default)")
            elif _mode == "llm":
                # Reuse the grounding VLM for text-only gate queries
                # (blank image); no second model load.
                controller.set_intent_gate(IntentGate(text_llm=vlm))
                logger.info("[INTENT-GATE] enabled=True mode=llm")
            else:
                logger.info("[HYBRID-GATE] mode=off: gate disabled")
            maybe_enable_captcha_gate(args, controller, "baseline controller")
            if dashboard is not None:
                bridge, server = dashboard
                server.attach_controller(controller)
                outcome = server.run_agent_foreground(
                    lambda: controller.run(args.task), args.task)
            else:
                outcome = controller.run(args.task)
        if traj is not None and not args.long_horizon:
            traj.set_metadata({
                "method": method,
                "controller": "baseline",
                "action_chunking": args.action_chunking,
                "chunk_size": args.chunk_size if args.action_chunking else None,
                "adaptive_chunking": getattr(args, "adaptive_chunking", False),
                "adaptive_chunk_max": getattr(args, "adaptive_chunk_max", 3) if getattr(args, "adaptive_chunking", False) else None,
                # Additive: whether the terminal goal check was actually
                # available on the ACC path (None when ACC off).
                "adaptive_goal_verification": (
                    bool(goal_verifier) and enable_goal_check
                ) if getattr(args, "adaptive_chunking", False) else None,
                "goal_mode": args.goal_mode,
                "goal_verification_enabled": enable_goal_check,
                "state_verification_enabled": not args.no_verify,
                "recovery_enabled": not args.no_recovery,
                "model": args.vlm,
                "model_path": args.model_path,
                "device": args.device,
                "max_tokens": args.max_tokens,
            })
        print(json.dumps(outcome, indent=2, default=str))
        if traj is not None:
            print(f"\nTrajectory saved to: {traj.path}")
    finally:
        browser.close()


if __name__ == "__main__":
    main()
