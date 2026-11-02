#!/usr/bin/env python3
"""Agent Dashboard entry point.

Same agent as scripts/run_agent.py, plus a single-port dashboard
(noVNC browser + HITL chat) on :3000. `--dashboard` is forced on; all
other run_agent.py flags work unchanged.

Server side (once):
  Xvfb :99 & export DISPLAY=:99
  x11vnc -display :99 -localhost -xkb &          # RFB on 5900
  websockify --web /usr/share/novnc/ 6080 localhost:5900 &

Laptop side:
  ssh -L 3000:localhost:3000 USER@SERVER
  # open http://localhost:3000 (browser + agent chat, one page)

Example:
  python scripts/run_dashboard.py --task "Register on the demo site" \\
      --url "file://test_pages/test_page.html" --method som_dom \\
      --vlm mock --no-headless --human-mode --captcha-gate
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_agent


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--dashboard" not in argv:
        argv = ["--dashboard"] + argv
    run_agent.main(argv)
