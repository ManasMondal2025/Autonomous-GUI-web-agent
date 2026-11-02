"""Agent Dashboard: browser-based UI for the GUI Web Agent.

Single page (:3000) embedding the noVNC browser session plus an agent
chat with WebSocket HITL. Stdlib only -- no new dependencies.

Components:
  hitl_bridge.py  Agent<->chat bridge (ask_handler + CAPTCHA typing).
  server.py       HTTP + WebSocket + noVNC reverse proxy.
  static/         Dashboard frontend (no framework).
Entry point: scripts/run_dashboard.py (agent runs in a background thread;
server threads never block on HITL).
"""
