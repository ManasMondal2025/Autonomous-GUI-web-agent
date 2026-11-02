#!/usr/bin/env bash
# Display stack for the Agent Dashboard (idempotent -- safe to re-run).
# Provides: Xvfb (:99) + x11vnc (5900, localhost only) + websockify (6080).
# Your existing start_gui.sh does the same job; use either one.
#
#   ./scripts/dashboard_stack_start.sh
set -u
DISP="${DISPLAY_NUM:-99}"
export DISPLAY=":${DISP}"

need() { command -v "$1" >/dev/null 2>&1 || {
  echo "MISSING: $1 (sudo apt install -y $1)"; exit 1; }; }
need Xvfb; need x11vnc; need websockify

if ! pgrep -x Xvfb >/dev/null; then
  Xvfb ":${DISP}" -screen 0 1280x800x24 >/tmp/xvfb.log 2>&1 &
  echo "started Xvfb on :${DISP}"
else echo "Xvfb already running"; fi

if ! pgrep -x x11vnc >/dev/null; then
  # -forever: survive viewer disconnects (without it x11vnc exits with the
  # last viewer and :5900 goes dark while websockify still answers).
  # -shared: dashboard proxy + direct vnc.html may connect simultaneously.
  x11vnc -display ":${DISP}" -localhost -xkb \
    -forever -shared >/tmp/x11vnc.log 2>&1 &
  echo "started x11vnc (5900, localhost only, forever+shared)"
else echo "x11vnc already running"; fi

if ! pgrep -f "websockify.*6080" >/dev/null; then
  websockify --web /usr/share/novnc/ 6080 localhost:5900 \
    >/tmp/websockify.log 2>&1 &
  echo "started websockify :6080 -> :5900"
else echo "websockify already running"; fi

sleep 1
echo "---"
echo "DISPLAY=${DISPLAY} (export this before any Playwright command)"
(command -v ss >/dev/null && ss -ltn 2>/dev/null | grep -E ':6080|:5900') || true
echo "Dashboard will proxy this at :3000/novnc/*"
