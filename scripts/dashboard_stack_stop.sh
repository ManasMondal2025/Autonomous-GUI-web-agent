#!/usr/bin/env bash
# Stop the Agent Dashboard display stack (leaves agent browsers alone;
# stop those with: pkill -9 chrome; pkill -9 chromium).
set -u
pkill -x Xvfb 2>/dev/null && echo "stopped Xvfb" || echo "Xvfb not running"
pkill -x x11vnc 2>/dev/null && echo "stopped x11vnc" || echo "x11vnc not running"
pkill -f "websockify.*6080" 2>/dev/null && echo "stopped websockify" \
  || echo "websockify not running"
