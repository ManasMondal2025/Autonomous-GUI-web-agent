"""Agent Dashboard server: stdlib only (no new dependencies).

Serves on one port (default 127.0.0.1:3000, SSH-forwarded):
  GET  /                  dashboard page
  GET  /static/*          frontend assets
  GET  /api/status        JSON agent/chat status
  GET  /api/config        JSON frontend config (noVNC paths)
  GET  /ws                chat WebSocket (HITL + events + status)
  ALL  /novnc/*           reverse proxy to the noVNC/websockify upstream
                          (plain HTTP forwarded; Upgrade: websocket paths
                          are TCP-bridged so RFB keeps working). The user
                          only ever opens :3000.

Threading: ThreadingHTTPServer gives each connection a thread; the agent
runs on its own thread and blocks only on per-request Events inside the
HITL bridge. Waiting for a human never stalls other connections.
"""

import base64
import hashlib
import json
import logging
import mimetypes
import os
import select
import socket
import struct
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


# --------------------------------------------------------------------------
# Minimal WebSocket (RFC 6455) server framing -- text/ping/pong/close only.
# --------------------------------------------------------------------------

def _ws_accept_key(key: str) -> str:
    return base64.b64encode(
        hashlib.sha1((key + WS_GUID).encode()).digest()).decode()


def ws_send_text(sock: socket.socket, text: str) -> None:
    data = text.encode("utf-8")
    header = bytes([0x81])
    n = len(data)
    if n < 126:
        header += struct.pack("!B", n)
    elif n < (1 << 16):
        header += struct.pack("!B", 126) + struct.pack("!H", n)
    else:
        header += struct.pack("!B", 127) + struct.pack("!Q", n)
    sock.sendall(header + data)


def ws_send_close(sock: socket.socket) -> None:
    try:
        sock.sendall(bytes([0x88, 0x00]))
    except OSError:
        pass


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return buf


def ws_recv_message(sock: socket.socket) -> Optional[dict]:
    """Read one client frame. Returns {'opcode': int, 'data': bytes} or
    None on clean close. Raises ConnectionError on transport failure."""
    hdr = _recv_exact(sock, 2)
    b1, b2 = hdr[0], hdr[1]
    opcode = b1 & 0x0F
    masked = bool(b2 & 0x80)
    length = b2 & 0x7F
    if length == 126:
        length = struct.unpack("!H", _recv_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    mask = _recv_exact(sock, 4) if masked else None
    payload = _recv_exact(sock, length) if length else b""
    if masked and mask:
        payload = bytes(c ^ mask[i % 4] for i, c in enumerate(payload))
    if opcode == 0x8:
        return None
    if opcode == 0x9:  # ping -> caller replies pong
        return {"opcode": opcode, "data": payload}
    if opcode == 0xA:  # unsolicited pong: ignore
        return {"opcode": opcode, "data": payload}
    if opcode == 0x1:
        return {"opcode": opcode, "data": payload}
    raise ConnectionError(f"unsupported opcode {opcode}")


# --------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------

class DashboardServer:
    def __init__(self, bridge, host: str = "127.0.0.1", port: int = 3000,
                 novnc_base: str = "http://127.0.0.1:6080",
                 upstream_timeout_s: float = 5.0):
        self.bridge = bridge
        self.host = host
        self.port = port
        self.upstream_timeout_s = upstream_timeout_s
        parts = urlsplit(novnc_base)
        self.upstream_host = parts.hostname or "127.0.0.1"
        self.upstream_port = parts.port or 6080
        self.upstream_tls = parts.scheme == "https"
        if self.upstream_tls:
            raise ValueError("noVNC upstream must be plain http (same-host "
                             "websockify); got https")
        self._httpd: Optional[ThreadingHTTPServer] = None

    # -- lifecycle ------------------------------------------------------
    def serve_forever(self) -> None:
        handler = self._make_handler()
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        # threads must not keep the process alive on Ctrl-C
        self._httpd.daemon_threads = True
        logger.info("Dashboard on http://%s:%s (SSH: ssh -L %s:localhost:%s)",
                    self.host, self.port, self.port, self.port)
        try:
            self._httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None

    def attach_controller(self, controller: Any) -> None:
        """Point a (baseline or long-horizon) controller's HITL at the chat.

        Uses the bridge object itself so the baseline's for_action hook
        (CAPTCHA typing) keeps working; the LH controller calls it plainly.
        No controller logic is modified.
        """
        controller.ask_handler = self.bridge

    # -- helpers used by the handler -------------------------------------
    def novnc_reachable(self) -> bool:
        try:
            s = socket.create_connection(
                (self.upstream_host, self.upstream_port),
                timeout=self.upstream_timeout_s)
            s.close()
            return True
        except OSError:
            return False

    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "AgentDashboard/1.0"

            def log_message(self, fmt, *args):  # quieter access log
                logger.debug("HTTP %s", fmt % args)

            # -- routing -------------------------------------------------
            def do_GET(self):
                path = urlsplit(self.path).path
                if path == "/ws":
                    self._handle_ws()
                elif path == "/" or path == "/index.html":
                    self._serve_static("index.html")
                elif path.startswith("/static/"):
                    self._serve_static(path[len("/static/"):])
                elif path == "/api/status":
                    self._serve_json(self._status_payload())
                elif path == "/api/config":
                    self._serve_json({
                        "novnc_iframe": "/novnc/vnc.html?autoconnect=true"
                                        "&resize=scale&path=novnc/websockify",
                        "novnc_direct_hint": (
                            f"http://{server.upstream_host}:"
                            f"{server.upstream_port}/vnc.html"),
                    })
                elif path.startswith("/novnc/"):
                    self._proxy_novnc(path)
                else:
                    self.send_error(404, "not found")

            def do_POST(self):
                path = urlsplit(self.path).path
                if path.startswith("/novnc/"):
                    self._proxy_novnc(path)
                else:
                    self.send_error(404, "not found")

            # -- static / api ---------------------------------------------
            def _serve_static(self, name: str):
                target = (STATIC_DIR / name.lstrip("/")).resolve()
                try:
                    target.relative_to(STATIC_DIR.resolve())
                except ValueError:
                    self.send_error(403, "forbidden")
                    return
                if not target.is_file():
                    self.send_error(404, "not found")
                    return
                ctype, _ = mimetypes.guess_type(str(target))
                body = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type",
                                 ctype or "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(body)

            def _serve_json(self, obj: dict):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _status_payload(self) -> dict:
                snap = server.bridge.status_snapshot()
                snap["novnc_reachable"] = server.novnc_reachable()
                snap["novnc_upstream"] = (
                    f"{server.upstream_host}:{server.upstream_port}")
                return snap

            # -- chat WebSocket --------------------------------------------
            def _handle_ws(self):
                key = self.headers.get("Sec-WebSocket-Key")
                if not key or self.headers.get("Upgrade", "").lower() != \
                        "websocket":
                    self.send_error(400, "websocket upgrade required")
                    return
                try:
                    raw = self.connection
                    resp = (
                        "HTTP/1.1 101 Switching Protocols\r\n"
                        "Upgrade: websocket\r\n"
                        "Connection: Upgrade\r\n"
                        f"Sec-WebSocket-Accept: {_ws_accept_key(key.strip())}\r\n"
                        "\r\n").encode()
                    raw.sendall(resp)
                except OSError:
                    return
                send_lock = threading.Lock()

                def send(obj: dict) -> bool:
                    try:
                        with send_lock:
                            ws_send_text(raw, json.dumps(obj))
                        return True
                    except OSError:
                        return False

                cid = server.bridge.hub.register(send)
                try:
                    # Recovery: replay state + any waiting HITL requests so a
                    # refresh/reconnect never silently loses a question.
                    snap = server.bridge.status_snapshot()
                    send({"type": "STATUS", "state": snap["state"],
                          "message": snap["message"]})
                    for req in server.bridge.active_requests():
                        msg = ({"type": "CAPTCHA_REQUEST",
                                "request_id": req["request_id"],
                                "message": req["question"],
                                "action": "user_required"}
                               if req["kind"] == "captcha" else
                               {"type": "HITL_REQUEST",
                                "request_id": req["request_id"],
                                "question": req["question"],
                                "input_type": "text"})
                        send(msg)
                    while True:
                        try:
                            frame = ws_recv_message(raw)
                        except ConnectionError:
                            break
                        if frame is None:
                            break
                        if frame["opcode"] == 0x9:
                            with send_lock:
                                try:
                                    raw.sendall(b"\x8a" + bytes(
                                        [len(frame["data"])]) + frame["data"])
                                except OSError:
                                    break
                            continue
                        if frame["opcode"] != 0x1:
                            continue
                        server._handle_chat_message(send, frame["data"])
                finally:
                    server.bridge.hub.unregister(cid)
                    try:
                        ws_send_close(raw)
                    except OSError:
                        pass

            # -- noVNC reverse proxy ----------------------------------------
            def _proxy_novnc(self, path: str):
                upstream_path = path[len("/novnc"):] or "/"
                if self.headers.get("Upgrade", "").lower() == "websocket":
                    self._proxy_ws(upstream_path)
                else:
                    self._proxy_http(upstream_path)

            def _proxy_http(self, upstream_path: str):
                import http.client as _http
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else None
                try:
                    conn = _http.HTTPConnection(
                        server.upstream_host, server.upstream_port,
                        timeout=server.upstream_timeout_s)
                    fwd = {k: v for k, v in self.headers.items()
                           if k.lower() not in ("host", "connection")}
                    conn.request(self.command, upstream_path,
                                 body=body, headers=fwd)
                    resp = conn.getresponse()
                    data = resp.read()
                    self.send_response(resp.status, resp.reason)
                    for k, v in resp.getheaders():
                        if k.lower() in ("connection", "transfer-encoding",
                                         "content-length"):
                            continue
                        self.send_header(k, v)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except Exception as e:
                    logger.warning("noVNC proxy HTTP failed: %s", e)
                    self._novnc_unavailable(str(e))

            def _proxy_ws(self, upstream_path: str):
                # Read the client's handshake verbatim, bridge it to the
                # upstream websockify, then shuttle bytes both ways.
                try:
                    head = self._read_handshake()
                except Exception as e:
                    logger.warning("noVNC proxy WS handshake read failed: %s",
                                   e)
                    return
                try:
                    up = socket.create_connection(
                        (server.upstream_host, server.upstream_port),
                        timeout=server.upstream_timeout_s)
                except OSError as e:
                    logger.warning("noVNC upstream unreachable: %s", e)
                    try:
                        self.send_error(502, "noVNC upstream unavailable; "
                                       "start websockify on the server")
                    except OSError:
                        pass
                    return
                try:
                    up.sendall(head)
                    reply = self._read_handshake_raw(up)
                    # Split cleanly: headers to the client now, any pipelined
                    # post-handshake bytes into the bridge (never dropped).
                    cut = reply.find(b"\r\n\r\n")
                    head_out = reply if cut == -1 else reply[:cut + 4]
                    remainder = b"" if cut == -1 else reply[cut + 4:]
                    self.connection.sendall(head_out)
                    status_line = head_out.split(b"\r\n", 1)[0]
                    if b" 101 " not in status_line:
                        logger.warning(
                            "noVNC upstream refused WS upgrade: %.120s",
                            status_line)
                        up.close()
                        return
                    logger.info("noVNC WS bridged: %s (%d pipelined bytes)",
                                upstream_path[:80], len(remainder))
                    self._bridge_sockets(self.connection, up,
                                         preloaded=remainder)
                except OSError:
                    pass
                finally:
                    try:
                        up.close()
                    except OSError:
                        pass

            def _read_handshake(self) -> bytes:
                # Reconstruct from parsed request + headers (exact bytes not
                # required by websockify).
                lines = [f"{self.command} {self.path} {self.request_version}"]
                for k, v in self.headers.items():
                    if k.lower() == "host":
                        lines.append(
                            f"Host: {server.upstream_host}:"
                            f"{server.upstream_port}")
                    else:
                        lines.append(f"{k}: {v}")
                return ("\r\n".join(lines) + "\r\n\r\n").encode()

            @staticmethod
            def _read_handshake_raw(sock: socket.socket) -> bytes:
                buf = b""
                while b"\r\n\r\n" not in buf:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                    if len(buf) > 65536:
                        break
                return buf

            @staticmethod
            def _bridge_sockets(a: socket.socket, b: socket.socket,
                                preloaded: bytes = b"") -> None:
                # preloaded bytes came FROM upstream, counted b_to_a.
                stats = {"a_to_b": 0, "b_to_a": len(preloaded),
                         "first_close": None}

                def _pump(src, dst, key, other_key):
                    try:
                        while True:
                            chunk = src.recv(65536)
                            if not chunk:
                                if stats["first_close"] is None:
                                    stats["first_close"] = key
                                break
                            dst.sendall(chunk)
                            stats[key] += len(chunk)
                    except OSError as e:
                        if stats["first_close"] is None:
                            stats["first_close"] = f"{key}({e})"
                    finally:
                        for s in (src, dst):
                            try:
                                s.shutdown(socket.SHUT_RDWR)
                            except OSError:
                                pass

                if preloaded:
                    try:
                        a.sendall(preloaded)
                    except OSError:
                        pass
                t = threading.Thread(target=_pump, args=(b, a, "b_to_a",
                                                         "a_to_b"),
                                     daemon=True)
                t.start()
                _pump(a, b, "a_to_b", "b_to_a")
                t.join(timeout=5)
                logger.info("noVNC WS bridge closed: client->up %dB, "
                            "up->client %dB, first_close=%s",
                            stats["a_to_b"], stats["b_to_a"],
                            stats["first_close"])

            def _novnc_unavailable(self, detail: str) -> None:
                body = json.dumps({
                    "error": "noVNC upstream unavailable",
                    "detail": detail[:200],
                    "hint": "start Xvfb + headed Chromium + websockify on "
                            "the server (see scripts/run_dashboard.py --help)",
                }).encode()
                try:
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:
                    pass

        return Handler

    def serve_in_background(self) -> threading.Thread:
        """Serve HTTP/WS/proxy on a daemon thread.

        The agent (Playwright sync API is thread-affine: it must run on the
        thread that launched the browser, i.e. the main thread) runs in the
        caller afterwards. HITL waits block only the agent thread; the
        server threads stay live.
        """
        thread = threading.Thread(target=self.serve_forever, daemon=True,
                                  name="dashboard-server")
        thread.start()
        return thread

    def run_agent_foreground(self, agent_fn: Callable[[], Dict],
                             task_desc: str = "") -> Dict[str, Any]:
        """Run the agent on THIS thread (Playwright thread-affinity), then
        keep serving the dashboard until Ctrl-C so chat/noVNC stay usable."""
        self.serve_in_background()
        try:
            self.bridge.set_status("RUNNING",
                                   f"agent started: {task_desc}"[:200])
            self.bridge.publish_event("task_started", task_desc[:200])
            outcome = agent_fn()
        except KeyboardInterrupt:
            raise
        except Exception:
            logger.exception("Dashboard agent crashed")
            try:
                self.bridge.set_status("ERROR", "agent crashed (see logs)")
                self.bridge.publish_event("task_failed", "agent crashed")
            except Exception:
                pass
            outcome = {"success": False, "reason": "agent crashed (see logs)"}
        ok = bool((outcome or {}).get("success", False))
        try:
            self.bridge.set_status("COMPLETED" if ok else "ERROR",
                                   str((outcome or {}).get(
                                       "reason", ""))[:200])
            self.bridge.publish_event(
                "task_finished" if ok else "task_failed",
                str((outcome or {}).get("reason", ""))[:200])
        except Exception:
            pass
        logger.info("Agent finished; dashboard still serving "
                    "(Ctrl-C to stop)")
        try:
            # Block in the main thread until the operator stops us.
            import time as _time
            while True:
                _time.sleep(3600)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()
        return outcome

    # -- chat message dispatch (unit-testable without sockets) --------------
    def _handle_chat_message(self, send, raw: bytes) -> None:
        try:
            msg = json.loads(raw.decode("utf-8"))
        except Exception:
            send({"type": "ERROR", "message": "invalid JSON"})
            return
        if not isinstance(msg, dict):
            send({"type": "ERROR", "message": "message must be an object"})
            return
        mtype = msg.get("type")
        if mtype == "PING":
            send({"type": "PONG"})
        elif mtype == "HITL_RESPONSE":
            ok = self.bridge.resolve(msg.get("request_id", ""),
                                     msg.get("value", ""))
            if not ok:
                send({"type": "ERROR",
                      "message": "unknown or expired request_id"})
        elif mtype == "STATUS_REQUEST":
            snap = self.bridge.status_snapshot()
            send({"type": "STATUS", "state": snap["state"],
                  "message": snap["message"]})
        else:
            send({"type": "ERROR",
                  "message": f"unknown message type {mtype!r}"[:120]})
