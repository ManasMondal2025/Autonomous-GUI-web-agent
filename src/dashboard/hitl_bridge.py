"""Dashboard HITL bridge: agent ask_user <-> chat WebSocket.

The agent thread blocks on a per-request ``threading.Event``; the HTTP/WS
server threads keep serving (noVNC, status, other sockets) independently,
so only the agent execution waits -- never the server loop.

Underlying HITL semantics are unchanged: same max_asks budget (enforced by
the controller), same memory/trajectory logging, same STV bypass. This
module only changes the TRANSPORT (chat instead of stdin).

Protocol (JSON over /ws):
  -> {"type": "HITL_REQUEST", "request_id, "question", "input_type": "text"}
  -> {"type": "CAPTCHA_REQUEST", "request_id", "message", "action": "user_required"}
  <- {"type": "HITL_RESPONSE", "request_id", "value": "..."}
  <-> {"type": "AGENT_EVENT", "event", "message"}
  <-> {"type": "STATUS", "state", "message"}
  <- {"type": "PONG"} (heartbeat answer; server sends PING)
"""

import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from agent.captcha_gate import CAPTCHA_REASON
except Exception:  # keep stdlib-only import surface; match the literal
    CAPTCHA_REASON = "captcha_detected"

HITL_TIMEOUT_S_DEFAULT = 600.0


class DashboardHub:
    """Thread-safe WebSocket client registry + broadcast fan-out."""

    def __init__(self):
        self._lock = threading.Lock()
        self._clients: Dict[int, Callable[[dict], bool]] = {}
        self._next_id = 0

    def register(self, send: Callable[[dict], bool]) -> int:
        with self._lock:
            self._next_id += 1
            cid = self._next_id
            self._clients[cid] = send
            return cid

    def unregister(self, cid: int) -> None:
        with self._lock:
            self._clients.pop(cid, None)

    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    def broadcast(self, message: dict) -> int:
        """Best-effort fan-out. Returns number of clients that accepted."""
        with self._lock:
            clients = list(self._clients.items())
        delivered = 0
        for cid, send in clients:
            try:
                if send(message) is not False:
                    delivered += 1
            except Exception as e:
                logger.warning("Hub drop client %s (%s)", cid, e)
                self.unregister(cid)
        return delivered


class _Pending:
    def __init__(self, request_id: str, kind: str, question: str):
        self.request_id = request_id
        self.kind = kind  # "hitl" | "captcha"
        self.question = question
        self.created = time.time()
        self.event = threading.Event()
        self.response: Optional[str] = None
        self.answered = False


class DashboardHITLBridge:
    """ask_handler implementation backed by the dashboard chat."""

    def __init__(self, hub: Optional[DashboardHub] = None,
                 hitl_timeout_s: float = HITL_TIMEOUT_S_DEFAULT,
                 input_fallback: bool = True):
        self.hub = hub or DashboardHub()
        self.hitl_timeout_s = hitl_timeout_s
        self.input_fallback = input_fallback
        self._lock = threading.Lock()
        self._pending: Dict[str, _Pending] = {}
        self._recently_answered: List[str] = []  # bounded id log for dup acks
        self._state = "IDLE"
        self._state_message = ""

    # -- status / events -------------------------------------------------
    def set_status(self, state: str, message: str = "") -> None:
        with self._lock:
            self._state = state
            self._state_message = message
        self.hub.broadcast({"type": "STATUS", "state": state,
                            "message": message})
        logger.info("[DASHBOARD] status=%s %s", state, message[:120])

    def publish_event(self, event: str, message: str = "") -> None:
        self.hub.broadcast({"type": "AGENT_EVENT", "event": event,
                            "message": message})

    def status_snapshot(self) -> dict:
        with self._lock:
            pending = [{"request_id": p.request_id, "kind": p.kind,
                        "question": p.question,
                        "waiting_s": round(time.time() - p.created, 1)}
                       for p in self._pending.values() if not p.answered]
            return {"state": self._state, "message": self._state_message,
                    "clients": self.hub.client_count(),
                    "pending": pending}

    # -- ask_handler API (called on the AGENT thread) ---------------------
    def for_action(self, action_dict: Optional[dict]):
        """Bound handler carrying the action context (CAPTCHA typing).

        The controller calls ``handler(question)``; plain functions have no
        such hook, so this is backwards compatible (see controller).
        """
        reason = ""
        try:
            reason = str((action_dict or {}).get("reason", ""))
        except Exception:
            reason = ""
        kind = "captcha" if reason == CAPTCHA_REASON else "hitl"

        def _bound(question: str) -> str:
            return self._ask(str(question or ""), kind)

        return _bound

    def ask_handler(self, question: str) -> str:
        return self._ask(str(question or ""), "hitl")

    def __call__(self, question: str) -> str:
        """Allow the bridge itself to serve as controller.ask_handler."""
        return self.ask_handler(question)

    def _ask(self, question: str, kind: str) -> str:
        request_id = uuid.uuid4().hex[:12]
        pending = _Pending(request_id, kind, question)
        with self._lock:
            self._pending[request_id] = pending
        if self.hub.client_count() == 0 and self.input_fallback:
            # No dashboard open: keep headless/CI behavior (terminal).
            logger.info("[DASHBOARD] no chat clients; terminal fallback "
                        "for %s", request_id)
            with self._lock:
                self._pending.pop(request_id, None)
            return self._terminal_ask(question)
        if kind == "captcha":
            self.hub.broadcast({
                "type": "CAPTCHA_REQUEST", "request_id": request_id,
                "message": question, "action": "user_required"})
            self.set_status("CAPTCHA_DETECTED",
                            "CAPTCHA detected — solve it in the browser")
        else:
            self.hub.broadcast({
                "type": "HITL_REQUEST", "request_id": request_id,
                "question": question, "input_type": "text"})
            self.set_status("WAITING_FOR_USER", question[:120])
        logger.info("[DASHBOARD] HITL request created: %s (%s)",
                    request_id, kind)
        logger.info("[DASHBOARD] Waiting for user response")
        answered = pending.event.wait(timeout=self.hitl_timeout_s)
        with self._lock:
            self._pending.pop(request_id, None)
            self._recently_answered.append(request_id)
            del self._recently_answered[:-100]
        if not answered or pending.response is None:
            logger.warning("[DASHBOARD] HITL request %s timed out; "
                           "recording empty answer", request_id)
            self.publish_event("hitl_timeout",
                               f"Request {request_id} timed out")
            self.set_status("RUNNING", "HITL timed out; agent continuing")
            return ""
        logger.info("[DASHBOARD] User response received for %s", request_id)
        self.set_status("RUNNING", "Answer received; agent continuing")
        self.publish_event("hitl_answered", f"Request {request_id} answered")
        return pending.response

    @staticmethod
    def _terminal_ask(question: str) -> str:
        print(f"\n[Agent Needs Help]: {question}")
        try:
            return input("Your response: ")
        except (EOFError, KeyboardInterrupt):
            return ""

    # -- WebSocket side (called on server threads) -------------------------
    def resolve(self, request_id: str, value: Any) -> bool:
        """Accept a HITL_RESPONSE. Unknown ids are rejected; duplicates are
        acknowledged but never overwrite the first answer."""
        with self._lock:
            pending = self._pending.get(str(request_id or ""))
            if pending is None and str(request_id or "") in \
                    self._recently_answered:
                logger.warning("[DASHBOARD] duplicate response for answered "
                               "%s; first answer kept", str(request_id)[:12])
                return True
        if pending is None:
            logger.warning("[DASHBOARD] unknown HITL request_id %r; ignored",
                           str(request_id or "")[:32])
            return False
        with self._lock:
            if pending.answered:
                logger.warning("[DASHBOARD] duplicate response for %s; "
                               "first answer kept", pending.request_id)
                return True
            pending.answered = True
            pending.response = "" if value is None else str(value)
            pending.event.set()
        return True

    def active_requests(self) -> List[dict]:
        with self._lock:
            return [{"request_id": p.request_id, "kind": p.kind,
                     "question": p.question}
                    for p in self._pending.values() if not p.answered]
