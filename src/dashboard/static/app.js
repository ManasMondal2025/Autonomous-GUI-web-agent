/* Agent Dashboard frontend: chat + status + noVNC embed. No framework. */
(function () {
  "use strict";

  var chat = document.getElementById("chat");
  var form = document.getElementById("answer-form");
  var input = document.getElementById("answer-input");
  var captchaBtn = document.getElementById("captcha-solved");
  var pill = document.getElementById("status-pill");
  var conn = document.getElementById("conn");
  var novncFrame = document.getElementById("novnc");
  var novncFallback = document.getElementById("novnc-fallback");
  var novncHint = document.getElementById("novnc-hint");

  var ws = null;
  var backoffMs = 1000;
  var activeHitl = null;    // {request_id}
  var activeCaptcha = null; // {request_id}
  var answered = {};        // request_id -> true (duplicate-submit guard)

  function ts() {
    return new Date().toLocaleTimeString();
  }

  function addMsg(cls, text) {
    var div = document.createElement("div");
    div.className = "msg " + cls;
    var span = document.createElement("span");
    span.textContent = text;
    div.appendChild(span);
    var t = document.createElement("span");
    t.className = "ts";
    t.textContent = ts();
    div.appendChild(t);
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
  }

  var STATE_LABEL = {
    IDLE: ["⚪ Idle", "idle"],
    RUNNING: ["🟢 Running", "running"],
    EXECUTING: ["🔵 Executing", "running"],
    WAITING_FOR_USER: ["🟡 Waiting for user", "waiting"],
    CAPTCHA_DETECTED: ["🟠 CAPTCHA detected", "captcha"],
    ERROR: ["🔴 Error", "error"],
    COMPLETED: ["✓ Completed", "done"]
  };

  function setStatus(state, message) {
    var entry = STATE_LABEL[state] || [state, "idle"];
    pill.textContent = entry[0] + (message ? " — " + message : "");
    pill.className = "pill " + entry[1];
  }

  function setFormEnabled(on, placeholder) {
    input.disabled = !on;
    form.querySelector("button").disabled = !on;
    if (placeholder) input.placeholder = placeholder;
    if (on) input.focus();
  }

  function showHitlInput(req) {
    activeHitl = req;
    addMsg("human-q", "🤖 " + req.question);
    setFormEnabled(true, "Type your answer…");
  }

  function hideHitlInput() {
    activeHitl = null;
    setFormEnabled(false, "Waiting for the agent…");
  }

  function showCaptcha(req) {
    activeCaptcha = req;
    addMsg("captcha", "🤖 ⚠ CAPTCHA detected.\n\n" + req.message +
      "\n\nPlease solve the CAPTCHA directly in the browser (left).");
    captchaBtn.classList.remove("hidden");
  }

  function hideCaptcha() {
    activeCaptcha = null;
    captchaBtn.classList.add("hidden");
  }

  form.addEventListener("submit", function (ev) {
    ev.preventDefault();
    if (!activeHitl || !ws) return;
    var reqId = activeHitl.request_id;
    if (answered[reqId]) return; // duplicate-submit guard
    var value = input.value;
    answered[reqId] = true;
    ws.send(JSON.stringify({ type: "HITL_RESPONSE", request_id: reqId, value: value }));
    addMsg("user", value || "(empty)");
    input.value = "";
    hideHitlInput();
  });

  captchaBtn.addEventListener("click", function () {
    if (!activeCaptcha || !ws) return;
    var reqId = activeCaptcha.request_id;
    if (answered[reqId]) return;
    answered[reqId] = true;
    ws.send(JSON.stringify({ type: "HITL_RESPONSE", request_id: reqId, value: "completed" }));
    addMsg("user", "✓ CAPTCHA marked solved — resuming.");
    hideCaptcha();
  });

  function onMessage(msg) {
    if (!msg || typeof msg !== "object") return;
    if (msg.type === "STATUS") {
      setStatus(msg.state, msg.message);
    } else if (msg.type === "AGENT_EVENT") {
      addMsg("event", "• " + (msg.message || msg.event));
      if (msg.event === "task_finished") setStatus("COMPLETED", msg.message);
      if (msg.event === "task_failed") setStatus("ERROR", msg.message);
    } else if (msg.type === "HITL_REQUEST") {
      setStatus("WAITING_FOR_USER", msg.question);
      showHitlInput(msg);
    } else if (msg.type === "CAPTCHA_REQUEST") {
      setStatus("CAPTCHA_DETECTED", msg.message);
      showCaptcha({ request_id: msg.request_id, message: msg.message });
    } else if (msg.type === "ERROR") {
      addMsg("error", "⚠ " + msg.message);
    }
  }

  function connect() {
    var proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(proto + "//" + location.host + "/ws");
    ws.onopen = function () {
      conn.textContent = "🟢 connected";
      backoffMs = 1000;
    };
    ws.onmessage = function (ev) {
      try { onMessage(JSON.parse(ev.data)); } catch (e) { /* ignore */ }
    };
    ws.onclose = function () {
      conn.textContent = "🔌 reconnecting…";
      ws = null;
      setTimeout(connect, backoffMs);
      backoffMs = Math.min(backoffMs * 2, 10000);
    };
    ws.onerror = function () { try { ws.close(); } catch (e) { /* ignore */ } };
  }

  function setupNovnc() {
    fetch("/api/config").then(function (r) { return r.json(); }).then(function (cfg) {
      novncFrame.src = cfg.novnc_iframe;
      novncHint.textContent = "If this panel stays black, the noVNC upstream is down. " +
        "Start Xvfb + headed Chromium + websockify on the server, or open it directly: " +
        (cfg.novnc_direct_hint || "");
      // If the iframe never paints anything useful we still keep it (RFB is
      // binary; load events don't prove health). The Retry button reloads it.
      setTimeout(function () {
        fetch("/api/status").then(function (r) { return r.json(); }).then(function (st) {
          if (!st.novnc_reachable) novncFallback.classList.remove("hidden");
        }).catch(function () { /* status endpoint is best-effort */ });
      }, 4000);
    }).catch(function () {
      novncFallback.classList.remove("hidden");
    });
    document.getElementById("novnc-retry").addEventListener("click", function () {
      novncFallback.classList.add("hidden");
      novncFrame.src = novncFrame.src;
    });
  }

  addMsg("agent", "🤖 GUI Web Agent dashboard ready. Start a run with scripts/run_dashboard.py.");
  setupNovnc();
  connect();
})();
