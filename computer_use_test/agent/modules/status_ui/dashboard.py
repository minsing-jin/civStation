"""
Dashboard — Single-page HTML/CSS/JS for the real-time status UI.

Supports WebSocket (primary) with automatic HTTP polling fallback.
"""

DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Civ6 Agent Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #1a1a2e;
    color: #e0e0e0;
    min-height: 100vh;
  }
  header {
    background: #16213e;
    padding: 12px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 2px solid #0f3460;
    flex-wrap: wrap;
    gap: 8px;
  }
  header h1 { font-size: 1.2rem; color: #e94560; }
  .header-right {
    display: flex; align-items: center; gap: 16px;
  }
  .conn-info {
    display: flex; align-items: center; gap: 6px; font-size: 0.82rem;
  }
  #conn-label {
    padding: 2px 8px; border-radius: 3px; font-weight: 600; font-size: 0.75rem;
  }
  .conn-ws { background: #4caf50; color: #000; }
  .conn-http { background: #f0a500; color: #000; }
  .turn-bar {
    display: flex; gap: 16px; font-size: 0.9rem;
  }
  .turn-bar span { padding: 4px 12px; background: #0f3460; border-radius: 4px; }
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    padding: 16px;
    max-width: 1200px;
    margin: 0 auto;
  }
  @media (max-width: 768px) {
    .grid { grid-template-columns: 1fr; }
    header { padding: 10px 12px; }
    .turn-bar { font-size: 0.8rem; gap: 8px; }
  }
  .panel {
    background: #16213e;
    border-radius: 8px;
    padding: 16px;
    border: 1px solid #0f3460;
  }
  .panel h2 {
    font-size: 0.95rem;
    color: #e94560;
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .panel p, .panel li {
    font-size: 0.88rem;
    line-height: 1.5;
    color: #ccc;
  }
  .panel ul { list-style: none; }
  .panel ul li { padding: 3px 0; border-bottom: 1px solid #0f346033; }
  .panel ul li:last-child { border-bottom: none; }
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 0.78rem;
    font-weight: 600;
  }
  .badge-strategy { background: #533483; }
  .badge-stop { background: #e94560; }
  .badge-pause { background: #f0a500; color: #000; }
  .badge-custom { background: #0f3460; }
  .badge-override { background: #00bcd4; color: #000; }
  .full-width { grid-column: 1 / -1; }
  .input-row {
    display: flex;
    gap: 8px;
    margin-top: 12px;
  }
  .input-row input {
    flex: 1;
    padding: 8px 12px;
    border: 1px solid #0f3460;
    border-radius: 4px;
    background: #1a1a2e;
    color: #e0e0e0;
    font-size: 0.88rem;
  }
  .input-row button {
    padding: 8px 16px;
    background: #e94560;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-weight: 600;
  }
  .input-row button:hover { background: #c73652; }
  .action-row {
    display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap;
  }
  .action-row button {
    padding: 6px 12px;
    border: 1px solid #0f3460;
    background: #1a1a2e;
    color: #e0e0e0;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.82rem;
  }
  .action-row button:hover { background: #0f3460; }
  .mode-toggle {
    display: flex; gap: 0; margin-top: 8px;
  }
  .mode-toggle button {
    padding: 6px 14px;
    border: 1px solid #0f3460;
    background: #1a1a2e;
    color: #888;
    cursor: pointer;
    font-size: 0.82rem;
    transition: all 0.2s;
  }
  .mode-toggle button:first-child { border-radius: 4px 0 0 4px; }
  .mode-toggle button:last-child { border-radius: 0 4px 4px 0; }
  .mode-toggle button.active {
    background: #0f3460;
    color: #e0e0e0;
    font-weight: 600;
  }
  .reasoning {
    margin-top: 8px;
    padding: 8px;
    background: #1a1a2e;
    border-radius: 4px;
    font-size: 0.82rem;
    color: #aaa;
    max-height: 100px;
    overflow-y: auto;
  }
  #status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-left: 8px;
  }
  .dot-ok { background: #4caf50; }
  .dot-err { background: #e94560; }
  #agent-phase {
    font-size: 0.82rem;
    color: #aaa;
    margin-top: 4px;
    font-style: italic;
  }
  /* QR Code */
  .qr-btn {
    padding: 6px 12px;
    background: #0f3460;
    color: #e0e0e0;
    border: 1px solid #533483;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.82rem;
    font-weight: 600;
    white-space: nowrap;
  }
  .qr-btn:hover { background: #533483; }
  .modal-overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.7);
    z-index: 1000;
    justify-content: center;
    align-items: center;
  }
  .modal-overlay.show { display: flex; }
  .modal-box {
    background: #16213e;
    border: 2px solid #0f3460;
    border-radius: 12px;
    padding: 28px;
    text-align: center;
    max-width: 360px;
    width: 90%;
  }
  .modal-box h3 {
    color: #e94560;
    margin-bottom: 16px;
    font-size: 1rem;
  }
  .modal-box .qr-canvas {
    background: #fff;
    border-radius: 8px;
    padding: 16px;
    display: inline-block;
    margin-bottom: 12px;
  }
  .modal-box .url-text {
    font-size: 0.88rem;
    color: #4caf50;
    word-break: break-all;
    padding: 8px;
    background: #1a1a2e;
    border-radius: 4px;
    margin-bottom: 12px;
    user-select: all;
    cursor: pointer;
  }
  .modal-box .url-hint {
    font-size: 0.75rem;
    color: #888;
    margin-bottom: 12px;
  }
  .modal-box button {
    padding: 8px 24px;
    background: #e94560;
    color: #fff;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-weight: 600;
  }
  .modal-box button:hover { background: #c73652; }
</style>
</head>
<body>
<header>
  <div>
    <h1>Civ6 Agent Dashboard <span id="status-dot" class="dot-ok"></span></h1>
    <div id="agent-phase"></div>
  </div>
  <div class="header-right">
    <button class="qr-btn" onclick="showQR()">QR Connect</button>
    <div class="conn-info">
      <span id="conn-label" class="conn-http">HTTP</span>
    </div>
    <div class="turn-bar">
      <span>Game Turn: <strong id="game-turn">-</strong></span>
      <span>Micro: <strong id="micro-turn">-</strong></span>
      <span>Macro: <strong id="macro-turn">-</strong></span>
    </div>
  </div>
</header>
<div class="grid">
  <!-- Strategy Panel -->
  <div class="panel">
    <h2>Current Strategy</h2>
    <p id="strategy-text">-</p>
    <p style="margin-top:6px;font-size:0.82rem;color:#888">
      Victory: <strong id="victory-goal">-</strong> | Phase: <strong id="game-phase">-</strong>
    </p>
  </div>
  <!-- Queue Panel -->
  <div class="panel">
    <h2>Queued Directives</h2>
    <ul id="queue-list"><li>No directives queued</li></ul>
  </div>
  <!-- Current Action Panel -->
  <div class="panel">
    <h2>Current Action</h2>
    <p>Primitive: <strong id="cur-primitive">-</strong></p>
    <p>Action: <strong id="cur-action">-</strong></p>
    <div class="reasoning" id="cur-reasoning">-</div>
  </div>
  <!-- Recent Actions Panel -->
  <div class="panel">
    <h2>Recent Actions</h2>
    <ul id="recent-list"><li>No actions yet</li></ul>
  </div>
  <!-- Input Panel -->
  <div class="panel full-width">
    <h2>Send Directive</h2>
    <div class="mode-toggle">
      <button id="mode-hl" class="active" onclick="setMode('high_level')">Strategy</button>
      <button id="mode-prim" onclick="setMode('primitive')">Primitive</button>
    </div>
    <div class="input-row">
      <input type="text" id="directive-input" placeholder="Enter new strategy or command...">
      <button onclick="sendDirective()">Send</button>
    </div>
    <div class="action-row">
      <button onclick="sendQuick('pause')">Pause</button>
      <button onclick="sendQuick('resume')">Resume</button>
      <button onclick="sendQuick('stop')">Stop</button>
    </div>
  </div>
</div>
<!-- QR Modal -->
<div class="modal-overlay" id="qr-modal" onclick="hideQR(event)">
  <div class="modal-box">
    <h3>Mobile Connect</h3>
    <div class="qr-canvas" id="qr-canvas"></div>
    <div class="url-text" id="qr-url">Loading...</div>
    <div class="url-hint">Same Wi-Fi network required</div>
    <button onclick="hideQR()">Close</button>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/qrcode-generator@1.4.4/qrcode.min.js"></script>
<script>
const POLL_MS = 1500;
let ws = null;
let pollTimer = null;
let currentMode = 'high_level';

function badgeClass(t) {
  if (t === 'change_strategy') return 'badge-strategy';
  if (t === 'stop') return 'badge-stop';
  if (t === 'pause') return 'badge-pause';
  if (t === 'primitive_override') return 'badge-override';
  return 'badge-custom';
}

function updateDashboard(d) {
  document.getElementById('game-turn').textContent = d.game_turn;
  document.getElementById('micro-turn').textContent = d.micro_turn;
  document.getElementById('macro-turn').textContent = d.macro_turn;
  document.getElementById('strategy-text').textContent = d.current_strategy || '-';
  document.getElementById('victory-goal').textContent = d.victory_goal || '-';
  document.getElementById('game-phase').textContent = d.game_phase || '-';
  document.getElementById('cur-primitive').textContent = d.current_primitive || '-';
  document.getElementById('cur-action').textContent = d.current_action || '-';
  document.getElementById('cur-reasoning').textContent = d.current_reasoning || '-';

  const ql = document.getElementById('queue-list');
  if (d.queued_directives && d.queued_directives.length) {
    ql.innerHTML = d.queued_directives.map(q =>
      `<li><span class="badge ${badgeClass(q.type)}">${q.type}</span> ${q.payload} <em>(${q.source})</em></li>`
    ).join('');
  } else {
    ql.innerHTML = '<li>No directives queued</li>';
  }

  const rl = document.getElementById('recent-list');
  if (d.recent_actions && d.recent_actions.length) {
    rl.innerHTML = d.recent_actions.map(a =>
      `<li>${a.type} (${a.x}, ${a.y}) [${a.primitive}] — ${a.result}</li>`
    ).join('');
  } else {
    rl.innerHTML = '<li>No actions yet</li>';
  }
}

function updatePhase(phase) {
  document.getElementById('agent-phase').textContent = phase;
}

/* --- WebSocket connection --- */
let pingTimer = null;

function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);

  ws.onopen = () => {
    document.getElementById('status-dot').className = 'dot-ok';
    document.getElementById('conn-label').textContent = 'WS';
    document.getElementById('conn-label').className = 'conn-ws';
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    // Client-side keepalive: send ping every 15s so proxies don't drop the connection
    if (pingTimer) clearInterval(pingTimer);
    pingTimer = setInterval(() => {
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({mode: 'ping'}));
    }, 15000);
  };

  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'status') updateDashboard(msg.data);
    else if (msg.type === 'phase') updatePhase(msg.phase);
    /* pong / unknown types are silently ignored */
  };

  ws.onclose = () => {
    if (pingTimer) { clearInterval(pingTimer); pingTimer = null; }
    document.getElementById('status-dot').className = 'dot-err';
    document.getElementById('conn-label').textContent = 'HTTP';
    document.getElementById('conn-label').className = 'conn-http';
    if (!pollTimer) pollTimer = setInterval(poll, POLL_MS);
    setTimeout(connectWS, 3000);
  };

  ws.onerror = () => { /* onclose will fire after this */ };
}

/* --- HTTP polling fallback --- */
async function poll() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    updateDashboard(d);
    document.getElementById('status-dot').className = 'dot-ok';
  } catch {
    document.getElementById('status-dot').className = 'dot-err';
  }
}

/* --- Mode toggle --- */
function setMode(mode) {
  currentMode = mode;
  document.getElementById('mode-hl').className = mode === 'high_level' ? 'active' : '';
  document.getElementById('mode-prim').className = mode === 'primitive' ? 'active' : '';
  const input = document.getElementById('directive-input');
  if (mode === 'primitive') {
    input.placeholder = 'Enter primitive JSON: {"action":"click","x":500,"y":300}';
  } else {
    input.placeholder = 'Enter new strategy or command...';
  }
}

/* --- Send directive: WS preferred, HTTP fallback --- */
function sendDirective() {
  const input = document.getElementById('directive-input');
  const text = input.value.trim();
  if (!text) return;

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({mode: currentMode, content: text}));
  } else {
    if (currentMode === 'primitive') {
      fetch('/api/directive', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: text, mode: 'primitive'})
      });
    } else {
      fetch('/api/directive', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: text})
      });
    }
  }
  input.value = '';
}

function sendQuick(cmd) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({mode: cmd, content: ''}));
  } else {
    fetch('/api/directive', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: cmd})
    });
  }
}

document.getElementById('directive-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') sendDirective();
});

/* --- QR Code --- */
let qrGenerated = false;

async function showQR() {
  document.getElementById('qr-modal').classList.add('show');
  if (qrGenerated) return;

  try {
    const r = await fetch('/api/connection-info');
    const info = await r.json();
    const url = info.url;

    document.getElementById('qr-url').textContent = url;

    if (typeof qrcode === 'function') {
      const qr = qrcode(0, 'M');
      qr.addData(url);
      qr.make();
      document.getElementById('qr-canvas').innerHTML = qr.createSvgTag(5, 0);
    } else {
      document.getElementById('qr-canvas').innerHTML =
        '<p style="color:#e94560;font-size:0.82rem">QR library failed to load</p>';
    }
    qrGenerated = true;
  } catch {
    document.getElementById('qr-url').textContent = location.href;
    document.getElementById('qr-canvas').innerHTML =
      '<p style="color:#e94560;font-size:0.82rem">Could not load connection info</p>';
  }
}

function hideQR(e) {
  if (!e || e.target === document.getElementById('qr-modal')) {
    document.getElementById('qr-modal').classList.remove('show');
  }
}

/* Start: try WebSocket first, polling as fallback */
pollTimer = setInterval(poll, POLL_MS);
poll();
connectWS();
</script>
</body>
</html>
"""
