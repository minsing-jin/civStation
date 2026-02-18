"""
Dashboard — Single-page HTML/CSS/JS for the real-time status UI.
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
  }
  header h1 { font-size: 1.2rem; color: #e94560; }
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
    display: flex; gap: 6px; margin-top: 8px;
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
</style>
</head>
<body>
<header>
  <h1>Civ6 Agent Dashboard <span id="status-dot" class="dot-ok"></span></h1>
  <div class="turn-bar">
    <span>Game Turn: <strong id="game-turn">-</strong></span>
    <span>Micro: <strong id="micro-turn">-</strong></span>
    <span>Macro: <strong id="macro-turn">-</strong></span>
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
<script>
const POLL_MS = 1500;

function badgeClass(t) {
  if (t === 'change_strategy') return 'badge-strategy';
  if (t === 'stop') return 'badge-stop';
  if (t === 'pause') return 'badge-pause';
  return 'badge-custom';
}

async function poll() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
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

    document.getElementById('status-dot').className = 'dot-ok';
  } catch {
    document.getElementById('status-dot').className = 'dot-err';
  }
}

async function sendDirective() {
  const input = document.getElementById('directive-input');
  const text = input.value.trim();
  if (!text) return;
  await fetch('/api/directive', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: text})
  });
  input.value = '';
}

async function sendQuick(cmd) {
  await fetch('/api/directive', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: cmd})
  });
}

document.getElementById('directive-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') sendDirective();
});

setInterval(poll, POLL_MS);
poll();
</script>
</body>
</html>
"""
