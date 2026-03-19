# Civ6 Computer Use Agent

An autonomous Civilization VI game agent powered by Vision-Language Models (VLMs).
Captures screenshots, classifies game state via routing, then dispatches to specialized primitives that generate normalized-coordinate actions executed through PyAutoGUI.

---

## Architecture

### Core Flow (One Turn)

```
Screenshot Capture
    │
    ▼
Router VLM ─── Classify Game State ──▶ Select Primitive
    │
    ▼
Planner VLM ─── Generate Action (normalized coords 0-1000)
    │
    ▼
execute_action() ─── Convert Coords ──▶ PyAutoGUI Execution
```

### Project Structure

```
computer_use_test/
├── agent/
│   ├── turn_runner.py              # CLI entry point (ConfigArgParse + config.yaml)
│   ├── turn_executor.py            # run_one_turn / run_multi_turn execution logic
│   └── modules/
│       ├── router/
│       │   └── primitive_registry.py       # Central primitive registry (auto-generates router prompt)
│       ├── primitive/                      # Primitive execution logic
│       ├── context/
│       │   ├── context_manager.py          # Game state context tracking
│       │   ├── context_updater.py          # Background screenshot analysis
│       │   └── macro_turn_manager.py       # In-game turn boundary detection
│       ├── strategy/
│       │   └── strategy_planner.py         # Strategy generation / HITL refinement
│       ├── knowledge/
│       │   ├── knowledge_manager.py        # Orchestrates document + web retrieval
│       │   ├── document_retriever.py       # Civopedia JSON index search
│       │   └── web_search_retriever.py     # Tavily web search integration
│       └── hitl/
│           ├── command_queue.py            # Thread-safe directive queue
│           ├── agent_gate.py              # External control gate (start/stop/pause)
│           ├── queue_listener.py          # Stdin listener
│           ├── status_ui/
│           │   ├── server.py              # FastAPI server (REST + WS + Discussion)
│           │   ├── state_bridge.py        # Agent ↔ UI thread bridge
│           │   ├── websocket_manager.py   # WS connection management + broadcast
│           │   ├── screen_streamer.py     # Real-time screen streaming via WS
│           │   └── dashboard.py           # Built-in HTML/JS dashboard
│           └── relay/
│               └── relay_client.py        # Remote HITL via external relay server
├── utils/
│   ├── llm_provider/               # Claude / Gemini / GPT / Mock providers
│   ├── screen.py                   # Screenshot capture + coord conversion (Retina-aware)
│   └── chatapp/
│       ├── discord_app.py          # Discord bot integration
│       ├── whatsapp_app.py         # WhatsApp bot integration
│       └── discussion/
│           ├── discussion_engine.py    # Multi-turn strategy discussion engine
│           └── discussion_schemas.py   # Session & message data models
└── evaluator/
    └── civ6/static_eval/           # Static evaluation framework (ground truth comparison)
```

### Primitive System

The router classifies each screenshot into one of 10 specialized primitives:

| Primitive | Responsibility |
|---|---|
| `unit_ops_primitive` | Unit movement, actions, and combat |
| `popup_primitive` | Popup / notification handling |
| `research_select_primitive` | Technology research selection |
| `city_production_primitive` | City production queue |
| `science_decision_primitive` | Tech tree decisions |
| `culture_decision_primitive` | Civics tree decisions |
| `governor_primitive` | Governor placement & promotion |
| `diplomatic_primitive` | Diplomacy interactions |
| `combat_primitive` | Dedicated combat situations |
| `policy_primitive` | Policy card management |

**Adding a new primitive:** Add an entry to `PRIMITIVE_REGISTRY` in `primitive_registry.py` — the router prompt, primitive names, and prompt lookup all auto-update.

### HITL (Human-in-the-Loop) System

```
External Controller (Web UI / Chat App / Relay)
    │  HTTP / WebSocket
    ▼
FastAPI Server (server.py)
    │
    ├── AgentGate ─── Start / Stop / Pause lifecycle
    │
    ├── CommandQueue ─── Directive queue
    │       │
    │       ▼
    │   turn_executor ─── Checks queue each turn
    │
    └── DiscussionEngine ─── Multi-turn strategy discussion via LLM
```

**Directive priority:** `STOP` > `PRIMITIVE_OVERRIDE` > `PAUSE` > `CHANGE_STRATEGY`

### Strategy Discussion

The discussion system enables real-time, multi-turn strategy conversations between the player and the LLM through the REST API. The LLM uses the current game context and strategy to provide advice.

- Supports multiple languages (`ko`, `en`, `ja`, `zh`) — set via the `language` field in requests
- Sessions are per-user and persist across multiple messages
- Finalization extracts a structured strategy (victory goal, phase, priorities) from the conversation

### Knowledge Retrieval

Optional modules that augment primitive decisions with external knowledge:

- **Document Retriever** — Searches a local Civopedia JSON index (`--knowledge-index`)
- **Web Search Retriever** — Queries Tavily for real-time game strategy info (`--enable-web-search`)

---

## Installation

```bash
# Install dependencies + pre-commit hooks
make install

# Or install directly
pip install -e ".[ui]"   # Includes FastAPI + uvicorn
```

### Environment Variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=sk-ant-...
GENAI_API_KEY=AIza...
OPENAI_API_KEY=sk-...          # Optional
DISCORD_BOT_TOKEN=...          # Optional, for Discord HITL
WHATSAPP_BOT_TOKEN=...         # Optional, for WhatsApp HITL
```

---

## Usage

### Basic Run

```bash
# Use config.yaml defaults
python -m computer_use_test.agent.turn_runner

# Specify via CLI
python -m computer_use_test.agent.turn_runner \
    --provider claude \
    --turns 20 \
    --strategy "Focus on science victory"
```

### Real-Time Dashboard

```bash
python -m computer_use_test.agent.turn_runner \
    --status-ui \
    --status-port 8765 \
    --turns 50
```

Open `http://localhost:8765` in a browser.

### Mobile QR Connect

1. Run the agent with `--status-ui`
2. Open `http://localhost:8765` on your PC browser
3. Click **"QR Connect"** in the top-right header
4. Scan the QR code with your phone (must be on the same Wi-Fi)

> The QR code auto-detects the server's LAN IP (`http://192.168.x.x:8765`).

### External Controller (Wait-for-Start Mode)

The agent starts the server first and waits for an external signal before executing turns:

```bash
python -m computer_use_test.agent.turn_runner \
    --status-ui \
    --wait-for-start \
    --turns 100
```

Control via HTTP:

```bash
curl -X POST http://localhost:8765/api/agent/start
curl -X POST http://localhost:8765/api/agent/pause
curl -X POST http://localhost:8765/api/agent/resume
curl -X POST http://localhost:8765/api/agent/stop
curl http://localhost:8765/api/agent/state
# → {"state": "running"}  # idle / running / paused / stopped
```

### Split Router / Planner Providers

```bash
python -m computer_use_test.agent.turn_runner \
    --router-provider gemini --router-model gemini-2.0-flash \
    --planner-provider claude --planner-model claude-sonnet-4-5
```

### Computer-Use Planner Providers

Use the normal vision provider for routing, and switch only the planner to a
computer-use provider for single-step action planning:

```bash
python -m computer_use_test.agent.turn_runner \
    --router-provider gemini --router-model gemini-2.0-flash \
    --planner-provider openai-computer --planner-model computer-use-preview

python -m computer_use_test.agent.turn_runner \
    --router-provider claude --router-model claude-4-5-sonnet-20241022 \
    --planner-provider anthropic-computer --planner-model claude-4-5-sonnet-20241022
```

Notes:

- `openai-computer` and `anthropic-computer` override only planner `analyze()` calls.
- Router classification and multi-action JSON flows still use the normal VLM path.
- Environment variables follow the existing vendor keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`.

### Chat App Integration (Discord / WhatsApp)

```bash
# Discord
python -m computer_use_test.agent.turn_runner \
    --chatapp discord \
    --discord-token $DISCORD_BOT_TOKEN \
    --discord-channel 123456789 \
    --enable-discussion

# WhatsApp
python -m computer_use_test.agent.turn_runner \
    --chatapp whatsapp \
    --whatsapp-token $WHATSAPP_BOT_TOKEN \
    --whatsapp-phone-number-id 123456
```

### Remote Relay (Headless HITL)

```bash
python -m computer_use_test.agent.turn_runner \
    --relay-url wss://your-relay-server.com \
    --relay-token $RELAY_TOKEN \
    --status-ui
```

### config.yaml

```yaml
provider: gemini
model: gemini-3-flash-preview
turns: 10
strategy: "Focus on science victory and strengthen scouting."
status-ui: true
```

---

## API Endpoints

### Agent Control

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Real-time dashboard (HTML) |
| `GET` | `/api/status` | Agent status snapshot (JSON) |
| `POST` | `/api/directive` | Submit a text directive |
| `GET` | `/api/agent/state` | Agent lifecycle state |
| `POST` | `/api/agent/start` | Start the agent |
| `POST` | `/api/agent/pause` | Pause execution |
| `POST` | `/api/agent/resume` | Resume execution |
| `POST` | `/api/agent/stop` | Stop the agent |
| `GET` | `/api/connection-info` | LAN IP + access URL |
| `WS` | `/ws` | WebSocket real-time channel |

### Strategy Discussion

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/discuss` | Send a message (auto-creates session) |
| `POST` | `/api/discuss/finalize` | Finalize session → extract strategy |
| `GET` | `/api/discuss/status` | Get session state + message history |

**`POST /api/discuss`** request body:

```json
{
  "message": "Should I prioritize Campus districts?",
  "user_id": "web_user",
  "mode": "in_game",
  "language": "en"
}
```

Response:

```json
{
  "session_id": "a1b2c3d4",
  "response": "Yes, Campus districts are essential for...",
  "message_count": 3
}
```

**`POST /api/discuss/finalize`** request body:

```json
{ "user_id": "web_user" }
```

Response:

```json
{ "ok": true, "strategy": "StructuredStrategy(...)" }
```

**`GET /api/discuss/status?user_id=web_user`** response:

```json
{
  "active": true,
  "session_id": "a1b2c3d4",
  "mode": "in_game",
  "message_count": 4,
  "messages": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

### WebSocket Protocol

**Type-based (recommended):**

```json
{"type": "control", "action": "start|stop|pause|resume"}
{"type": "command", "content": "Switch to culture victory"}
{"type": "ping"}
```

**Legacy mode-based:**

```json
{"mode": "high_level", "content": "Switch to culture victory"}
{"mode": "primitive", "content": "{\"action\":\"click\",\"x\":500,\"y\":300}"}
{"mode": "pause"}
```

---

## Development

```bash
make lint       # Ruff check
make format     # Ruff format + fix
make check      # Lint + type check
make test       # Run all tests
make coverage   # Coverage report
```

### Testing

```bash
pytest tests/evaluator/civ6_eval/ -v          # All evaluator tests
pytest tests/evaluator/civ6_eval/test_tolerance.py -v  # Single file
pytest -m "not integration"                    # Skip integration tests
```

---

## Coordinate Normalization

The VLM always works with `0 ~ normalizing_range` (default 1000) coordinates.
Conversion to actual screen coordinates is handled by `norm_to_real()` in `screen.py`.
Mac Retina display logical/physical pixel mismatch is handled automatically.

---

## License

MIT
