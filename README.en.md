# CivStation

`CivStation` is a layered Civilization VI computer-use stack. The repo is organized around `Context`, `Strategy`, `Action`, `HitL`, and a layered `MCP` surface instead of presenting the system as one opaque agent.

Canonical GitHub repository:

- `https://github.com/minsing-jin/civStation`

Note:

- the repository name has moved to `civStation`
- the current Python package name is still `computer-use-test`
- the current Python module name is still `computer_use_test`

## Language

- [README Hub](README.md)
- [한국어](README.ko.md)
- [中文](README.zh.md)

## At a Glance

### The Four Layers

| Layer | Core question | Main code | Details |
|---|---|---|---|
| `Context` | What is on the screen and what is the current game state? | `computer_use_test/agent/modules/context/` | [Context README](computer_use_test/agent/modules/context/README.md) |
| `Strategy` | Given the state and human intent, what should matter next? | `computer_use_test/agent/modules/strategy/` | [Strategy README](computer_use_test/agent/modules/strategy/README.md) |
| `Action` | Which primitive should handle this screen, and what action should it take? | `computer_use_test/agent/modules/router/`, `computer_use_test/agent/modules/primitive/` | [Router README](computer_use_test/agent/modules/router/README.md), [Primitive README](computer_use_test/agent/modules/primitive/README.md) |
| `HitL` | How can a human intervene while the agent is running? | `computer_use_test/agent/modules/hitl/` | [HitL README](computer_use_test/agent/modules/hitl/README.md) |

### High-Level Flow

```text
Screenshot
  -> Context
  -> Strategy
  -> Action
  -> Execution

Human-in-the-Loop can intervene at:
  - lifecycle: start / pause / resume / stop
  - strategy: high-level intent change
  - action: primitive override / direct command
```

## About MCP

This repository exposes the same architecture through a layered MCP server, so external clients do not need to import internal Python modules directly.

Main tool groups:

- `context_*`
- `strategy_*`
- `action_*`
- `hitl_*`
- `workflow_*`
- `session_*`

Key docs:

- [MCP README](computer_use_test/mcp/README.md)
- [Layered MCP Tool Map](docs/layered_mcp.md)

Why use MCP here:

- stable external contract over the same internal layers
- session-level isolated state
- import/export and adapter overrides
- easy integration with external tools, automation, and agent skills

## Extensibility

### 1. MCP Extensibility

The MCP layer is designed to support adapter swapping rather than forcing one hardcoded implementation.

Extension slots:

- `action_router`
- `action_planner`
- `context_observer`
- `strategy_refiner`
- `action_executor`

Relevant files:

- [runtime.py](computer_use_test/mcp/runtime.py)
- [server.py](computer_use_test/mcp/server.py)
- [session.py](computer_use_test/mcp/session.py)

Typical extension flow:

1. register a custom adapter in `LayerAdapterRegistry`
2. set `adapter_overrides` during `session_create`
3. or change them later with `session_config_update`

This lets you keep the same MCP contract while swapping the internal router, planner, observer, refiner, or executor per session.

### 2. Skill Extensibility

This repo is also designed to work well with skill-based coding/agent workflows.

Current skill locations:

- `.codex/skills/`
- `.agents/skills/`

Existing project-facing example:

- `.codex/skills/computer-use-mcp/SKILL.md`

Recommended pattern:

1. use MCP as the stable control plane instead of importing internals directly
2. keep domain workflows in isolated skill folders
3. define the workflow in `SKILL.md`
4. add `scripts/`, `assets/`, or `references/` next to the skill when needed

Example layout:

```text
.codex/skills/my-civ-skill/
├── SKILL.md
├── scripts/
└── references/
```

Practical skill categories that fit this repo well:

- `strategy-only`
- `plan-only`
- `hitl-ops`
- `evaluation`
- `dataset-collection`

So the extensibility story is not only about swapping runtime adapters, but also about building reusable operator-side skills on top of the same layered MCP surface.

## Repository Map

```text
computer_use_test/
├── agent/
│   ├── turn_runner.py
│   ├── turn_executor.py
│   └── modules/
│       ├── context/
│       ├── strategy/
│       ├── router/
│       ├── primitive/
│       └── hitl/
├── mcp/
├── utils/
└── evaluation/
```

## Quick Start

### Install

```bash
make install
```

or:

```bash
pip install -e ".[ui]"
```

### Environment

```env
ANTHROPIC_API_KEY=...
GENAI_API_KEY=...
OPENAI_API_KEY=...
DISCORD_BOT_TOKEN=...
WHATSAPP_BOT_TOKEN=...
```

### Run The Agent

```bash
python -m computer_use_test.agent.turn_runner \
  --provider claude \
  --turns 20 \
  --strategy "Focus on science victory" \
  --status-ui \
  --status-port 8765
```

Open:

```text
http://localhost:8765
```

### Run The Layered MCP Server

```bash
python -m computer_use_test.mcp.server
```

or:

```bash
computer_use_test_mcp
```

## HitL Usage

In this repository, `HitL` means a human can supervise and steer the running agent through external channels.

There are three practical modes:

1. local dashboard
2. direct HTTP/WebSocket control
3. remote phone controller through `tacticall`

### 1. Local Dashboard

Run the agent in wait mode:

```bash
python -m computer_use_test.agent.turn_runner \
  --provider claude \
  --turns 100 \
  --status-ui \
  --wait-for-start \
  --status-port 8765
```

Available endpoints:

- `POST /api/agent/start`
- `POST /api/agent/pause`
- `POST /api/agent/resume`
- `POST /api/agent/stop`
- `POST /api/directive`
- `POST /api/discuss`

Examples:

```bash
curl -X POST http://127.0.0.1:8765/api/agent/start
curl -X POST http://127.0.0.1:8765/api/agent/pause
curl -X POST http://127.0.0.1:8765/api/agent/resume
curl -X POST http://127.0.0.1:8765/api/agent/stop
```

### 2. WebSocket Control

Built-in agent WebSocket:

```text
ws://127.0.0.1:8765/ws
```

Supported messages:

```json
{ "type": "control", "action": "start" }
{ "type": "control", "action": "pause" }
{ "type": "control", "action": "resume" }
{ "type": "control", "action": "stop" }
{ "type": "command", "content": "Switch to culture victory and stop expanding" }
{ "type": "ping" }
```

### 3. Remote Phone Controller: `tacticall`

The remote `HitL` controller lives in the separate `tacticall` repo under `controller/`.

- controller repo: [`minsing-jin/tacticall`](https://github.com/minsing-jin/tacticall)
- controller package: `controller/`

Architecture:

```text
Phone browser
  <-> tacticall relay server (/ws on 8787)
  <-> tacticall bridge.js on the host machine
  <-> local agent websocket (ws://127.0.0.1:8765/ws)
  <-> local discussion API (http://127.0.0.1:8765/api/discuss)
```

#### A. Start the local agent

```bash
python -m computer_use_test.agent.turn_runner \
  --provider claude \
  --turns 100 \
  --status-ui \
  --wait-for-start \
  --status-port 8765
```

#### B. Start the relay/controller

```bash
cd /Users/jinminseong/Desktop/tacticall/controller
npm install
npm start
```

Relay/controller endpoints:

```text
http://127.0.0.1:8787
ws://127.0.0.1:8787/ws
```

#### C. Configure the bridge

```bash
cd /Users/jinminseong/Desktop/tacticall/controller
cp host-config.example.json host-config.json
```

Example:

```json
{
  "relayUrl": "ws://127.0.0.1:8787/ws",
  "controllerBaseUrl": "auto",
  "localApiBaseUrl": "http://127.0.0.1:8765",
  "localAgentUrl": "ws://127.0.0.1:8765/ws",
  "discussionUserId": "web_user",
  "discussionMode": "in_game",
  "discussionLanguage": "ko",
  "roomId": "civ6-room",
  "hostKey": "change-this-host-key"
}
```

Important:

- `tacticall/controller/host-config.example.json` defaults `localAgentUrl` to `ws://localhost:8000/ws`
- for this project you should change it to `ws://127.0.0.1:8765/ws`

#### D. Start the bridge

```bash
cd /Users/jinminseong/Desktop/tacticall/controller
npm run host
```

The bridge will:

1. authenticate to the relay as the host
2. connect to the local agent websocket
3. print a pairing QR code

#### E. Scan the QR code on your phone

After pairing:

- `start/pause/resume/stop` buttons send WebSocket `control`
- text commands send WebSocket `command`
- the discussion panel sends `discussion_query`
- the bridge forwards those messages to the local agent WebSocket or `POST /api/discuss`
- the phone receives `status`, `agent_state`, `video_frame`, and discussion responses

### How The Pieces Interact

#### Lifecycle control

```text
phone/web UI -> control(start|pause|resume|stop)
-> bridge.js
-> ws://127.0.0.1:8765/ws
-> AgentGate
```

#### Strategy change

```text
phone/web UI -> command("Focus on science")
-> bridge.js
-> ws://127.0.0.1:8765/ws
-> CommandQueue
-> turn_executor checkpoint
-> strategy override applied
```

#### Discussion mode

```text
phone/web UI -> discussion_query
-> bridge.js
-> POST http://127.0.0.1:8765/api/discuss
-> Strategy discussion engine
-> answer returned to phone
```

## MCP Usage Pattern

Typical external control flow:

1. `session_create`
2. `context_get` or `workflow_observe`
3. `strategy_refine`
4. `action_route` / `action_plan` or `workflow_step`
5. `hitl_send`
6. `hitl_status`

Examples:

- `hitl_send(session_id, directive_type="start")`
- `hitl_send(session_id, directive_type="pause")`
- `hitl_send(session_id, directive_type="resume")`
- `hitl_send(session_id, directive_type="stop")`
- `hitl_send(session_id, directive_type="change_strategy", payload="Avoid war and rush Campus")`

## Development

```bash
make lint
make format
make check
make test
make coverage
```
