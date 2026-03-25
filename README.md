# CivStation

> A controllable Civ6 computer-use stack for people who want more than "run the bot and hope."
>
> Observe the screen, refine strategy, plan the next move, and intervene live through `HitL` or `MCP`.

You can also think of CivStation as a `VLM harness` for Civilization VI: it gives a vision-language model a structured loop for observation, strategy, action planning, execution, and human override.

Canonical GitHub repository:

- `https://github.com/minsing-jin/civStation`

Current package and module names are still:

- Python package: `computer-use-test`
- Python module: `computer_use_test`

Languages:

- English (default)
- [한국어](README.ko.md)
- [中文](README.zh.md)

## Quick Start

Install:

```bash
make install
```

Set your keys:

```env
ANTHROPIC_API_KEY=...
GENAI_API_KEY=...
OPENAI_API_KEY=...
```

Run the agent with live control enabled:

```bash
python -m computer_use_test.agent.turn_runner \
  --provider claude \
  --turns 100 \
  --strategy "Focus on science victory" \
  --status-ui \
  --wait-for-start \
  --status-port 8765
```

Open the dashboard:

```text
http://127.0.0.1:8765
```

Optional: run the layered MCP server in another terminal:

```bash
python -m computer_use_test.mcp.server
```

## Why CivStation?

- `Layered by design`: the agent is broken into inspectable layers instead of one opaque loop.
- `Human-steerable`: pause, resume, stop, change strategy, and discuss the next move while the run is live.
- `MCP-first`: the same architecture is exposed as a stable external control surface.
- `Extensible`: swap adapters, add skills, and change orchestration without rewriting the whole system.
- `Operator-friendly`: local dashboard, WebSocket control, and remote phone control are all supported.
- `A practical VLM harness`: instead of calling a VLM on raw screenshots ad hoc, CivStation wraps the model in a reusable control loop with context, routing, planning, execution, and intervention points.

## Architecture

### The Four Layers

| Layer | Core question | Main code | Details |
|---|---|---|---|
| `Context` | What is on the screen and what is the current game state? | `computer_use_test/agent/modules/context/` | [Context README](computer_use_test/agent/modules/context/README.md) |
| `Strategy` | Given the state and human intent, what should matter next? | `computer_use_test/agent/modules/strategy/` | [Strategy README](computer_use_test/agent/modules/strategy/README.md) |
| `Action` | Which primitive should handle this screen, and what action should it take? | `computer_use_test/agent/modules/router/`, `computer_use_test/agent/modules/primitive/` | [Router README](computer_use_test/agent/modules/router/README.md), [Primitive README](computer_use_test/agent/modules/primitive/README.md) |
| `HitL` | How can a human intervene while the agent is running? | `computer_use_test/agent/modules/hitl/` | [HitL README](computer_use_test/agent/modules/hitl/README.md) |

### Folder Mapping

Yes, the abstractions now map directly to folders.

- `Context` lives in `computer_use_test/agent/modules/context/`
- `Strategy` lives in `computer_use_test/agent/modules/strategy/`
- `HitL` lives in `computer_use_test/agent/modules/hitl/`
- `Action` is the one deliberate split:
  it lives across `computer_use_test/agent/modules/router/` and `computer_use_test/agent/modules/primitive/`

That split is intentional: routing and primitive execution are separate responsibilities.

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

## HitL in 60 Seconds

There are three practical control modes:

1. local dashboard
2. direct HTTP / WebSocket control
3. remote phone controller through `tacticall/controller`

### Local Dashboard

Run:

```bash
python -m computer_use_test.agent.turn_runner \
  --provider claude \
  --turns 100 \
  --status-ui \
  --wait-for-start \
  --status-port 8765
```

Use:

- `POST /api/agent/start`
- `POST /api/agent/pause`
- `POST /api/agent/resume`
- `POST /api/agent/stop`
- `POST /api/directive`
- `POST /api/discuss`

### WebSocket Control

Agent socket:

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
```

### Remote Phone Controller

The phone controller lives in the separate [`minsing-jin/tacticall`](https://github.com/minsing-jin/tacticall) repo under `controller/`.

Architecture:

```text
Phone browser
  <-> tacticall relay server (/ws on 8787)
  <-> tacticall bridge.js on the host machine
  <-> local agent websocket (ws://127.0.0.1:8765/ws)
  <-> local discussion API (http://127.0.0.1:8765/api/discuss)
```

Minimal setup:

```bash
cd /Users/jinminseong/Desktop/tacticall/controller
npm install
npm start
cp host-config.example.json host-config.json
```

Important bridge config:

```json
{
  "relayUrl": "ws://127.0.0.1:8787/ws",
  "controllerBaseUrl": "auto",
  "localApiBaseUrl": "http://127.0.0.1:8765",
  "localAgentUrl": "ws://127.0.0.1:8765/ws",
  "roomId": "civ6-room",
  "hostKey": "change-this-host-key"
}
```

Then start the bridge:

```bash
cd /Users/jinminseong/Desktop/tacticall/controller
npm run host
```

## MCP and Skill Extensibility

### MCP

This repository exposes the same architecture through a layered MCP server.

Tool groups:

- `context_*`
- `strategy_*`
- `action_*`
- `hitl_*`
- `workflow_*`
- `session_*`

Run it with:

```bash
python -m computer_use_test.mcp.server
```

Docs:

- [MCP README](computer_use_test/mcp/README.md)
- [Layered MCP Tool Map](docs/layered_mcp.md)

### Adapter Extensibility

The MCP runtime is adapter-driven.

Default extension slots:

- `action_router`
- `action_planner`
- `context_observer`
- `strategy_refiner`
- `action_executor`

You can register adapters in `LayerAdapterRegistry` and select them per session through `adapter_overrides`.

### Skill Extensibility

This repo also supports skill-based agent workflows.

Current skill roots:

- `.codex/skills/`
- `.agents/skills/`

Existing project-facing example:

- `.codex/skills/computer-use-mcp/SKILL.md`

Recommended pattern:

1. keep skills thin and stable
2. use MCP as the control plane
3. put reusable workflows in `SKILL.md`
4. keep scripts and references next to the skill

## Documentation

Detailed layer docs:

- [Context README](computer_use_test/agent/modules/context/README.md)
- [Strategy README](computer_use_test/agent/modules/strategy/README.md)
- [Router README](computer_use_test/agent/modules/router/README.md)
- [Primitive README](computer_use_test/agent/modules/primitive/README.md)
- [HitL README](computer_use_test/agent/modules/hitl/README.md)
- [MCP README](computer_use_test/mcp/README.md)

Other languages:

- [한국어](README.ko.md)
- [中文](README.zh.md)

## Development

```bash
make lint
make format
make check
make test
make coverage
```
