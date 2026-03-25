# CivStation

> A controllable Civ6 computer-use stack for people who want more than "run the bot and hope."
>
> Observe the screen, refine strategy, plan the next move, and intervene live through `HitL` or `MCP`.

You can also think of CivStation as a `VLM harness` for Civilization VI: it gives a vision-language model a structured loop for observation, strategy, action planning, execution, and human override.

Canonical GitHub repository:

- `https://github.com/minsing-jin/civStation`

Current package and module names are still:

- Python package: `civStation`
- Python module: `civStation`

<div align="center">

**Languages**

[English](README.md) | [한국어](README.ko.md) | [中文](README.zh.md)

</div>

## 📚 Index

- [🚀 Quick Start](#-quick-start)
- [🎮 Playing Civ6 With `civ6_tacticall` Mobile QR Control](#-playing-civ6-with-civ6_tacticall-mobile-qr-control)
- [✨ Why CivStation?](#-why-civstation)
- [🏗️ Architecture](#-architecture)
- [🕹️ HitL Control Surfaces](#-hitl-control-surfaces)
- [🧩 MCP and Skill Extensibility](#-mcp-and-skill-extensibility)
- [📖 Documentation](#-documentation)
- [🛠️ Development](#-development)

## 🚀 Quick Start

This is the fastest path to letting CivStation actually play Civilization VI through `HitL`.

> [!NOTE]
> Recommended starting model: `gemini-3-flash`.
> If you want one default that is fast, practical, and easy to operate for CivStation, start with `gemini-3-flash` before tuning anything else.

### 1. Prepare the host machine

- Run on the same machine that has Civilization VI open.
- Grant your terminal / Python process `Screen Recording` and `Accessibility` permissions.
- Keep Civilization VI visible and unobstructed while the agent is running.
- Recommended setup: Civ6 on the main screen, controller on your phone or another device.

Why this matters:

- CivStation captures the game screen and clicks through PyAutoGUI.
- On macOS, when the game is in windowed or borderless mode, `capture_screen_pil()` automatically detects the Civ6 window and crops to it.

### 2. Prepare the game state

- Launch Civilization VI.
- Start a new game or load an existing save.
- Wait until the game is on a stable, playable screen.
- If you want the agent to play from the beginning, stop on the first interactive map screen before giving the start signal.
- If you want the agent to continue an existing run, load the save and stop on the exact screen you want it to reason from.

### 3. Start the CivStation agent server in wait mode

```bash
python -m civStation.agent.turn_runner \
  --provider gemini \
  --model gemini-3-flash \
  --turns 100 \
  --strategy "Focus on science victory" \
  --status-ui \
  --wait-for-start \
  --status-port 8765
```

Important:

- with `--wait-for-start`, the agent does **not** begin playing immediately
- it starts the dashboard/API/WebSocket server first
- actual gameplay begins only after a `start` signal arrives from `HitL`

Open the built-in dashboard:

```text
http://127.0.0.1:8765
```

### 4. Start the `civ6_tacticall` mobile controller

The mobile QR controller lives in the separate [`minsing-jin/civ6_tacticall`](https://github.com/minsing-jin/civ6_tacticall.git) repo.

```bash
git clone https://github.com/minsing-jin/civ6_tacticall.git
cd civ6_tacticall
npm install
npm start
```

This starts the QR-ready mobile controller UI and relay:

```text
http://127.0.0.1:8787
ws://127.0.0.1:8787/ws
```

### 5. Configure the bridge between `civ6_tacticall` and CivStation

```bash
cd civ6_tacticall
cp host-config.example.json host-config.json
```

Use a config like this:

```json
{
  "relayUrl": "ws://127.0.0.1:8787/ws",
  "controllerBaseUrl": "auto",
  "localApiBaseUrl": "http://127.0.0.1:8765",
  "localAgentUrl": "ws://127.0.0.1:8765/ws",
  "discussionUserId": "web_user",
  "discussionMode": "in_game",
  "discussionLanguage": "en",
  "roomId": "civ6-room",
  "hostKey": "change-this-host-key"
}
```

Important:

- `localAgentUrl` must point to CivStation's WebSocket server
- the default template may still point to `ws://localhost:8000/ws`
- for CivStation it should be `ws://127.0.0.1:8765/ws`

### 6. Start the bridge

```bash
cd civ6_tacticall
npm run host
```

What the bridge does:

1. connects to the `civ6_tacticall` relay as the host
2. connects to the local CivStation WebSocket server
3. prints a QR code for controller pairing

### 7. Pair the controller

- Scan the QR code with your phone
- or open the controller in a browser and pair manually
- once paired, the controller can send commands and receive live status

### 8. Start gameplay from HitL

This is the step people miss:

- CivStation is still idle until `HitL start` is sent
- pressing `Start` in the controller sends a `control:start` message
- that message reaches CivStation through the bridge and triggers `AgentGate.start()`
- only then does the agent begin actually playing Civilization VI

Equivalent ways to start the run:

- press `Start` in the `civ6_tacticall` controller
- press `Start` in the local CivStation dashboard
- call `POST /api/agent/start`
- send WebSocket `{ "type": "control", "action": "start" }`

### 9. Observe and intervene while it plays

While the run is active, you can:

- `pause`
- `resume`
- `stop`
- send a high-level command
- ask a discussion question
- change the strategy mid-run

### 10. Stop safely

When you want the run to end:

- send `stop` from the controller
- or use `POST /api/agent/stop`
- or stop from the local dashboard

## 🎮 Playing Civ6 With `civ6_tacticall` Mobile QR Control

### Relationship

```text
Civilization VI game window
  <- screen capture + action execution -> CivStation
  <- local WebSocket/API bridge -> civ6_tacticall
  <- remote mobile UI -> phone browser via QR
```

### End-to-end control flow

```text
Phone / Browser
  -> civ6_tacticall controller
  -> civ6_tacticall relay
  -> bridge.js on host
  -> CivStation WebSocket/API
  -> AgentGate / CommandQueue / Discussion API
  -> Civ6 gameplay
```

### What `start` actually does

```text
Controller Start button
  -> WebSocket control:start
  -> bridge.js
  -> ws://127.0.0.1:8765/ws
  -> AgentGate.start()
  -> turn_runner exits wait state
  -> turn_executor begins playing turns
```

### Recommended operator setup

- Keep Civ6 on the main display and visible at all times.
- Do not cover the game window with the local controller UI.
- Prefer controlling from a phone or secondary device.
- Pair the mobile browser by scanning the QR code printed by `npm run host`.
- Use windowed or borderless mode if you want reliable automatic game-window cropping on macOS.
- Keep the game at a stable resolution during a run.

## ✨ Why CivStation?

- `Layered by design`: the agent is broken into inspectable layers instead of one opaque loop.
- `Human-steerable`: pause, resume, stop, change strategy, and discuss the next move while the run is live.
- `MCP-first`: the same architecture is exposed as a stable external control surface.
- `Extensible`: swap adapters, add skills, and change orchestration without rewriting the whole system.
- `Operator-friendly`: local dashboard, WebSocket control, and remote phone control are all supported.
- `A practical VLM harness`: instead of calling a VLM on raw screenshots ad hoc, CivStation wraps the model in a reusable control loop with context, routing, planning, execution, and intervention points.

## 🏗️ Architecture

### The Four Layers

| Layer | Core question | Main code | Details |
|---|---|---|---|
| `Context` | What is on the screen and what is the current game state? | `civStation/agent/modules/context/` | [Context README](civStation/agent/modules/context/README.md) |
| `Strategy` | Given the state and human intent, what should matter next? | `civStation/agent/modules/strategy/` | [Strategy README](civStation/agent/modules/strategy/README.md) |
| `Action` | Which primitive should handle this screen, and what action should it take? | `civStation/agent/modules/router/`, `civStation/agent/modules/primitive/` | [Router README](civStation/agent/modules/router/README.md), [Primitive README](civStation/agent/modules/primitive/README.md) |
| `HitL` | How can a human intervene while the agent is running? | `civStation/agent/modules/hitl/` | [HitL README](civStation/agent/modules/hitl/README.md) |

### Folder Mapping

Yes, the abstractions now map directly to folders.

- `Context` lives in `civStation/agent/modules/context/`
- `Strategy` lives in `civStation/agent/modules/strategy/`
- `HitL` lives in `civStation/agent/modules/hitl/`
- `Action` is the one deliberate split:
  it lives across `civStation/agent/modules/router/` and `civStation/agent/modules/primitive/`

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

## 🕹️ HitL Control Surfaces

### Local dashboard

- `http://127.0.0.1:8765`
- `POST /api/agent/start`
- `POST /api/agent/pause`
- `POST /api/agent/resume`
- `POST /api/agent/stop`
- `POST /api/directive`
- `POST /api/discuss`

### WebSocket

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

### Remote controller

- [`minsing-jin/civ6_tacticall`](https://github.com/minsing-jin/civ6_tacticall.git)
- mobile QR controller + relay + bridge

## 🧩 MCP and Skill Extensibility

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
python -m civStation.mcp.server
```

Docs:

- [MCP README](civStation/mcp/README.md)
- [Layered MCP Tool Map](docs/layered_mcp.md)

### Adapter extensibility

The MCP runtime is adapter-driven.

Default extension slots:

- `action_router`
- `action_planner`
- `context_observer`
- `strategy_refiner`
- `action_executor`

You can register adapters in `LayerAdapterRegistry` and select them per session through `adapter_overrides`.

### Skill extensibility

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

## 📖 Documentation

Hosted docs:

- `https://minsing-jin.github.io/civStation/`

Local docs:

- `make docs-serve`
- `make docs-build`

Detailed layer docs:

- [Context README](civStation/agent/modules/context/README.md)
- [Strategy README](civStation/agent/modules/strategy/README.md)
- [Router README](civStation/agent/modules/router/README.md)
- [Primitive README](civStation/agent/modules/primitive/README.md)
- [HitL README](civStation/agent/modules/hitl/README.md)
- [MCP README](civStation/mcp/README.md)

Other languages:

- [한국어](README.ko.md)
- [中文](README.zh.md)

## 🛠️ Development

```bash
make lint
make format
make check
make test
make coverage
```
