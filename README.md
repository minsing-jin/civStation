# CivStation

> A controllable Civ6 computer-use stack for people who want more than "run the bot and hope."
>
> Observe the screen, refine strategy, plan the next move, and intervene live through `HitL` or `MCP`.

You can also think of CivStation as a `VLM harness` for Civilization VI: it gives a vision-language model a structured loop for observation, strategy, action planning, execution, and human override.

Canonical GitHub repository:

- `https://github.com/minsing-jin/civStation`

If you install, clone, or use CivStation, a GitHub star genuinely helps.

- Fastest CLI action: `gh repo star minsing-jin/civStation`

Current package and module names are still:

- Python package: `civStation`
- Python module: `civStation`

<div align="center">

**Languages**

[English](README.md) | [한국어](README.ko.md) | [中文](README.zh.md)

</div>

## 📚 Index

- [🚀 30-Second Quick Start](#-30-second-quick-start)
- [🧭 Should I Use Docker?](#-should-i-use-docker)
- [▶️ Recommended Run Flow](#-recommended-run-flow)
- [📱 Mobile QR Quick Start](#-mobile-qr-quick-start)
- [🧠 Why HitL Matters](#-why-hitl-matters)
- [🎮 Detailed Mobile QR Flow](#-detailed-mobile-qr-flow)
- [✨ Why CivStation?](#-why-civstation)
- [🧵 Runtime Separation](#-runtime-separation)
- [🏗️ Architecture](#-architecture)
- [🕹️ HitL Control Surfaces](#-hitl-control-surfaces)
- [🧩 MCP and Skill Extensibility](#-mcp-and-skill-extensibility)
- [📖 Documentation](#-documentation)
- [🛠️ Development](#-development)

## 🚀 30-Second Quick Start

If you just want to see CivStation move in Civilization VI as fast as possible, do this:

> [!NOTE]
> Recommended starting model: `gemini-3-flash`.
> If you want one default that is fast, practical, and easy to operate for CivStation, start with `gemini-3-flash` before tuning anything else.

1. Open Civilization VI and stop on a playable map screen.
2. Run:

```bash
uv run civstation run \
  --provider gemini \
  --model gemini-3-flash \
  --turns 100 \
  --status-ui \
  --wait-for-start \
  --status-port 8765
```

3. Open `http://127.0.0.1:8765`
4. Press `Start`

That is the simplest possible path.

If you just want the setup checklist first, run:

```bash
uv run civstation
```

If macOS blocks screenshot or control access, grant:

- `Screen Recording`
- `Accessibility`

to your terminal, `uv`, or Python app, then try again.

## 🧭 Should I Use Docker?

Short answer: `no` for live gameplay.

Why:

- CivStation captures the host desktop and Civ6 window with `pyautogui` in [screen.py](/Users/jinminseong/Desktop/civStation/civStation/utils/screen.py).
- The status streamer captures the primary monitor directly with `mss` in [screen_streamer.py](/Users/jinminseong/Desktop/civStation/civStation/agent/modules/hitl/status_ui/screen_streamer.py).
- The runtime also needs host-level `Screen Recording` and `Accessibility` permissions on macOS.
- For actual play, the agent must see and click the real Civ6 window on the same machine and desktop session.

That means Docker is the wrong default for:

- live Civ6 play
- local screen capture
- mouse/keyboard control
- the mobile-control flow where the host machine is also running the game

Docker is only reasonable here for non-GUI tasks such as:

- docs builds
- linting
- tests that do not need the real game window

## ▶️ Recommended Run Flow

If you cloned the repo and want to run CivStation correctly:

1. Clone and enter the repo.

```bash
git clone https://github.com/minsing-jin/civStation.git
cd civStation
```

2. Sync the local environment.

```bash
uv sync
```

3. Print the operator guide first.

```bash
uv run civstation
```

4. Put Civ6 on the main monitor and leave the actual game screen visible.
5. If you want remote control, keep the dashboard off the game screen and use your phone or a secondary device.
6. Start the agent.

```bash
uv run civstation run \
  --provider gemini \
  --model gemini-3-flash \
  --turns 100 \
  --status-ui \
  --wait-for-start \
  --status-port 8765
```

7. Open `http://127.0.0.1:8765` and press `Start`.

Installed command aliases also work:

```bash
civstation
civstation run --provider gemini --model gemini-3-flash --turns 100 --status-ui --wait-for-start
```

Legacy fallback is still available, but not preferred:

```bash
python -m civStation
python -m civStation.agent.turn_runner --provider gemini --status-ui
```

## 📱 Mobile QR Quick Start

If you want to control the run from your phone:

1. Clone and start the mobile controller:

```bash
git clone https://github.com/minsing-jin/civ6_tacticall.git
cd civ6_tacticall
npm install
npm start
```

2. Create the bridge config:

```bash
cp host-config.example.json host-config.json
```

3. Put CivStation's local server into the config:

```json
{
  "relayUrl": "ws://127.0.0.1:8787/ws",
  "localApiBaseUrl": "http://127.0.0.1:8765",
  "localAgentUrl": "ws://127.0.0.1:8765/ws",
  "roomId": "civ6-room",
  "hostKey": "change-this-host-key"
}
```

4. Start the bridge:

```bash
npm run host
```

5. Scan the QR code with your phone
6. Press `Start` on your phone

That `Start` signal is what actually begins gameplay.

## 🧠 Why HitL Matters

> [!IMPORTANT]
> CivStation is **not** a fully autonomous agent today.
> If you do not use `HitL`, the agent can get noticeably dumber in real play.

Why:

- screen state can be ambiguous
- long-term intent can drift
- unexpected Civ6 UI states still happen
- a human is still the fastest recovery mechanism

In practice, HitL makes the agent:

- less brittle
- easier to recover
- more aligned with the goal you actually want

The easiest beginner setup is:

- local dashboard first
- mobile QR second
- full MCP automation later

## 🎮 Detailed Mobile QR Flow

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
- `Real runtime separation`: context/strategy work, main-thread action work, and HITL control are split into different runtime lanes.
- `Extensible`: swap adapters, add skills, and change orchestration without rewriting the whole system.
- `Operator-friendly`: local dashboard, WebSocket control, and remote phone control are all supported.
- `A practical VLM harness`: instead of calling a VLM on raw screenshots ad hoc, CivStation wraps the model in a reusable control loop with context, routing, planning, execution, and intervention points.

## 🧵 Runtime Separation

The MCP session/runtime model matters because it mirrors the real execution split:

- `background runtime`
  - context observation and turn tracking
  - strategy refresh and background reasoning
- `main-thread action runtime`
  - route the current screen
  - plan the primitive action
  - execute the action safely on the game window
- `hitl runtime`
  - external controller, dashboard, relay, or mobile client
  - sends lifecycle and strategy/control directives into the running system

This is the core value of the layered runtime:

- expensive background reasoning does not have to block the action loop
- the action loop stays deterministic and interruptible
- HITL stays outside the action thread, but can still steer it safely through queues and gates
- MCP sessions become real runtime containers instead of just serialized state blobs

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

This repository exposes the same architecture through a layered MCP server so the Civ agent can be reused as a portable capability instead of only as an internal code path.

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

or install the console script and run:

```bash
civStation_mcp
```

For remote or hosted MCP clients:

```bash
python -m civStation.mcp.server \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 8000 \
  --streamable-http-path /mcp \
  --json-response \
  --stateless-http
```

Docs:

- [MCP README](civStation/mcp/README.md)
- [Layered MCP Tool Map](docs/layered_mcp.md)

### Strict Layer Separation

The MCP contract preserves the runtime split used by the project:

- `strategy/context`: background-oriented
- `primitive action`: main-thread oriented
- `hitl`: external queue / relay oriented

This separation is exposed as a portable contract through tools, resources, and prompts rather than through host-specific wrapper logic.

### Adapter extensibility

The MCP runtime is still adapter-driven inside those layer boundaries.

Default extension slots:

- `action_router`
- `action_planner`
- `context_observer`
- `strategy_refiner`
- `action_executor`

You can register adapters in `LayerAdapterRegistry` and select them per session through `adapter_overrides`.

### Portable host setup

This repo ships host templates and an installer instead of hard-wiring one host's local skill/config folder into the repository.

Templates:

- `templates/clients/codex/`
- `templates/clients/claude-code/`

Installer:

```bash
python -m civStation.mcp.install_client_assets --client codex --write
python -m civStation.mcp.install_client_assets --client claude-code --write
```

Setup resources exposed by MCP:

- `civ6://install/codex-config`
- `civ6://install/claude-code-project-mcp-json`
- `civ6://install/http-client-example`
- `civ6://contracts/layers`

### Safety defaults

Live action execution is disabled by default.

- default session runtime: `execution_mode="dry_run"`
- live execution requires `session_config_update(... execution_mode="live")`
- if confirmation is enabled, callers must also pass `confirm_execute=true`

That keeps the MCP surface safe for new users while still allowing real execution when explicitly unlocked.

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
