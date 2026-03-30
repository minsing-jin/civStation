# 🧩 Layered MCP

## 📚 Index

- [🧠 Mental Model](#-mental-model)
- [📁 Main Files](#-main-files)
- [🛠️ Tool Groups](#-tool-groups)
- [🔧 Extensibility](#-extensibility)
- [🧵 Runtime Split](#-runtime-split)
- [🧠 Skill Integration](#-skill-integration)
- [🚀 Typical Usage](#-typical-usage)
- [▶️ Run](#-run)

This folder exposes the repo as a layered MCP server instead of forcing callers to import internal Python modules directly.

## 🧠 Mental Model

The MCP contract matches the architecture described in the root README:

- `context`
- `strategy`
- `action`
- `hitl`

plus higher-level orchestration:

- `workflow`
- `session`

## 📁 Main Files

- `server.py`
  Registers MCP tools, prompts, and resources
- `session.py`
  Session isolation and import/export
- `runtime.py`
  Adapter registry and runtime configuration
- `codec.py`
  Serialization and patch helpers

## 🛠️ Tool Groups

### Session

- `session_create`
- `session_list`
- `session_get`
- `session_export`
- `session_import`
- `session_delete`
- `session_config_get`
- `session_config_update`
- `adapter_list`

### Context

- `context_get`
- `context_update`
- `context_observe`

### Strategy

- `strategy_get`
- `strategy_set`
- `strategy_refine`

### Action

- `action_route`
- `action_plan`
- `action_execute`
- `action_route_and_plan`

### HitL

- `hitl_send`
- `hitl_status`

### Workflow

- `workflow_observe`
- `workflow_decide`
- `workflow_act`
- `workflow_step`

## 🔧 Extensibility

The MCP layer is intentionally adapter-driven.

Default extension slots:

- `action_router`
- `action_planner`
- `context_observer`
- `strategy_refiner`
- `action_executor`

You can:

1. register new adapters in `LayerAdapterRegistry`
2. select them at `session_create`
3. or switch them later with `session_config_update`

This keeps the external MCP contract stable while letting you swap internal implementations per session.

## 🧵 Runtime Split

The important point is that MCP is not just exposing serialized state.
It reflects a real layered runtime split:

- `background runtime`
  - context observation
  - turn detection
  - strategy refresh
- `main-thread action runtime`
  - route
  - plan
  - execute
- `hitl runtime`
  - dashboard
  - relay/mobile controller
  - lifecycle and directive injection

This is why MCP sessions matter:

- they can hold isolated runtime state for each lane
- they let external tools reason about the system in the same way the live runtime is structured
- they keep background work, foreground action, and HITL control conceptually separate
- they make adapter overrides meaningful at the runtime level, not only at the serialization level

## 🧠 Skill Integration

This MCP surface is also the recommended control plane for agent skills.

Why:

- skills should avoid importing unstable internal modules when a stable MCP contract exists
- the same workflow can be reused across Codex/Claude-style skill systems
- project-specific skills can wrap common MCP sequences such as strategy-only, plan-only, or HITL operations

Current project-facing example:

- `.codex/skills/computer-use-mcp/SKILL.md`

## 🚀 Typical Usage

### Strategy-only

1. `session_create`
2. `context_get`
3. `strategy_refine`
4. `strategy_get`

### Plan-only

1. `session_create`
2. `workflow_observe`
3. `action_route`
4. `action_plan`

### Full step

1. `session_create`
2. `workflow_step(execute=true)`
3. `hitl_status`

## ▶️ Run

```bash
python -m civStation.mcp.server
```

or:

```bash
civStation_mcp
```

For the complete tool map and prompt resources, see [docs/layered_mcp.md](../../docs/layered_mcp.md).
