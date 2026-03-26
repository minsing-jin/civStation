# Layered MCP Server

`civStation` now exposes a layered MCP server that wraps the existing architecture without forcing a full rewrite.

## What It Exposes

### Layer tools

- `context_get`
- `context_update`
- `context_observe`
- `strategy_get`
- `strategy_set`
- `strategy_refine`
- `memory_get`
- `memory_start_task`
- `memory_update`
- `memory_reset`
- `action_route`
- `action_plan`
- `action_execute`
- `action_route_and_plan`
- `hitl_send`
- `hitl_status`

### Orchestration tools

- `workflow_observe`
- `workflow_decide`
- `workflow_act`
- `workflow_step`

### Session and adapter tools

- `session_create`
- `session_list`
- `session_get`
- `session_export`
- `session_import`
- `session_delete`
- `session_config_get`
- `session_config_update`
- `adapter_list`

## Design

The outer MCP contract is layer-oriented and general:

- `action`
- `strategy`
- `context`
- `memory`
- `hitl`

The inner implementation stays close to the current project:

- `turn_executor.route_primitive()`
- `turn_executor.plan_action()`
- `StrategyPlanner`
- `ContextUpdater`
- `CommandQueue` / `AgentGate`

`ContextManager` is still reused for core logic, but MCP sessions keep isolated state snapshots and only hydrate the singleton temporarily when the legacy code path is needed.

The portable "skill surface" is the MCP server itself:

- tools for action / strategy / context / memory / HITL
- resources for installation and session inspection
- prompts for common workflows

This means the Civ agent is consumable as a reusable capability without requiring a host-specific Codex skill bundle in the repository.

## Session Model

Each MCP session owns:

- independent `context`
- independent `memory`
- independent HITL queue / gate state
- runtime provider config
- adapter override config
- last capture / route / plan artifacts

This makes it possible to:

- refine only `strategy/context/memory`
- use only `action.plan`
- call `action.execute` only when needed
- keep HITL fully external

## Adapter Customization

The MCP server supports named adapter overrides per session.

Default layer slots:

- `action_router`
- `action_planner`
- `context_observer`
- `strategy_refiner`
- `action_executor`

By default these resolve to `builtin`, but Python callers can register their own adapters through `LayerAdapterRegistry`.

## Run

```bash
python -m civStation.mcp.server
```

or

```bash
civStation_mcp
```

The default transport is stdio, which is the right fit for local MCP clients.

### Remote / HTTP

For remote or hosted MCP clients, start the server with streamable HTTP:

```bash
python -m civStation.mcp.server \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 8000 \
  --streamable-http-path /mcp \
  --json-response \
  --stateless-http
```

Then point the client at:

```text
http://127.0.0.1:8000/mcp
```

## Safety Defaults

Live action execution is intentionally not enabled by default.

- default session runtime: `execution_mode = "dry_run"`
- default live gate: `require_execute_confirmation = true`

That means `action_execute`, `workflow_act`, and `workflow_step(execute=true)` will not perform real actions until you explicitly enable them.

### Enable live execution for one session

1. `session_create`
2. `session_config_update` with:

```json
{
  "runtime_patch": {
    "execution_mode": "live",
    "require_execute_confirmation": true
  }
}
```

3. Call `action_execute` or `workflow_act` with `confirm_execute=true`

This keeps the default posture safe for external users and generic MCP clients.

## Setup Resources

The server exposes installation resources for common clients:

- `civ6://install/codex-config`
- `civ6://install/claude-desktop-config`
- `civ6://install/http-client-example`

And a setup prompt:

- `client_setup_workflow`

## Example Usage Patterns

### Strategy-only

1. `session_create`
2. `context_get`
3. `memory_get`
4. `strategy_refine`
5. `strategy_get`

### Plan-only

1. `session_create`
2. `workflow_observe`
3. `action_route`
4. `action_plan`

### Selective execution

1. `workflow_decide`
2. inspect `plan`
3. `session_config_update` to enable `execution_mode="live"` if needed
4. `action_execute(confirm_execute=true)` or `workflow_act(confirm_execute=true)`

### Full orchestration

1. `workflow_observe`
2. `workflow_decide`
3. `workflow_step(execute=true)`
4. `hitl_status`
