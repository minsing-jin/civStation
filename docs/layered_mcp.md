# Layered MCP Server

`computer_use_test` now exposes a layered MCP server that wraps the existing architecture without forcing a full rewrite.

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
python -m computer_use_test.mcp.server
```

or

```bash
computer_use_test_mcp
```

The default transport is stdio, which is the right fit for local MCP clients.

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
3. `action_execute` or `workflow_act`

### Full orchestration

1. `workflow_observe`
2. `workflow_decide`
3. `workflow_step(execute=true)`
4. `hitl_status`
