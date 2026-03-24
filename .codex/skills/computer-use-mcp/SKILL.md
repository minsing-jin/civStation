---
name: computer-use-mcp
description: "Use the layered computer-use MCP server for strategy/context/memory/action/HITL workflows."
origin: project
---

# Computer Use MCP

Use this skill when you want to operate the repo through the layered MCP server instead of reaching directly into the internal modules.

## Intent

This MCP surface is designed so a caller can:

- read and update `strategy`, `context`, `memory`, and `action`
- use only the layers they care about
- use orchestration tools for simpler flows
- keep HITL as a separate client while still communicating through MCP
- swap Python adapters behind the same MCP contract

## Core Tool Groups

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

### Memory

- `memory_get`
- `memory_start_task`
- `memory_update`
- `memory_reset`

### Action

- `action_route`
- `action_plan`
- `action_execute`
- `action_route_and_plan`

### Workflow

- `workflow_observe`
- `workflow_decide`
- `workflow_act`
- `workflow_step`

### HITL

- `hitl_send`
- `hitl_status`

## Playbooks

### Strategy-only

Use when the caller wants to keep the current action layer but improve reasoning.

1. `session_create`
2. `context_get`
3. `memory_get`
4. `strategy_refine`
5. `strategy_get`

### Plan-only

Use when the caller wants a proposed next action but not execution.

1. `session_create`
2. `workflow_observe`
3. `action_route`
4. `action_plan`

### Full orchestrate

Use when the caller wants the MCP server to drive a whole step.

1. `session_create`
2. `workflow_observe`
3. `workflow_decide`
4. `workflow_step` with `execute=true`
5. `hitl_status`

### Relay / external HITL

Use when HITL is outside the local client but still needs lifecycle or directives.

1. `session_create`
2. `hitl_send` with `start` / `pause` / `resume` / `stop`
3. `hitl_send` with `change_strategy` or `primitive_override`
4. `hitl_status`

## Guidance

- Prefer layer tools when the caller wants full control.
- Prefer workflow tools when the caller wants a stable high-level loop.
- Use `session_export` / `session_import` when state needs to move across processes or be versioned.
- Use `session_config_update` and adapter overrides when swapping implementations without changing the external MCP contract.
