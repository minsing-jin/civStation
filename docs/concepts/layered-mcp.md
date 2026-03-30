# Layered MCP

The MCP server exposes the project as a stable layered interface instead of forcing callers to import internal Python modules.

## Why It Exists

The live runtime is useful for local operation. The MCP layer is useful when you want:

- session isolation
- structured orchestration
- adapter overrides
- resources and prompts
- a contract that survives internal refactors

## Mental Model

The MCP contract mirrors the architecture:

- `context`
- `strategy`
- `memory`
- `action`
- `hitl`

And adds orchestration on top:

- `workflow`
- `session`

## Runtime Split

The session/runtime model should be understood as a real layered runtime, not just a state bag.

- `background runtime`
  - context observation
  - turn detection
  - strategy refresh
- `main-thread action runtime`
  - routing
  - planning
  - execution
- `hitl runtime`
  - external controller
  - dashboard
  - relay/mobile client

This split is the core value:

- background reasoning can stay asynchronous
- the action loop can remain focused and interruptible
- HITL can stay outside the action loop while still steering it safely
- sessions become useful runtime containers for skills and external agents

## Session Model

Each MCP session owns:

- independent context
- independent short-term memory
- independent HITL queue and gate state
- runtime config
- adapter overrides
- last capture, route, and plan artifacts

That isolation is what makes the MCP surface reusable by skills and external agents.

## Adapter Model

Default adapter slots:

- `action_router`
- `action_planner`
- `context_observer`
- `strategy_refiner`
- `action_executor`

The runtime resolves these by name, so a session can switch implementations without changing the public tool names.

## Resources and Prompts

The server also registers resources and prompt templates.

Resources:

- `civ6://sessions`
- `civ6://sessions/{session_id}/state`
- `civ6://sessions/{session_id}/context`
- `civ6://sessions/{session_id}/memory`

Prompts:

- `strategy_only_workflow`
- `plan_only_workflow`
- `full_orchestration_workflow`
- `relay_controlled_workflow`

## Run

```bash
python -m civStation.mcp.server
```

The default transport is stdio, which is the right fit for local MCP clients.

## Use It When

- you are building project-specific skills
- you want stable orchestration primitives
- you do not want external tools to depend on internal Python imports

Use [MCP Tools](../reference/mcp-tools.md) for the full grouped tool list.
