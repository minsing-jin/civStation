# Human-in-the-Loop

`HitL` is not an add-on. It is one of the core layers.

## What It Answers

```text
How can a human supervise, interrupt, or redirect the agent while it is running?
```

## Control Surfaces

### Local dashboard

The built-in FastAPI dashboard exposes:

- browser UI
- REST endpoints
- WebSocket connection
- screen/status streaming

### Direct HTTP and WebSocket control

This is the lightest-weight external control path. It is enough for local scripts, custom dashboards, or operational tooling.

### Remote relay/controller

The project can connect to a relay-backed phone controller through the separate `tacticall` controller repo.

## Directive Priority

When multiple directives are pending, the runtime treats them in this order:

```text
STOP > PRIMITIVE_OVERRIDE > PAUSE > CHANGE_STRATEGY
```

That ordering matters. It is the guardrail that keeps emergency stop behavior reliable even when other commands are queued.

## Common Interventions

- start, pause, resume, stop
- change high-level strategy
- force a primitive override
- inject direct commands
- discuss the current run and finalize strategy changes

## Core Files

- `command_queue.py`
- `agent_gate.py`
- `turn_checkpoint.py`
- `status_ui/server.py`
- `status_ui/state_bridge.py`
- `relay/relay_client.py`

## When to Use Which Surface

| Need | Best surface |
| --- | --- |
| Local manual control | built-in dashboard |
| Tool-driven local control | REST + WebSocket |
| Remote mobile control | relay + phone controller |
| Structured external orchestration | MCP tools plus `hitl_*` |

Use [Control and Discussion](../guides/control-and-discussion.md) for concrete endpoint examples.
