# HitL Layer

`HitL` means `Human-in-the-Loop`.

This layer answers:

```text
How can a human supervise, interrupt, or redirect the agent while it is running?
```

## Responsibility

This layer provides:

- lifecycle control: `start`, `pause`, `resume`, `stop`
- directive delivery: strategy changes, primitive overrides, custom commands
- local dashboard and API
- WebSocket control channel
- remote relay/controller support

## Main Files

- `agent_gate.py`
  Lifecycle state machine
- `command_queue.py`
  Thread-safe directive queue
- `turn_checkpoint.py`
  Mid-turn pause/stop/change checkpoints
- `status_ui/server.py`
  FastAPI dashboard, REST API, WebSocket server
- `status_ui/state_bridge.py`
  State snapshot broadcaster
- `relay/relay_client.py`
  Remote relay client
- `queue_listener.py`
  Async stdin/chat input listener

## Directive Priority

When multiple directives are pending, the executor applies them in this order:

```text
STOP > PRIMITIVE_OVERRIDE > PAUSE > CHANGE_STRATEGY
```

## Control Surfaces

### Local

- browser dashboard
- HTTP API
- direct WebSocket client

### Remote

- relay-backed phone controller through [`tacticall`](https://github.com/minsing-jin/tacticall)

The recommended remote controller flow is documented in the root [README](../../../../README.md).

## REST Endpoints

- `GET /api/status`
- `POST /api/directive`
- `GET /api/agent/state`
- `POST /api/agent/start`
- `POST /api/agent/pause`
- `POST /api/agent/resume`
- `POST /api/agent/stop`
- `POST /api/discuss`
- `POST /api/discuss/finalize`

## WebSocket Messages

Controller to agent:

```json
{ "type": "control", "action": "start" }
{ "type": "command", "content": "Prioritize Campus" }
```

Agent to controller:

- `status`
- `phase`
- `message`
- dashboard snapshots

## MCP Mapping

- `hitl_send`
- `hitl_status`

See also:

- [MCP README](../../../mcp/README.md)
- [Layered MCP doc](../../../../docs/layered_mcp.md)
