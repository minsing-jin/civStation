# Control and Discussion

This page is the operator-facing guide to the runtime control surface.

## REST Endpoints

Agent lifecycle:

```text
GET  /api/agent/state
POST /api/agent/start
POST /api/agent/pause
POST /api/agent/resume
POST /api/agent/stop
```

Directive and status:

```text
GET  /api/status
GET  /api/connection-info
POST /api/directive
```

Discussion:

```text
POST /api/discuss
POST /api/discuss/finalize
GET  /api/discuss/status
```

## WebSocket

The default socket is:

```text
ws://127.0.0.1:8765/ws
```

Examples:

```json
{ "type": "control", "action": "start" }
{ "type": "control", "action": "pause" }
{ "type": "control", "action": "resume" }
{ "type": "control", "action": "stop" }
{ "type": "command", "content": "Prioritize Campus and stop training settlers." }
```

## Sending a Strategy Directive

```bash
curl -X POST http://127.0.0.1:8765/api/directive \
  -H "Content-Type: application/json" \
  -d '{"text":"Focus on culture victory and avoid war for the next 10 turns."}'
```

Quick commands such as `stop`, `pause`, and `resume` are recognized and converted into lifecycle directives.

## Discussion Mode

When discussion is enabled, the project can keep a strategy discussion session alive and later finalize that discussion into a strategy update.

Example:

```bash
curl -X POST http://127.0.0.1:8765/api/discuss \
  -H "Content-Type: application/json" \
  -d '{
        "user_id":"operator",
        "message":"We are over-expanding. Tighten economy and tech first.",
        "mode":"in_game",
        "language":"ko"
      }'
```

## Remote Phone Controller

The remote phone controller lives in the separate `minsing-jin/tacticall` repository under `controller/`.

High-level flow:

```text
Phone browser
  <-> relay server
  <-> host bridge
  <-> local agent websocket
  <-> local discussion API
```

Use the relay mode when local browser control is not enough and you want mobile access.

## MCP Mapping

The closest MCP tools are:

- `hitl_send`
- `hitl_status`

Use MCP when the controller itself is another agent or skill rather than a human in a browser.
