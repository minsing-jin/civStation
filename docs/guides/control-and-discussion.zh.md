# 控制与讨论

这页从操作员视角解释运行时控制表面。

## REST Endpoints

Agent lifecycle:

```text
GET  /api/agent/state
POST /api/agent/start
POST /api/agent/pause
POST /api/agent/resume
POST /api/agent/stop
```

Directive 与 status:

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

默认 socket：

```text
ws://127.0.0.1:8765/ws
```

消息示例：

```json
{ "type": "control", "action": "start" }
{ "type": "control", "action": "pause" }
{ "type": "control", "action": "resume" }
{ "type": "control", "action": "stop" }
{ "type": "command", "content": "Prioritize Campus and stop training settlers." }
```

## 发送 strategy directive

```bash
curl -X POST http://127.0.0.1:8765/api/directive \
  -H "Content-Type: application/json" \
  -d '{"text":"Focus on culture victory and avoid war for the next 10 turns."}'
```

像 `stop`、`pause`、`resume` 这样的 quick commands 会被解释为生命周期 directive。

## discussion mode

启用 discussion 后，项目可以保持一个 strategy discussion session，并把讨论结果整理成最终 strategy update。

示例：

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

## 远程手机控制器

远程手机控制器位于独立的 `minsing-jin/tacticall` 仓库中。

高层流程：

```text
Phone browser
  <-> relay server
  <-> host bridge
  <-> local agent websocket
  <-> local discussion API
```

当本地浏览器控制不足以满足需要，并且你想从移动端介入时，就使用 relay 模式。

## MCP 对应关系

最接近的 MCP tools：

- `hitl_send`
- `hitl_status`

当 controller 本身是另一个 agent 或 skill，而不是人工浏览器时，MCP 更合适。
