# Human-in-the-Loop

`HitL` 不是附加功能，而是核心层之一。

## 这一层回答什么问题

```text
How can a human supervise, interrupt, or redirect the agent while it is running?
```

## 控制表面

### 本地 dashboard

内置 FastAPI dashboard 提供：

- 浏览器 UI
- REST endpoints
- WebSocket connection
- screen/status streaming

### 直接 HTTP 与 WebSocket 控制

这是最轻量的外部控制路径，适合本地脚本、自定义 dashboard 或运维工具。

### 远程 relay/controller

项目可以通过独立的 `tacticall` controller 仓库接入 relay 驱动的手机控制器。

## directive 优先级

当多个 directive 同时排队时，运行时会按以下顺序处理：

```text
STOP > PRIMITIVE_OVERRIDE > PAUSE > CHANGE_STRATEGY
```

这个顺序是保证紧急停止可靠性的安全机制。

## 常见介入方式

- start, pause, resume, stop
- 修改高层 strategy
- 强制 primitive override
- 注入直接 command
- 围绕当前 run 发起 discussion 并 finalize strategy

## 核心文件

- `command_queue.py`
- `agent_gate.py`
- `turn_checkpoint.py`
- `status_ui/server.py`
- `status_ui/state_bridge.py`
- `relay/relay_client.py`

## 何时使用哪种表面

| 需求 | 最合适的表面 |
| --- | --- |
| 本地人工控制 | 内置 dashboard |
| 工具驱动的本地控制 | REST + WebSocket |
| 远程移动端控制 | relay + phone controller |
| 结构化外部编排 | MCP tools + `hitl_*` |

具体 endpoint 示例请看 [控制与讨论](../guides/control-and-discussion.md)。
