# CivStation

> 一个可控的 Civ6 computer-use 技术栈，而不是“把机器人跑起来然后祈祷它别出错”。
>
> 你可以观察屏幕、精炼策略、规划下一步动作，并通过 `HitL` 或 `MCP` 实时介入。

正式 GitHub 仓库：

- `https://github.com/minsing-jin/civStation`

当前包名和模块名仍然是：

- Python package: `computer-use-test`
- Python module: `computer_use_test`

语言：

- [English (default)](README.md)
- [English mirror](README.en.md)
- [한국어](README.ko.md)

## Quick Start

安装：

```bash
make install
```

设置 API Key：

```env
ANTHROPIC_API_KEY=...
GENAI_API_KEY=...
OPENAI_API_KEY=...
```

以可实时控制的方式运行代理：

```bash
python -m computer_use_test.agent.turn_runner \
  --provider claude \
  --turns 100 \
  --strategy "Focus on science victory" \
  --status-ui \
  --wait-for-start \
  --status-port 8765
```

打开仪表盘：

```text
http://127.0.0.1:8765
```

可选：在另一个终端启动 layered MCP 服务器：

```bash
python -m computer_use_test.mcp.server
```

## Why CivStation?

- `Layered by design`：代理被拆成可观察、可替换的层，而不是一个黑盒循环。
- `Human-steerable`：运行中可以 pause、resume、stop、change strategy 和 discussion。
- `MCP-first`：同样的架构通过稳定的外部控制面暴露出来。
- `Extensible`：无需重写整个系统，就能替换 adapter、增加 skill、改变 orchestration。
- `Operator-friendly`：支持本地仪表盘、WebSocket 控制和手机远程控制。

## Architecture

### 四个核心层

| 层 | 核心问题 | 主要代码 | 详细文档 |
|---|---|---|---|
| `Context` | 当前屏幕和游戏状态是什么？ | `computer_use_test/agent/modules/context/` | [Context README](computer_use_test/agent/modules/context/README.md) |
| `Strategy` | 在当前状态和人的意图下，下一步应该优先什么？ | `computer_use_test/agent/modules/strategy/` | [Strategy README](computer_use_test/agent/modules/strategy/README.md) |
| `Action` | 当前画面应该由哪个 primitive 处理，下一步动作是什么？ | `computer_use_test/agent/modules/router/`, `computer_use_test/agent/modules/primitive/` | [Router README](computer_use_test/agent/modules/router/README.md), [Primitive README](computer_use_test/agent/modules/primitive/README.md) |
| `HitL` | 人如何在运行中介入代理？ | `computer_use_test/agent/modules/hitl/` | [HitL README](computer_use_test/agent/modules/hitl/README.md) |

### 文件夹映射

是的，现在这些抽象模块和文件夹是一一对应的。

- `Context` -> `computer_use_test/agent/modules/context/`
- `Strategy` -> `computer_use_test/agent/modules/strategy/`
- `HitL` -> `computer_use_test/agent/modules/hitl/`
- `Action` 是唯一刻意拆开的部分：
  它分布在 `computer_use_test/agent/modules/router/` 和 `computer_use_test/agent/modules/primitive/`

这是有意的设计，因为“决定由哪个 primitive 处理屏幕”和“真正规划/执行 primitive 动作”是两个不同职责。

### 高层流程

```text
Screenshot
  -> Context
  -> Strategy
  -> Action
  -> Execution

Human-in-the-Loop can intervene at:
  - lifecycle: start / pause / resume / stop
  - strategy: high-level intent change
  - action: primitive override / direct command
```

## HitL 60 秒概览

实际有三种控制方式：

1. 本地仪表盘
2. HTTP / WebSocket 直接控制
3. 通过 `tacticall/controller` 的手机远程控制

### 本地仪表盘

运行：

```bash
python -m computer_use_test.agent.turn_runner \
  --provider claude \
  --turns 100 \
  --status-ui \
  --wait-for-start \
  --status-port 8765
```

可用接口：

- `POST /api/agent/start`
- `POST /api/agent/pause`
- `POST /api/agent/resume`
- `POST /api/agent/stop`
- `POST /api/directive`
- `POST /api/discuss`

### WebSocket 控制

代理 WebSocket：

```text
ws://127.0.0.1:8765/ws
```

支持消息：

```json
{ "type": "control", "action": "start" }
{ "type": "control", "action": "pause" }
{ "type": "control", "action": "resume" }
{ "type": "control", "action": "stop" }
{ "type": "command", "content": "Switch to culture victory and stop expanding" }
```

### 手机远程控制器

手机控制器位于独立仓库 [`minsing-jin/tacticall`](https://github.com/minsing-jin/tacticall) 的 `controller/` 目录。

架构：

```text
Phone browser
  <-> tacticall relay server (/ws on 8787)
  <-> tacticall bridge.js on the host machine
  <-> local agent websocket (ws://127.0.0.1:8765/ws)
  <-> local discussion API (http://127.0.0.1:8765/api/discuss)
```

最小配置：

```bash
cd /Users/jinminseong/Desktop/tacticall/controller
npm install
npm start
cp host-config.example.json host-config.json
```

重要 bridge 配置：

```json
{
  "relayUrl": "ws://127.0.0.1:8787/ws",
  "controllerBaseUrl": "auto",
  "localApiBaseUrl": "http://127.0.0.1:8765",
  "localAgentUrl": "ws://127.0.0.1:8765/ws",
  "roomId": "civ6-room",
  "hostKey": "change-this-host-key"
}
```

然后启动 bridge：

```bash
cd /Users/jinminseong/Desktop/tacticall/controller
npm run host
```

## MCP 与 Skill 可扩展性

### MCP

这个仓库通过 layered MCP 服务器暴露同样的架构。

工具分组：

- `context_*`
- `strategy_*`
- `action_*`
- `hitl_*`
- `workflow_*`
- `session_*`

运行方式：

```bash
python -m computer_use_test.mcp.server
```

文档：

- [MCP README](computer_use_test/mcp/README.md)
- [Layered MCP Tool Map](docs/layered_mcp.md)

### Adapter 可扩展性

MCP 运行时是 adapter 驱动的。

默认扩展槽位：

- `action_router`
- `action_planner`
- `context_observer`
- `strategy_refiner`
- `action_executor`

你可以在 `LayerAdapterRegistry` 中注册 adapter，并通过 session 级 `adapter_overrides` 选择它们。

### Skill 可扩展性

这个仓库也支持 skill-based 的 agent 工作流。

当前 skill 根目录：

- `.codex/skills/`
- `.agents/skills/`

现有示例：

- `.codex/skills/computer-use-mcp/SKILL.md`

推荐模式：

1. 让 skill 保持轻量和稳定
2. 把 MCP 作为控制面
3. 把可复用工作流写进 `SKILL.md`
4. 把脚本和参考资料放在 skill 目录旁边

## Documentation

详细层级文档：

- [Context README](computer_use_test/agent/modules/context/README.md)
- [Strategy README](computer_use_test/agent/modules/strategy/README.md)
- [Router README](computer_use_test/agent/modules/router/README.md)
- [Primitive README](computer_use_test/agent/modules/primitive/README.md)
- [HitL README](computer_use_test/agent/modules/hitl/README.md)
- [MCP README](computer_use_test/mcp/README.md)

其他语言：

- [English (default)](README.md)
- [한국어](README.ko.md)

## Development

```bash
make lint
make format
make check
make test
make coverage
```
