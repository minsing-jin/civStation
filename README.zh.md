# CivStation

`CivStation` 是一个面向 Civilization VI 的分层 computer-use 系统。这个仓库不是把代理描述成一个黑盒，而是按 `Context`、`Strategy`、`Action`、`HitL` 和 `MCP` 这些层来组织和说明。

正式 GitHub 仓库：

- `https://github.com/minsing-jin/civStation`

注意：

- 仓库名已经变成 `civStation`
- 当前 Python 包名仍然是 `computer-use-test`
- 当前 Python 模块名仍然是 `computer_use_test`

## Language

- [English (default)](README.md)
- [한국어](README.ko.md)
- [English](README.en.md)

## 总览

### 四个核心层

| 层 | 核心问题 | 主要代码 | 详细文档 |
|---|---|---|---|
| `Context` | 当前屏幕和游戏状态是什么？ | `computer_use_test/agent/modules/context/` | [Context README](computer_use_test/agent/modules/context/README.md) |
| `Strategy` | 在当前状态和人的意图下，下一步应该优先什么？ | `computer_use_test/agent/modules/strategy/` | [Strategy README](computer_use_test/agent/modules/strategy/README.md) |
| `Action` | 当前画面应该由哪个 primitive 处理，下一步动作是什么？ | `computer_use_test/agent/modules/router/`, `computer_use_test/agent/modules/primitive/` | [Router README](computer_use_test/agent/modules/router/README.md), [Primitive README](computer_use_test/agent/modules/primitive/README.md) |
| `HitL` | 人如何在运行中干预代理？ | `computer_use_test/agent/modules/hitl/` | [HitL README](computer_use_test/agent/modules/hitl/README.md) |

### 高层流程

```text
Screenshot
  -> Context
  -> Strategy
  -> Action
  -> Execution

Human-in-the-Loop can intervene at:
  - lifecycle: start / pause / resume / stop
  - strategy: change high-level intent
  - action: primitive override / direct command
```

## 关于 MCP

这个仓库把同样的架构通过分层 MCP 服务器暴露出来，因此外部客户端不必直接导入内部 Python 模块。

主要工具分组：

- `context_*`
- `strategy_*`
- `action_*`
- `hitl_*`
- `workflow_*`
- `session_*`

关键文档：

- [MCP README](computer_use_test/mcp/README.md)
- [Layered MCP Tool Map](docs/layered_mcp.md)

为什么这里要用 MCP：

- 为同一套内部层提供稳定的外部契约
- 每个 session 拥有隔离状态
- 支持 import/export 与 adapter override
- 方便接入外部工具、自动化和 agent skill

## 可扩展性

### 1. MCP 可扩展性

MCP 层不是简单的包装器，它从一开始就考虑了 adapter 替换。

扩展槽位：

- `action_router`
- `action_planner`
- `context_observer`
- `strategy_refiner`
- `action_executor`

相关文件：

- [runtime.py](computer_use_test/mcp/runtime.py)
- [server.py](computer_use_test/mcp/server.py)
- [session.py](computer_use_test/mcp/session.py)

典型扩展流程：

1. 在 `LayerAdapterRegistry` 中注册自定义 adapter
2. 在 `session_create` 时传入 `adapter_overrides`
3. 或之后通过 `session_config_update` 修改

这样可以在不改变 MCP 对外契约的前提下，为不同 session 替换内部 router、planner、observer、refiner 和 executor。

### 2. Skill 可扩展性

这个仓库同样适合 skill-based 的编码/代理工作流。

当前 skill 目录：

- `.codex/skills/`
- `.agents/skills/`

现有项目示例：

- `.codex/skills/computer-use-mcp/SKILL.md`

推荐模式：

1. skill 尽量把 MCP 作为稳定控制面，而不是直接 import 内部实现
2. 把领域工作流放在独立 skill 文件夹里
3. 在 `SKILL.md` 中定义工作流
4. 有需要时，在 skill 旁边增加 `scripts/`、`assets/`、`references/`

示例结构：

```text
.codex/skills/my-civ-skill/
├── SKILL.md
├── scripts/
└── references/
```

很适合这个仓库的 skill 类型：

- `strategy-only`
- `plan-only`
- `hitl-ops`
- `evaluation`
- `dataset-collection`

所以这里的扩展性不仅是运行时 adapter 的扩展，也包括在同一套 layered MCP 之上构建可复用的 operator-side skill。

## 仓库结构

```text
computer_use_test/
├── agent/
│   ├── turn_runner.py
│   ├── turn_executor.py
│   └── modules/
│       ├── context/
│       ├── strategy/
│       ├── router/
│       ├── primitive/
│       └── hitl/
├── mcp/
├── utils/
└── evaluation/
```

## 快速开始

### 安装

```bash
make install
```

或者：

```bash
pip install -e ".[ui]"
```

### 环境变量

```env
ANTHROPIC_API_KEY=...
GENAI_API_KEY=...
OPENAI_API_KEY=...
DISCORD_BOT_TOKEN=...
WHATSAPP_BOT_TOKEN=...
```

### 运行代理

```bash
python -m computer_use_test.agent.turn_runner \
  --provider claude \
  --turns 20 \
  --strategy "Focus on science victory" \
  --status-ui \
  --status-port 8765
```

打开：

```text
http://localhost:8765
```

### 运行 Layered MCP 服务器

```bash
python -m computer_use_test.mcp.server
```

或者：

```bash
computer_use_test_mcp
```

## HitL 使用方法

在这个仓库里，`HitL` 指的是人在代理运行过程中，通过外部通道进行监督和控制。

有三种常见方式：

1. 本地仪表盘
2. HTTP/WebSocket 直接控制
3. 通过 `tacticall` 的手机远程控制器

### 1. 本地仪表盘

让代理等待人工启动：

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

### 2. WebSocket 控制

内置 WebSocket：

```text
ws://127.0.0.1:8765/ws
```

可发送消息：

```json
{ "type": "control", "action": "start" }
{ "type": "control", "action": "pause" }
{ "type": "control", "action": "resume" }
{ "type": "control", "action": "stop" }
{ "type": "command", "content": "Switch to culture victory and stop expanding" }
{ "type": "ping" }
```

### 3. 手机远程控制器：`tacticall`

远程 `HitL` 控制器位于独立仓库 `tacticall` 的 `controller/` 目录中。

- controller repo: [`minsing-jin/tacticall`](https://github.com/minsing-jin/tacticall)
- controller package: `controller/`

架构：

```text
Phone browser
  <-> tacticall relay server (/ws on 8787)
  <-> tacticall bridge.js on the host machine
  <-> local agent websocket (ws://127.0.0.1:8765/ws)
  <-> local discussion API (http://127.0.0.1:8765/api/discuss)
```

#### A. 启动本地代理

```bash
python -m computer_use_test.agent.turn_runner \
  --provider claude \
  --turns 100 \
  --status-ui \
  --wait-for-start \
  --status-port 8765
```

#### B. 启动 relay/controller

```bash
cd /Users/jinminseong/Desktop/tacticall/controller
npm install
npm start
```

地址：

```text
http://127.0.0.1:8787
ws://127.0.0.1:8787/ws
```

#### C. 配置 bridge

```bash
cd /Users/jinminseong/Desktop/tacticall/controller
cp host-config.example.json host-config.json
```

示例：

```json
{
  "relayUrl": "ws://127.0.0.1:8787/ws",
  "controllerBaseUrl": "auto",
  "localApiBaseUrl": "http://127.0.0.1:8765",
  "localAgentUrl": "ws://127.0.0.1:8765/ws",
  "discussionUserId": "web_user",
  "discussionMode": "in_game",
  "discussionLanguage": "ko",
  "roomId": "civ6-room",
  "hostKey": "change-this-host-key"
}
```

重要：

- `tacticall/controller/host-config.example.json` 默认 `localAgentUrl` 是 `ws://localhost:8000/ws`
- 对接本项目时应该改成 `ws://127.0.0.1:8765/ws`

#### D. 启动 bridge

```bash
cd /Users/jinminseong/Desktop/tacticall/controller
npm run host
```

bridge 会做这些事：

1. 作为 host 登录 relay
2. 连接本地 agent WebSocket
3. 在终端输出配对二维码

#### E. 用手机扫码

配对后：

- `start/pause/resume/stop` 按钮发送 WebSocket `control`
- 文本命令发送 WebSocket `command`
- discussion 面板发送 `discussion_query`
- bridge 把这些消息转发到本地 agent WebSocket 或 `POST /api/discuss`
- 手机端接收 `status`、`agent_state`、`video_frame` 和 discussion 回复

### 各部分如何互相控制

#### 生命周期控制

```text
phone/web UI -> control(start|pause|resume|stop)
-> bridge.js
-> ws://127.0.0.1:8765/ws
-> AgentGate
```

#### 策略修改

```text
phone/web UI -> command("Focus on science")
-> bridge.js
-> ws://127.0.0.1:8765/ws
-> CommandQueue
-> turn_executor checkpoint
-> strategy override applied
```

#### discussion 模式

```text
phone/web UI -> discussion_query
-> bridge.js
-> POST http://127.0.0.1:8765/api/discuss
-> Strategy discussion engine
-> answer returned to phone
```

## MCP 使用模式

常见外部控制流程：

1. `session_create`
2. `context_get` 或 `workflow_observe`
3. `strategy_refine`
4. `action_route` / `action_plan` 或 `workflow_step`
5. `hitl_send`
6. `hitl_status`

示例：

- `hitl_send(session_id, directive_type="start")`
- `hitl_send(session_id, directive_type="pause")`
- `hitl_send(session_id, directive_type="resume")`
- `hitl_send(session_id, directive_type="stop")`
- `hitl_send(session_id, directive_type="change_strategy", payload="Avoid war and rush Campus")`

## 开发

```bash
make lint
make format
make check
make test
make coverage
```
