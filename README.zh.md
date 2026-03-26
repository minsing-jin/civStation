# CivStation

> 一个可控的 Civ6 computer-use 技术栈，而不是“把机器人跑起来然后祈祷它别出错”。
>
> 你可以观察屏幕、精炼策略、规划下一步动作，并通过 `HitL` 或 `MCP` 实时介入。

换个角度看，CivStation 也可以理解成一个面向 Civilization VI 的 `VLM harness`。它不是把视觉语言模型直接临时接到截图上，而是把观察、策略、动作规划、执行和人工介入封装进一个结构化循环里。

正式 GitHub 仓库：

- `https://github.com/minsing-jin/civStation`

当前包名和模块名仍然是：

- Python package: `civStation`
- Python module: `civStation`

<div align="center">

**语言选择**

[English](README.md) | [한국어](README.ko.md) | [中文](README.zh.md)

</div>

## 📚 Index

- [🚀 Quick Start](#-quick-start)
- [🎮 通过 `civ6_tacticall` 手机二维码控制器玩 Civ6](#-通过-civ6_tacticall-手机二维码控制器玩-civ6)
- [✨ Why CivStation?](#-why-civstation)
- [🧵 Runtime Separation](#-runtime-separation)
- [🏗️ Architecture](#-architecture)
- [🕹️ HitL 控制面](#-hitl-控制面)
- [🧩 MCP 与 Skill 可扩展性](#-mcp-与-skill-可扩展性)
- [📖 Documentation](#-documentation)
- [🛠️ Development](#-development)

## 🚀 Quick Start

这是让 CivStation 通过 `HitL` 真正开始玩 Civilization VI 的最快路径。

> [!NOTE]
> 推荐起始模型：`gemini-3-flash`。
> 如果你想先用一个默认模型把 CivStation 跑起来，并兼顾速度与实用性，建议先从 `gemini-3-flash` 开始。

### 1. 准备宿主机器

- 在 **运行 Civilization VI 的同一台机器** 上运行代理。
- 给终端或 Python 进程授予 `Screen Recording` 和 `Accessibility` 权限。
- 代理运行期间，Civilization VI 必须始终可见且不要被遮挡。
- 推荐布局：Civ6 在主屏幕，控制器放在手机或第二台设备上。

为什么重要：

- CivStation 会捕获游戏画面，并通过 PyAutoGUI 执行动作。
- 在 macOS 上，如果游戏以 windowed 或 borderless 模式运行，`capture_screen_pil()` 会自动识别 Civ6 窗口并裁剪到游戏区域。

### 2. 准备游戏状态

- 启动 Civilization VI。
- 新开一局或者加载现有存档。
- 等游戏进入稳定、可操作的界面。
- 如果想让代理从开局开始玩，就停在第一张可交互地图画面再发送开始信号。
- 如果想从中途继续，就加载存档并停在你希望代理开始推理的那个画面。

### 3. 以 wait mode 启动 CivStation 代理服务器

```bash
python -m civStation.agent.turn_runner \
  --provider gemini \
  --model gemini-3-flash \
  --turns 100 \
  --strategy "Focus on science victory" \
  --status-ui \
  --wait-for-start \
  --status-port 8765
```

重要：

- 开启 `--wait-for-start` 后，代理 **不会立即开始游戏**
- 它会先启动 dashboard / API / WebSocket 服务
- 只有收到 `HitL start` 信号后，才会真正开始操作 Civilization VI

内置仪表盘：

```text
http://127.0.0.1:8765
```

### 4. 启动 `civ6_tacticall` 手机控制器

手机二维码控制器位于独立仓库 [`minsing-jin/civ6_tacticall`](https://github.com/minsing-jin/civ6_tacticall.git)。

```bash
git clone https://github.com/minsing-jin/civ6_tacticall.git
cd civ6_tacticall
npm install
npm start
```

这会启动二维码移动控制器 UI 和 relay：

```text
http://127.0.0.1:8787
ws://127.0.0.1:8787/ws
```

### 5. 配置 `civ6_tacticall` 与 CivStation 之间的 bridge

```bash
cd civ6_tacticall
cp host-config.example.json host-config.json
```

配置示例：

```json
{
  "relayUrl": "ws://127.0.0.1:8787/ws",
  "controllerBaseUrl": "auto",
  "localApiBaseUrl": "http://127.0.0.1:8765",
  "localAgentUrl": "ws://127.0.0.1:8765/ws",
  "discussionUserId": "web_user",
  "discussionMode": "in_game",
  "discussionLanguage": "zh",
  "roomId": "civ6-room",
  "hostKey": "change-this-host-key"
}
```

重要：

- `localAgentUrl` 必须指向 CivStation 的 WebSocket 服务器
- 模板默认值可能仍然是 `ws://localhost:8000/ws`
- 对 CivStation 应该改成 `ws://127.0.0.1:8765/ws`

### 6. 启动 bridge

```bash
cd civ6_tacticall
npm run host
```

bridge 会做这些事：

1. 以 host 身份连接到 `civ6_tacticall` relay
2. 连接本地 CivStation WebSocket 服务器
3. 打印控制器配对二维码

### 7. 配对控制器

- 用手机扫描二维码
- 或在浏览器中手动打开控制器并完成配对
- 配对成功后，控制器就可以发送命令并接收实时状态

### 8. 从 HitL 真正开始游戏

很多人会漏掉这一步：

- 此时 CivStation 仍然可能处于 idle 状态
- 在控制器里点击 `Start` 会发送 `control:start`
- 这个消息通过 bridge 到达 CivStation，并触发 `AgentGate.start()`
- **只有这时** 代理才会真正开始玩 Civilization VI

等价的启动方式：

- 在 `civ6_tacticall` 控制器中点击 `Start`
- 在本地 CivStation dashboard 中点击 `Start`
- 调用 `POST /api/agent/start`
- 发送 WebSocket `{ "type": "control", "action": "start" }`

### 9. 运行中介入

运行中你可以：

- `pause`
- `resume`
- `stop`
- 发送高层命令
- 发起 discussion 问题
- 中途修改策略

### 10. 安全结束

想结束这次运行时：

- 从控制器发送 `stop`
- 或调用 `POST /api/agent/stop`
- 或在本地 dashboard 中停止

## 🎮 通过 `civ6_tacticall` 手机二维码控制器玩 Civ6

### 关系

```text
Civilization VI game window
  <- screen capture + action execution -> CivStation
  <- local WebSocket/API bridge -> civ6_tacticall
  <- 远程移动端 UI -> 通过二维码配对的手机浏览器
```

### 端到端控制流

```text
Phone / Browser
  -> civ6_tacticall controller
  -> civ6_tacticall relay
  -> bridge.js on host
  -> CivStation WebSocket/API
  -> AgentGate / CommandQueue / Discussion API
  -> Civ6 gameplay
```

### `start` 实际上做了什么

```text
Controller Start button
  -> WebSocket control:start
  -> bridge.js
  -> ws://127.0.0.1:8765/ws
  -> AgentGate.start()
  -> turn_runner exits wait state
  -> turn_executor begins playing turns
```

### 推荐操作方式

- 让 Civ6 始终在主屏幕可见。
- 不要让本地控制器 UI 覆盖游戏窗口。
- 尽量用手机或第二台设备操作控制器。
- 推荐使用 `npm run host` 打印出的二维码，让手机浏览器完成配对。
- 在 macOS 上，如果想稳定使用自动窗口裁剪，推荐 windowed 或 borderless 模式。
- 运行过程中尽量不要改变分辨率。

## ✨ Why CivStation?

- `Layered by design`：代理被拆成可观察、可替换的层，而不是一个黑盒循环。
- `Human-steerable`：运行中可以 pause、resume、stop、change strategy 和 discussion。
- `MCP-first`：同样的架构通过稳定的外部控制面暴露出来。
- `真实运行时分离`：context/strategy 工作、主线程 action 工作、以及 HITL 控制被拆成不同的 runtime lane。
- `Extensible`：无需重写整个系统，就能替换 adapter、增加 skill、改变 orchestration。
- `Operator-friendly`：支持本地仪表盘、WebSocket 控制和手机远程控制。
- `实用型 VLM harness`：不是临时把 VLM 调在原始截图上，而是把上下文、路由、规划、执行和介入点组织成可复用的控制循环。

## 🧵 Runtime Separation

MCP session/runtime 的关键价值在于，它映射了真实执行时的分离结构：

- `background runtime`
  - context 观察与 turn tracking
  - strategy refresh 与后台推理
- `main-thread action runtime`
  - 当前画面的 routing
  - primitive action planning
  - 对游戏窗口执行真正的 action
- `hitl runtime`
  - 外部 controller、dashboard、relay、移动端客户端
  - 向运行中的系统发送 lifecycle / strategy / control 指令

这种 layered runtime 的核心价值是：

- 重的后台推理不会阻塞 action loop
- action loop 保持可中断、可预测
- HITL 位于 action thread 之外，但仍能通过 queue/gate 安全地介入
- MCP session 不再只是序列化状态块，而是真正的 runtime container

## 🏗️ Architecture

## 🏗️ Architecture

### 四个核心层

| 层 | 核心问题 | 主要代码 | 详细文档 |
|---|---|---|---|
| `Context` | 当前屏幕和游戏状态是什么？ | `civStation/agent/modules/context/` | [Context README](civStation/agent/modules/context/README.md) |
| `Strategy` | 在当前状态和人的意图下，下一步应该优先什么？ | `civStation/agent/modules/strategy/` | [Strategy README](civStation/agent/modules/strategy/README.md) |
| `Action` | 当前画面应该由哪个 primitive 处理，下一步动作是什么？ | `civStation/agent/modules/router/`, `civStation/agent/modules/primitive/` | [Router README](civStation/agent/modules/router/README.md), [Primitive README](civStation/agent/modules/primitive/README.md) |
| `HitL` | 人如何在运行中介入代理？ | `civStation/agent/modules/hitl/` | [HitL README](civStation/agent/modules/hitl/README.md) |

### 文件夹映射

是的，现在这些抽象模块和文件夹是一一对应的。

- `Context` -> `civStation/agent/modules/context/`
- `Strategy` -> `civStation/agent/modules/strategy/`
- `HitL` -> `civStation/agent/modules/hitl/`
- `Action` 是唯一刻意拆开的部分：
  它分布在 `civStation/agent/modules/router/` 和 `civStation/agent/modules/primitive/`

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

## 🕹️ HitL 控制面

### 本地 dashboard

- `http://127.0.0.1:8765`
- `POST /api/agent/start`
- `POST /api/agent/pause`
- `POST /api/agent/resume`
- `POST /api/agent/stop`
- `POST /api/directive`
- `POST /api/discuss`

### WebSocket

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

### 远程控制器

- [`minsing-jin/civ6_tacticall`](https://github.com/minsing-jin/civ6_tacticall.git)
- 手机二维码控制器 + relay + bridge

## 🧩 MCP 与 Skill 可扩展性

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
python -m civStation.mcp.server
```

文档：

- [MCP README](civStation/mcp/README.md)
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

- [Context README](civStation/agent/modules/context/README.md)
- [Strategy README](civStation/agent/modules/strategy/README.md)
- [Router README](civStation/agent/modules/router/README.md)
- [Primitive README](civStation/agent/modules/primitive/README.md)
- [HitL README](civStation/agent/modules/hitl/README.md)
- [MCP README](civStation/mcp/README.md)

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
