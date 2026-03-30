# 架构概览

CivStation 是一套让视觉语言模型真正游玩 Civilization VI 的架构。关键不只是让 VLM 在 Civ6 UI 上行动，而是让人通过 HitL 持续细化 strategy，使长时程玩法不会偏离原本意图。

## 它要解决什么问题

很多 VLM 游戏演示都停在这一步：

```text
screenshot -> model -> click
```

一次性演示没问题，但一旦你需要下面这些能力，就会立刻变难：

- 持续状态
- 跨多个回合的长期 strategy
- 运行中持续修正 strategy 的人工介入
- routing、planning、observation 的职责拆分
- 基于 MCP 的外部编排

## CivStation 的回答

CivStation 把 VLM 游戏流程变成一套架构：

```text
screen
  -> context
  -> strategy
  -> action routing
  -> action planning
  -> execution
  -> human intervention and strategy refinement when needed
```

每一层都有明确职责，也都映射到代码库中的具体目录。

## 两种理解方式

### 作为 VLM 游戏运行时

你运行 `turn_runner.py`，面对 Civ6 游戏界面，让系统把当前 UI 状态路由到 primitive、规划下一步动作，并在本地执行。

### 作为人主导的长时程系统

人在运行过程中可以通过 HitL 持续升级和修正 strategy，这正是系统能在长时程中保持与意图一致的关键。

### 作为分层平台

你把 MCP server 的 sessions、resources、prompts 当作稳定契约，供 skills 或更高层的 orchestration 系统使用。

## 为什么要这样拆分

这种拆分不是为了好看，而是因为每一层的失败方式都不同。

- `Context` 回答代理现在知道什么
- `Strategy` 回答跨多个回合什么最重要
- `Action` 回答当前屏幕上该做什么
- `HitL` 回答人如何细化 strategy 并介入

这正是 CivStation 与单体 prompt 代理之间最关键的区别。

## 核心循环在哪里

- CLI 入口：`civStation/agent/turn_runner.py`
- 纯执行循环：`civStation/agent/turn_executor.py`
- MCP facade：`civStation/mcp/server.py`

如果只记住一句话，那就是：CivStation 的目标不是“放一个 VLM 去点屏幕”，而是让 VLM 玩 Civ6，同时让人持续把长时程 strategy 拉回到正确方向。
