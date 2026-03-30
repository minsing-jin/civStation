# 分层

只要把抽象概念映射到目录，整体结构就会变得很清楚。

## 分层映射

| 层 | 核心问题 | 主要代码 | 主要输出 |
| --- | --- | --- | --- |
| `Context` | 当前屏幕与状态被理解成什么？ | `civStation/agent/modules/context/` | 情况摘要、回合数据、本地状态 |
| `Strategy` | 在当前状态和人的意图下，接下来什么最重要？ | `civStation/agent/modules/strategy/` | `StructuredStrategy` |
| `Action` | 哪个 primitive 应该处理这个画面，它应该做什么？ | `civStation/agent/modules/router/`, `civStation/agent/modules/primitive/` | routed primitive + normalized action |
| `HitL` | 人如何监督、打断或重定向运行？ | `civStation/agent/modules/hitl/` | 生命周期控制、指令、dashboard 状态 |

## Context

context 层是其他系统读取的共享状态表面。

包含：

- `GlobalContext`
- `HighLevelContext`
- `PrimitiveContext`

关键文件：

- `context_manager.py`
- `context_updater.py`
- `turn_detector.py`
- `macro_turn_manager.py`

## Strategy

strategy 层把自由文本指导转成结构化意图。

核心产物是 `StructuredStrategy`，其中包含：

- `text`
- `victory_goal`
- `current_phase`
- `primitive_directives`
- optional `primitive_hint`

关键文件：

- `strategy_planner.py`
- `strategy_updater.py`
- `strategy_schemas.py`
- `prompts/strategy_prompts.py`

## Action

action 层被有意拆成两部分。

### Router

负责为当前屏幕选择 primitive。

关键文件：

- `primitive_registry.py`
- `router.py`
- `base_router.py`

### Primitive

负责规划真正可执行的动作或动作序列。

关键文件：

- `multi_step_process.py`
- `runtime_hooks.py`
- `base_primitive.py`
- `primitives.py`

## HitL

human-in-the-loop 层负责运行时控制。

关键文件：

- `agent_gate.py`
- `command_queue.py`
- `turn_checkpoint.py`
- `status_ui/server.py`
- `relay/relay_client.py`

## 目录映射

- `Context` -> `civStation/agent/modules/context/`
- `Strategy` -> `civStation/agent/modules/strategy/`
- `HitL` -> `civStation/agent/modules/hitl/`
- `Action` -> `router/` + `primitive/`

这种拆分是刻意的，因为 classification 与 action generation 是两类不同问题。
