# 分层 MCP

MCP server 会把项目暴露为稳定的分层接口，外部调用方不需要直接 import 内部 Python 模块。

## 为什么需要它

实时运行时适合本地操作，而 MCP 层适合这些场景：

- session isolation
- 结构化 orchestration
- adapter overrides
- resources 与 prompts
- 能承受内部重构的稳定契约

## 心智模型

MCP 契约与架构一一对应：

- `context`
- `strategy`
- `memory`
- `action`
- `hitl`

并在上面增加一层 orchestration：

- `workflow`
- `session`

## Session 模型

每个 MCP session 都拥有独立的：

- context
- short-term memory
- HITL queue 与 gate state
- runtime config
- adapter overrides
- last capture、route、plan artifacts

这种隔离让 skills 与外部 agents 可以复用同一个服务器。

## adapter 模型

默认 adapter slots：

- `action_router`
- `action_planner`
- `context_observer`
- `strategy_refiner`
- `action_executor`

session 通过名字来选择 adapter，因此无需改变公开工具名就能替换内部实现。

## resources 与 prompts

服务器还会注册 resources 与 prompt templates。

Resources:

- `civ6://sessions`
- `civ6://sessions/{session_id}/state`
- `civ6://sessions/{session_id}/context`
- `civ6://sessions/{session_id}/memory`

Prompts:

- `strategy_only_workflow`
- `plan_only_workflow`
- `full_orchestration_workflow`
- `relay_controlled_workflow`

## 运行

```bash
python -m civStation.mcp.server
```

默认 transport 是 stdio，这正是本地 MCP clients 最适合的形态。

## 什么时候用

- 当你要构建项目专用 skills
- 当你需要稳定的 orchestration primitives
- 当你不希望外部工具依赖内部 Python imports

完整列表见 [MCP 工具](../reference/mcp-tools.md)。
