# 扩展栈

这个项目被设计成可以按层扩展。

## 添加 primitive

从这里开始：

- `civStation/agent/modules/router/primitive_registry.py`
- `civStation/agent/modules/primitive/`

典型流程：

1. 在 registry 中定义 primitive
2. 添加或修改 prompt logic
3. 处理 planning 或 multi-step behavior
4. 增加 tests
5. 更新文档

## 添加或替换 MCP adapter

从这里开始：

- `civStation/mcp/runtime.py`
- `civStation/mcp/server.py`

如果你想在不改变 public MCP surface 的前提下替换 router、planner、context observer、strategy refiner 或 executor，请使用 adapter overrides。

## 扩展 HitL

从这里开始：

- `civStation/agent/modules/hitl/command_queue.py`
- `civStation/agent/modules/hitl/status_ui/server.py`
- `civStation/agent/modules/hitl/relay/relay_client.py`

directive priority 与 lifecycle semantics 不是小功能，它们直接关系到操作安全。

## 扩展 provider 或 image handling

从这里开始：

- `civStation/utils/llm_provider/`
- `civStation/utils/image_pipeline.py`

这里适合放模型专属行为、预处理 preset 与 transport-specific 图像调优。

## 扩展 skill

项目技能位于：

- `.codex/skills/`
- `.agents/skills/`

推荐模式：

1. 保持 skill 轻量
2. 使用 MCP 作为稳定控制平面
3. 把可复用 workflow 放在 `SKILL.md`
4. 需要时把 helper script 放在 skill 旁边

## 扩展 evaluation

从这里开始：

- `civStation/evaluation/evaluator/action_eval/bbox_eval/`
- `civStation/evaluation/evaluator/action_eval/civ6_eval/`

如果你想要更通用、可复用的 action evaluator，优先考虑 `bbox_eval`。
