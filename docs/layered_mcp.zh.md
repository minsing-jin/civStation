# 分层 MCP 服务器

这页是为了在门户中保留旧版 MCP 工具映射文档。

## 它暴露什么

### Layer tools

- `context_get`
- `context_update`
- `context_observe`
- `strategy_get`
- `strategy_set`
- `strategy_refine`
- `memory_get`
- `memory_start_task`
- `memory_update`
- `memory_reset`
- `action_route`
- `action_plan`
- `action_execute`
- `action_route_and_plan`
- `hitl_send`
- `hitl_status`

### Orchestration tools

- `workflow_observe`
- `workflow_decide`
- `workflow_act`
- `workflow_step`

### Session and adapter tools

- `session_create`
- `session_list`
- `session_get`
- `session_export`
- `session_import`
- `session_delete`
- `session_config_get`
- `session_config_update`
- `adapter_list`

## 设计

外层 MCP 契约按 layer 组织，而内层实现仍尽量贴近当前项目结构。

如果你想看更新、更系统的版本，请阅读 [Concepts / Layered MCP](concepts/layered-mcp.md) 与 [Reference / MCP Tools](reference/mcp-tools.md)。
