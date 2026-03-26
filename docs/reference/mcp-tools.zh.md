# MCP 工具

layered MCP server 按职责对工具进行分组。

## Session 工具

- `adapter_list`
- `session_create`
- `session_list`
- `session_get`
- `session_export`
- `session_import`
- `session_delete`
- `session_config_get`
- `session_config_update`

用来创建隔离 session、检查状态并切换 runtime/adapters。

## Context 工具

- `context_get`
- `context_update`
- `context_observe`

## Strategy 工具

- `strategy_get`
- `strategy_set`
- `strategy_refine`

## Memory 工具

- `memory_get`
- `memory_start_task`
- `memory_update`
- `memory_reset`

对于需要 task-local state 的 multi-step primitive 流程尤其重要。

## Action 工具

- `action_route`
- `action_plan`
- `action_execute`
- `action_route_and_plan`

## Workflow 工具

- `workflow_observe`
- `workflow_decide`
- `workflow_act`
- `workflow_step`

当你想复用 observe-decide-act 的通用顺序，而不是自己手动编排时很有用。

## HitL 工具

- `hitl_send`
- `hitl_status`

## Resources

- `civ6://sessions`
- `civ6://sessions/{session_id}/state`
- `civ6://sessions/{session_id}/context`
- `civ6://sessions/{session_id}/memory`

## Prompt Templates

- `strategy_only_workflow`
- `plan_only_workflow`
- `full_orchestration_workflow`
- `relay_controlled_workflow`

## 典型序列

### strategy only

1. `session_create`
2. `context_get`
3. `memory_get`
4. `strategy_refine`
5. `strategy_get`

### plan only

1. `session_create`
2. `workflow_observe`
3. `action_route`
4. `action_plan`

### full orchestration

1. `session_create`
2. `workflow_observe`
3. `workflow_decide`
4. `workflow_act` 或 `workflow_step(execute=true)`
5. `hitl_status`
