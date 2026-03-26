# MCP 도구

layered MCP 서버는 책임별로 도구를 그룹화합니다.

## Session 도구

- `adapter_list`
- `session_create`
- `session_list`
- `session_get`
- `session_export`
- `session_import`
- `session_delete`
- `session_config_get`
- `session_config_update`

세션 생성, 상태 확인, runtime/adapters 교체에 사용합니다.

## Context 도구

- `context_get`
- `context_update`
- `context_observe`

## Strategy 도구

- `strategy_get`
- `strategy_set`
- `strategy_refine`

## Memory 도구

- `memory_get`
- `memory_start_task`
- `memory_update`
- `memory_reset`

task-local state가 필요한 multi-step primitive 흐름에서 특히 중요합니다.

## Action 도구

- `action_route`
- `action_plan`
- `action_execute`
- `action_route_and_plan`

## Workflow 도구

- `workflow_observe`
- `workflow_decide`
- `workflow_act`
- `workflow_step`

observe-decide-act 번들을 직접 짜기보다 공통 순서를 쓰고 싶을 때 유용합니다.

## HitL 도구

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

## 대표 시퀀스

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
4. `workflow_act` 또는 `workflow_step(execute=true)`
5. `hitl_status`
