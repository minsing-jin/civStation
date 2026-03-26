# 레이어드 MCP 서버

이 페이지는 기존 MCP 도구 맵 문서를 포털 안에서 유지하기 위한 레거시 버전입니다.

## 노출되는 것

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

## 설계

바깥 MCP 계약은 레이어 기준이고, 안쪽 구현은 현재 프로젝트 구조에 가깝습니다.

현재 기준의 더 정리된 설명은 [개념 / 레이어드 MCP](concepts/layered-mcp.md)와 [레퍼런스 / MCP 도구](reference/mcp-tools.md)를 보세요.
