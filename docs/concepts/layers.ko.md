# 레이어

상위 구조는 폴더에 정확히 대응시키면 쉽게 이해할 수 있습니다.

## 레이어 맵

| 레이어 | 핵심 질문 | 주요 코드 | 주요 출력 |
| --- | --- | --- | --- |
| `Context` | 지금 화면과 현재 상태를 무엇으로 이해하고 있는가? | `civStation/agent/modules/context/` | 상황 요약, 턴 데이터, 로컬 상태 |
| `Strategy` | 현재 상태와 사람의 의도를 바탕으로 다음에 무엇이 중요해야 하는가? | `civStation/agent/modules/strategy/` | `StructuredStrategy` |
| `Action` | 어떤 primitive가 이 화면을 처리하고 어떤 행동을 해야 하는가? | `civStation/agent/modules/router/`, `civStation/agent/modules/primitive/` | routed primitive + normalized action |
| `HitL` | 사람이 어떻게 감독하고 중단하거나 방향을 바꿀 수 있는가? | `civStation/agent/modules/hitl/` | lifecycle control, directives, dashboard state |

## Context

context 레이어는 나머지 시스템이 읽는 공유 상태 표면입니다.

구성 요소:

- `GlobalContext`
- `HighLevelContext`
- `PrimitiveContext`

핵심 파일:

- `context_manager.py`
- `context_updater.py`
- `turn_detector.py`
- `macro_turn_manager.py`

## Strategy

strategy 레이어는 자유 형식의 지시를 구조화된 의도로 바꿉니다.

주요 산출물은 `StructuredStrategy`이며 다음을 포함합니다.

- `text`
- `victory_goal`
- `current_phase`
- `primitive_directives`
- optional `primitive_hint`

핵심 파일:

- `strategy_planner.py`
- `strategy_updater.py`
- `strategy_schemas.py`
- `prompts/strategy_prompts.py`

## Action

action 레이어는 의도적으로 둘로 나뉩니다.

### Router

현재 화면에 맞는 primitive를 선택합니다.

핵심 파일:

- `primitive_registry.py`
- `router.py`
- `base_router.py`

### Primitive

실행 가능한 실제 행동이나 행동 시퀀스를 계획합니다.

핵심 파일:

- `multi_step_process.py`
- `runtime_hooks.py`
- `base_primitive.py`
- `primitives.py`

## HitL

human-in-the-loop 레이어는 런타임 제어가 일어나는 곳입니다.

핵심 파일:

- `agent_gate.py`
- `command_queue.py`
- `turn_checkpoint.py`
- `status_ui/server.py`
- `relay/relay_client.py`

## 폴더 매핑

- `Context` -> `civStation/agent/modules/context/`
- `Strategy` -> `civStation/agent/modules/strategy/`
- `HitL` -> `civStation/agent/modules/hitl/`
- `Action` -> `router/` + `primitive/`

이 분리는 classification과 action generation이 다른 문제이기 때문에 의도된 설계입니다.
