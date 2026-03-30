# 레이어드 MCP

MCP 서버는 프로젝트를 안정적인 레이어 인터페이스로 노출합니다. 외부 호출자가 내부 Python 모듈을 직접 import할 필요가 없습니다.

## 왜 필요한가

라이브 런타임은 로컬 운영에 좋고, MCP 레이어는 아래가 필요할 때 유용합니다.

- session isolation
- 구조화된 orchestration
- adapter overrides
- resources와 prompts
- 내부 리팩터링에도 버틸 수 있는 계약

## 멘탈 모델

MCP 계약은 아키텍처를 그대로 반영합니다.

- `context`
- `strategy`
- `memory`
- `action`
- `hitl`

그리고 상위 orchestration을 추가합니다.

- `workflow`
- `session`

## 세션 모델

각 MCP session은 다음을 독립적으로 가집니다.

- context
- short-term memory
- HITL queue와 gate state
- runtime config
- adapter overrides
- last capture, route, plan artifacts

이 격리가 skills와 외부 에이전트가 같은 서버를 재사용할 수 있게 만듭니다.

## adapter 모델

기본 adapter slots:

- `action_router`
- `action_planner`
- `context_observer`
- `strategy_refiner`
- `action_executor`

세션은 이들을 이름으로 선택하므로, public tool names를 바꾸지 않고도 구현을 교체할 수 있습니다.

## resources와 prompts

서버는 resources와 prompt templates도 등록합니다.

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

## 실행

```bash
python -m civStation.mcp.server
```

기본 transport는 stdio입니다. 로컬 MCP clients에 가장 잘 맞는 기본값입니다.

## 언제 써야 하나

- 프로젝트 전용 skills를 만들 때
- 안정적인 orchestration primitives가 필요할 때
- 외부 도구가 내부 Python import에 의존하지 않게 하고 싶을 때

전체 목록은 [MCP 도구](../reference/mcp-tools.md)를 보세요.
