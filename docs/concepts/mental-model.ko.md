# 아키텍처 개요

CivStation은 비전-언어 모델이 Civilization VI를 실제로 플레이할 수 있게 만드는 아키텍처입니다. 핵심은 VLM이 Civ6 UI 위에서 행동하되, 사람이 HitL로 전략을 계속 고도화해서 long-horizon play가 의도에서 벗어나지 않게 만드는 것입니다.

## 해결하려는 문제

많은 VLM 플레이 데모는 이 패턴에서 멈춥니다.

```text
screenshot -> model -> click
```

한 번짜리 데모에는 괜찮지만, 아래가 필요해지면 바로 어려워집니다.

- 지속되는 상태
- 여러 턴에 걸친 장기 전략
- 실행 중 전략을 수정하는 사람의 개입
- routing, planning, observation의 역할 분리
- MCP 기반 외부 orchestration

## CivStation의 답

CivStation은 VLM 플레이를 아키텍처로 바꿉니다.

```text
screen
  -> context
  -> strategy
  -> action routing
  -> action planning
  -> execution
  -> human intervention and strategy refinement when needed
```

각 부분은 책임이 명확하고, 코드베이스에서도 폴더로 나뉘어 있습니다.

## 이 프로젝트를 보는 두 가지 방법

### VLM 플레이 런타임

`turn_runner.py`를 실행하고, 게임 화면을 대상으로 현재 UI 상태를 primitive로 라우팅하고, 다음 행동을 계획하고, 로컬에서 실행하면서 턴을 이어갑니다.

### 사람이 끌고 가는 장기 전략 시스템

실행 중에도 사람은 HitL로 strategy를 계속 고도화할 수 있습니다. 이 구조가 long horizon에서 run이 의도와 어긋나지 않게 만드는 핵심입니다.

### 레이어드 플랫폼

MCP 서버의 sessions, resources, prompts를 사용해 skills나 상위 orchestration 시스템이 안정적인 계약 위에서 동작하게 합니다.

## 왜 이렇게 나누는가

이 분리는 보기 좋으라고 한 게 아닙니다. 레이어마다 실패 방식이 다르기 때문입니다.

- `Context`는 에이전트가 무엇을 알고 있는지
- `Strategy`는 여러 턴에 걸쳐 무엇이 중요해야 하는지
- `Action`은 현재 화면에서 무엇을 해야 하는지
- `HitL`은 사람이 어떻게 전략을 고도화하고 개입하는지

이것이 monolithic prompt 기반 에이전트와 CivStation을 가르는 핵심 차이입니다.

## 핵심 루프가 있는 곳

- CLI 엔트리: `civStation/agent/turn_runner.py`
- 순수 실행 루프: `civStation/agent/turn_executor.py`
- MCP facade: `civStation/mcp/server.py`

한 문장으로 요약하면, CivStation은 VLM이 문명을 플레이하되 사람이 long-horizon 전략을 계속 맞춰 줄 수 있게 만든 구조입니다.
