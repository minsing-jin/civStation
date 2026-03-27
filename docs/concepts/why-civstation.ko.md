# 왜 CivStation인가

CivStation은 "그냥 봇 돌리고 되길 바라기" 이상을 원하는 사람을 위해 만들어졌습니다.

## 핵심 생각

Civilization VI는 원클릭 환경이 아닙니다. 로컬하게는 맞는 행동이더라도 여러 턴 뒤에는 전체적으로 나쁜 선택이 될 수 있는, 장기 상태를 가진 전략 게임입니다.

그래서 CivStation은 두 가지 의미를 가집니다.

- Civ6를 플레이하기 위한 실전형 VLM 런타임
- long-horizon visual agent를 위한 연구/엔지니어링용 제어 가능한 기반

## README가 실제로 주장하는 것

README의 철학은 비교적 명확합니다.

- 시스템은 opaque하면 안 되고 inspectable해야 한다
- 인간의 전략은 계속 루프 안에 있어야 한다
- 런타임 일은 책임별로 분리되어야 한다
- 같은 아키텍처는 MCP로도 접근 가능해야 한다
- 평가는 라이브 플레이 밖에서도 가능해야 한다

이건 구현 디테일이 아니라 제품 철학입니다.

## 왜 이렇게 구성됐는가

- `Layered by design`: context, strategy, action, HitL은 실패 방식이 다름
- `Human-steerable`: 전략 drift를 실행 중에도 고칠 수 있어야 함
- `MCP-first`: 외부 도구가 내부 구현에 묶이지 않아야 함
- `Runtime-separated`: background reasoning이 action execution을 막지 않아야 함
- `Operator-friendly`: dashboard, WebSocket, mobile controller가 1급 표면이어야 함

## 한 줄 요약

CivStation은 "Civ6를 클릭하는 VLM"이 아닙니다.

VLM이 Civilization VI를 플레이하되, 많은 턴에 걸쳐 run이 관찰 가능하고, 조정 가능하고, 의도에 맞게 유지되도록 만드는 시스템입니다.
