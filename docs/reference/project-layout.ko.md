# 프로젝트 구조

중요한 것들이 어디에 있는지 빠르게 찾기 위한 맵입니다.

## 최상위 폴더

```text
civStation/
docs/
paper/
scripts/
tests/
.agents/
.codex/
```

## 런타임 코드

```text
civStation/
  agent/
    turn_runner.py
    turn_executor.py
    models/
    modules/
      context/
      strategy/
      router/
      primitive/
      hitl/
      knowledge/
      memory/
  mcp/
  utils/
  evaluation/
```

### `agent/`

라이브 런타임 엔트리 포인트와 orchestration logic.

### `mcp/`

layered MCP facade, session model, runtime config, serialization helpers, tool registration.

### `utils/`

다음과 같은 공용 인프라:

- provider implementations
- image preprocessing
- screen capture와 execution
- logging과 run-log cache
- prompt helpers
- chat app integration

### `evaluation/`

action evaluation datasets, runners, scoring logic, metrics.

## 문서와 설계 노트

### `docs/`

사람이 읽는 문서, 테마 오버라이드, docs build config가 모두 여기 있습니다.

- `docs/mkdocs.yml`
- `docs/assets/`
- `docs/overrides/`
- `docs/plans/`

### `paper/`

논문 초안 소스, bibliography, validation artifacts.

## 테스트

```text
tests/
  agent/
  evaluator/
  utils/
  mcp/
  rough_test/
```

- `agent/`는 turn loop와 runtime modules
- `evaluator/`는 bbox와 Civ6 evaluation
- `utils/`는 lower-level helpers
- `mcp/`는 layered server
- `rough_test/`는 exploratory/heavier test material

## skill 폴더

```text
.agents/skills/
.codex/skills/
```

프로젝트 전용 및 공유 agent workflows가 있는 skill roots입니다.

## 처음 열어볼 파일

1. `README.md`
2. `civStation/agent/turn_runner.py`
3. `civStation/agent/turn_executor.py`
4. `civStation/mcp/server.py`
5. `civStation/agent/modules/*/README.md`
