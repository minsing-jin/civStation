# CivStation

> "그냥 봇을 돌리고 잘 되길 바란다"에서 끝나지 않는, 제어 가능한 Civ6 computer-use 스택.
>
> 화면을 관찰하고, 전략을 정제하고, 다음 행동을 계획하고, `HitL` 또는 `MCP`로 실시간 개입할 수 있습니다.

정식 GitHub 저장소:

- `https://github.com/minsing-jin/civStation`

현재 패키지/모듈 이름은 아직 다음과 같습니다:

- Python package: `computer-use-test`
- Python module: `computer_use_test`

언어:

- [English (default)](README.md)
- [English mirror](README.en.md)
- [中文](README.zh.md)

## Quick Start

설치:

```bash
make install
```

API 키 설정:

```env
ANTHROPIC_API_KEY=...
GENAI_API_KEY=...
OPENAI_API_KEY=...
```

실시간 제어가 가능한 상태로 에이전트 실행:

```bash
python -m computer_use_test.agent.turn_runner \
  --provider claude \
  --turns 100 \
  --strategy "Focus on science victory" \
  --status-ui \
  --wait-for-start \
  --status-port 8765
```

대시보드 열기:

```text
http://127.0.0.1:8765
```

선택 사항: 다른 터미널에서 layered MCP 서버 실행:

```bash
python -m computer_use_test.mcp.server
```

## Why CivStation?

- `Layered by design`: 에이전트가 하나의 불투명한 루프가 아니라, 관찰 가능한 레이어로 나뉘어 있습니다.
- `Human-steerable`: 실행 중에도 pause, resume, stop, strategy change, discussion이 가능합니다.
- `MCP-first`: 같은 아키텍처가 안정적인 외부 제어 인터페이스로도 노출됩니다.
- `Extensible`: 전체를 다시 쓰지 않고도 adapter, skill, orchestration을 바꿀 수 있습니다.
- `Operator-friendly`: 로컬 대시보드, WebSocket 제어, 원격 폰 제어까지 지원합니다.

## Architecture

### 4개 레이어

| 레이어 | 핵심 질문 | 대표 코드 | 상세 문서 |
|---|---|---|---|
| `Context` | 지금 화면과 게임 상태를 무엇으로 이해하고 있나? | `computer_use_test/agent/modules/context/` | [Context README](computer_use_test/agent/modules/context/README.md) |
| `Strategy` | 이 상태에서 무엇을 우선해야 하나? | `computer_use_test/agent/modules/strategy/` | [Strategy README](computer_use_test/agent/modules/strategy/README.md) |
| `Action` | 어떤 primitive가 이 화면을 처리하고, 다음 행동은 무엇인가? | `computer_use_test/agent/modules/router/`, `computer_use_test/agent/modules/primitive/` | [Router README](computer_use_test/agent/modules/router/README.md), [Primitive README](computer_use_test/agent/modules/primitive/README.md) |
| `HitL` | 사람이 어떻게 실행 중 개입할 수 있나? | `computer_use_test/agent/modules/hitl/` | [HitL README](computer_use_test/agent/modules/hitl/README.md) |

### 폴더 매핑

네. 지금 추상화된 모듈은 폴더 기준으로 직접 대응됩니다.

- `Context` -> `computer_use_test/agent/modules/context/`
- `Strategy` -> `computer_use_test/agent/modules/strategy/`
- `HitL` -> `computer_use_test/agent/modules/hitl/`
- `Action`만 예외적으로 둘로 나뉩니다:
  `computer_use_test/agent/modules/router/` + `computer_use_test/agent/modules/primitive/`

이 분리는 의도적인 설계입니다. 화면을 어떤 primitive가 처리할지 고르는 책임과, 실제 primitive 행동을 계획/실행하는 책임이 다르기 때문입니다.

### 전체 흐름

```text
Screenshot
  -> Context
  -> Strategy
  -> Action
  -> Execution

Human-in-the-Loop can intervene at:
  - lifecycle: start / pause / resume / stop
  - strategy: high-level intent change
  - action: primitive override / direct command
```

## HitL 60초 요약

실제로는 세 가지 제어 방식이 있습니다:

1. 로컬 대시보드
2. HTTP / WebSocket 직접 제어
3. `tacticall/controller`를 통한 원격 폰 컨트롤러

### 로컬 대시보드

실행:

```bash
python -m computer_use_test.agent.turn_runner \
  --provider claude \
  --turns 100 \
  --status-ui \
  --wait-for-start \
  --status-port 8765
```

사용:

- `POST /api/agent/start`
- `POST /api/agent/pause`
- `POST /api/agent/resume`
- `POST /api/agent/stop`
- `POST /api/directive`
- `POST /api/discuss`

### WebSocket 제어

에이전트 소켓:

```text
ws://127.0.0.1:8765/ws
```

지원 메시지:

```json
{ "type": "control", "action": "start" }
{ "type": "control", "action": "pause" }
{ "type": "control", "action": "resume" }
{ "type": "control", "action": "stop" }
{ "type": "command", "content": "Switch to culture victory and stop expanding" }
```

### 원격 폰 컨트롤러

폰 컨트롤러는 별도 저장소 [`minsing-jin/tacticall`](https://github.com/minsing-jin/tacticall) 의 `controller/` 아래에 있습니다.

구조:

```text
Phone browser
  <-> tacticall relay server (/ws on 8787)
  <-> tacticall bridge.js on the host machine
  <-> local agent websocket (ws://127.0.0.1:8765/ws)
  <-> local discussion API (http://127.0.0.1:8765/api/discuss)
```

최소 설정:

```bash
cd /Users/jinminseong/Desktop/tacticall/controller
npm install
npm start
cp host-config.example.json host-config.json
```

중요 bridge 설정:

```json
{
  "relayUrl": "ws://127.0.0.1:8787/ws",
  "controllerBaseUrl": "auto",
  "localApiBaseUrl": "http://127.0.0.1:8765",
  "localAgentUrl": "ws://127.0.0.1:8765/ws",
  "roomId": "civ6-room",
  "hostKey": "change-this-host-key"
}
```

그 다음 bridge 실행:

```bash
cd /Users/jinminseong/Desktop/tacticall/controller
npm run host
```

## MCP와 Skill 확장성

### MCP

이 저장소는 같은 아키텍처를 layered MCP 서버로도 노출합니다.

도구 그룹:

- `context_*`
- `strategy_*`
- `action_*`
- `hitl_*`
- `workflow_*`
- `session_*`

실행:

```bash
python -m computer_use_test.mcp.server
```

문서:

- [MCP README](computer_use_test/mcp/README.md)
- [Layered MCP Tool Map](docs/layered_mcp.md)

### Adapter 확장성

MCP 런타임은 adapter 중심으로 설계되어 있습니다.

기본 확장 슬롯:

- `action_router`
- `action_planner`
- `context_observer`
- `strategy_refiner`
- `action_executor`

`LayerAdapterRegistry`에 adapter를 등록하고, 세션별 `adapter_overrides`로 고를 수 있습니다.

### Skill 확장성

이 저장소는 skill 기반 에이전트 워크플로도 지원합니다.

현재 skill 루트:

- `.codex/skills/`
- `.agents/skills/`

대표 예시:

- `.codex/skills/computer-use-mcp/SKILL.md`

권장 패턴:

1. skill은 얇고 안정적으로 유지
2. MCP를 제어면으로 사용
3. 재사용 가능한 워크플로를 `SKILL.md`에 작성
4. 필요한 스크립트와 참고자료를 skill 옆에 둠

## Documentation

상세 레이어 문서:

- [Context README](computer_use_test/agent/modules/context/README.md)
- [Strategy README](computer_use_test/agent/modules/strategy/README.md)
- [Router README](computer_use_test/agent/modules/router/README.md)
- [Primitive README](computer_use_test/agent/modules/primitive/README.md)
- [HitL README](computer_use_test/agent/modules/hitl/README.md)
- [MCP README](computer_use_test/mcp/README.md)

다른 언어:

- [English (default)](README.md)
- [中文](README.zh.md)

## Development

```bash
make lint
make format
make check
make test
make coverage
```
