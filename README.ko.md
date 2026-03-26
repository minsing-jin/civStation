# CivStation

> "그냥 봇을 돌리고 잘 되길 바란다"에서 끝나지 않는, 제어 가능한 Civ6 computer-use 스택.
>
> 화면을 관찰하고, 전략을 정제하고, 다음 행동을 계획하고, `HitL` 또는 `MCP`로 실시간 개입할 수 있습니다.

다르게 보면 CivStation은 Civilization VI를 위한 `VLM harness`라고도 볼 수 있습니다. 비전-언어 모델을 그냥 스크린샷에 바로 붙이는 대신, 관찰, 전략, 행동 계획, 실행, 사람 개입까지 하나의 구조화된 루프로 감쌉니다.

정식 GitHub 저장소:

- `https://github.com/minsing-jin/civStation`

현재 패키지/모듈 이름은 아직 다음과 같습니다:

- Python package: `civStation`
- Python module: `civStation`

<div align="center">

**언어 선택**

[English](README.md) | [한국어](README.ko.md) | [中文](README.zh.md)

</div>

## 📚 Index

- [🚀 Quick Start](#-quick-start)
- [🎮 `civ6_tacticall` 모바일 QR 컨트롤러로 Civ6 플레이하기](#-civ6_tacticall-모바일-qr-컨트롤러로-civ6-플레이하기)
- [✨ Why CivStation?](#-why-civstation)
- [🧵 Runtime Separation](#-runtime-separation)
- [🏗️ Architecture](#-architecture)
- [🕹️ HitL 제어면](#-hitl-제어면)
- [🧩 MCP와 Skill 확장성](#-mcp와-skill-확장성)
- [📖 Documentation](#-documentation)
- [🛠️ Development](#-development)

## 🚀 Quick Start

이 섹션은 CivStation이 실제로 Civilization VI를 플레이하기 시작하도록 만드는 가장 빠른 절차입니다.

> [!NOTE]
> 권장 시작 모델은 `gemini-3-flash`입니다.
> CivStation에서 하나의 기본값으로 시작해야 한다면, 운영 속도와 실용성 측면에서 먼저 `gemini-3-flash`를 쓰는 것을 권장합니다.

### 1. 호스트 머신 준비

- Civilization VI가 실행되는 **같은 머신**에서 에이전트를 돌립니다.
- 터미널 또는 Python 프로세스에 `Screen Recording`과 `Accessibility` 권한을 부여합니다.
- 에이전트가 실행되는 동안 Civilization VI 창은 계속 보이고 가려지지 않아야 합니다.
- 권장 구성: Civ6는 메인 화면, 컨트롤러는 휴대폰이나 보조 기기에서 조작

왜 필요한가:

- CivStation은 게임 화면을 캡처하고 PyAutoGUI로 클릭을 실행합니다.
- macOS에서는 게임이 windowed/borderless 모드일 때 `capture_screen_pil()`이 Civ6 창을 자동 탐지하고 게임 영역만 크롭합니다.

### 2. 게임 상태 준비

- Civilization VI를 실행합니다.
- 새 게임을 시작하거나 기존 저장 파일을 불러옵니다.
- 게임이 실제로 플레이 가능한 안정된 화면에 도달할 때까지 기다립니다.
- 턴 1부터 맡기고 싶으면 첫 지도 화면에서 멈춘 뒤 시작 신호를 보냅니다.
- 중간 저장부터 이어서 맡기고 싶으면, 에이전트가 해석을 시작하길 원하는 정확한 화면에서 멈춥니다.

### 3. CivStation 에이전트 서버를 wait mode로 실행

```bash
python -m civStation.agent.turn_runner \
  --provider gemini \
  --model gemini-3-flash \
  --turns 100 \
  --strategy "Focus on science victory" \
  --status-ui \
  --wait-for-start \
  --status-port 8765
```

중요:

- `--wait-for-start`를 켜면 에이전트는 **즉시 플레이를 시작하지 않습니다**
- 먼저 dashboard / API / WebSocket 서버만 띄웁니다
- 실제 플레이는 `HitL start` 신호가 도착한 뒤에만 시작됩니다

내장 대시보드:

```text
http://127.0.0.1:8765
```

### 4. `civ6_tacticall` 모바일 컨트롤러 실행

모바일 QR 컨트롤러는 별도 저장소 [`minsing-jin/civ6_tacticall`](https://github.com/minsing-jin/civ6_tacticall.git)에 있습니다.

```bash
git clone https://github.com/minsing-jin/civ6_tacticall.git
cd civ6_tacticall
npm install
npm start
```

이렇게 하면 QR 기반 모바일 컨트롤러 UI와 relay가 실행됩니다:

```text
http://127.0.0.1:8787
ws://127.0.0.1:8787/ws
```

### 5. `civ6_tacticall`과 CivStation 사이 bridge 설정

```bash
cd civ6_tacticall
cp host-config.example.json host-config.json
```

예시 설정:

```json
{
  "relayUrl": "ws://127.0.0.1:8787/ws",
  "controllerBaseUrl": "auto",
  "localApiBaseUrl": "http://127.0.0.1:8765",
  "localAgentUrl": "ws://127.0.0.1:8765/ws",
  "discussionUserId": "web_user",
  "discussionMode": "in_game",
  "discussionLanguage": "ko",
  "roomId": "civ6-room",
  "hostKey": "change-this-host-key"
}
```

중요:

- `localAgentUrl`은 CivStation의 WebSocket 서버를 가리켜야 합니다
- 템플릿 기본값은 아직 `ws://localhost:8000/ws`일 수 있습니다
- CivStation에는 `ws://127.0.0.1:8765/ws`를 써야 합니다

### 6. bridge 실행

```bash
cd civ6_tacticall
npm run host
```

bridge가 하는 일:

1. `civ6_tacticall` relay에 host로 접속
2. 로컬 CivStation WebSocket 서버에 접속
3. 컨트롤러 pairing용 QR 코드를 출력

### 7. 컨트롤러 페어링

- 휴대폰으로 QR 코드를 스캔하거나
- 브라우저에서 직접 컨트롤러를 열어 수동 pairing합니다
- pairing이 끝나면 컨트롤러가 실시간 상태를 받고 명령을 보낼 수 있습니다

### 8. HitL에서 실제 플레이 시작

많이 놓치는 부분이 바로 이 단계입니다.

- 이 시점에서도 CivStation은 아직 idle 상태일 수 있습니다
- 컨트롤러에서 `Start`를 누르면 `control:start` 메시지가 전송됩니다
- 이 메시지가 bridge를 거쳐 CivStation으로 들어가고 `AgentGate.start()`를 호출합니다
- **그때부터** 에이전트가 실제로 Civilization VI를 플레이하기 시작합니다

동일한 시작 방법:

- `civ6_tacticall` 컨트롤러에서 `Start` 누르기
- 로컬 CivStation dashboard에서 `Start` 누르기
- `POST /api/agent/start` 호출
- WebSocket `{ "type": "control", "action": "start" }` 전송

### 9. 플레이 중 개입

실행 중에는 다음이 가능합니다:

- `pause`
- `resume`
- `stop`
- high-level command 전송
- discussion 질문
- 전략 변경

### 10. 안전하게 종료

종료하고 싶으면:

- 컨트롤러에서 `stop`
- 또는 `POST /api/agent/stop`
- 또는 로컬 dashboard에서 `stop`

## 🎮 `civ6_tacticall` 모바일 QR 컨트롤러로 Civ6 플레이하기

### 관계

```text
Civilization VI game window
  <- screen capture + action execution -> CivStation
  <- local WebSocket/API bridge -> civ6_tacticall
  <- 원격 모바일 UI -> QR로 연결된 휴대폰 브라우저
```

### end-to-end 제어 흐름

```text
Phone / Browser
  -> civ6_tacticall controller
  -> civ6_tacticall relay
  -> bridge.js on host
  -> CivStation WebSocket/API
  -> AgentGate / CommandQueue / Discussion API
  -> Civ6 gameplay
```

### `start`가 실제로 하는 일

```text
Controller Start button
  -> WebSocket control:start
  -> bridge.js
  -> ws://127.0.0.1:8765/ws
  -> AgentGate.start()
  -> turn_runner exits wait state
  -> turn_executor begins playing turns
```

### 권장 운영 방식

- Civ6는 메인 화면에서 항상 보여야 합니다.
- 로컬 컨트롤러 UI가 게임 창 위를 덮지 않도록 합니다.
- 가능하면 휴대폰이나 보조 기기에서 조작합니다.
- `npm run host`가 출력한 QR 코드를 휴대폰으로 스캔해 모바일 브라우저를 pairing하는 방식을 권장합니다.
- macOS에서 자동 게임창 크롭을 쓰려면 windowed 또는 borderless 모드를 권장합니다.
- 실행 중에는 해상도를 바꾸지 않는 편이 좋습니다.

## ✨ Why CivStation?

- `Layered by design`: 에이전트가 하나의 불투명한 루프가 아니라, 관찰 가능한 레이어로 나뉘어 있습니다.
- `Human-steerable`: 실행 중에도 pause, resume, stop, strategy change, discussion이 가능합니다.
- `MCP-first`: 같은 아키텍처가 안정적인 외부 제어 인터페이스로도 노출됩니다.
- `실제 런타임 분리`: context/strategy 작업, 메인 스레드 action 작업, HITL 제어가 서로 다른 runtime lane으로 분리됩니다.
- `Extensible`: 전체를 다시 쓰지 않고도 adapter, skill, orchestration을 바꿀 수 있습니다.
- `Operator-friendly`: 로컬 대시보드, WebSocket 제어, 원격 폰 제어까지 지원합니다.
- `실전형 VLM harness`: 단순히 VLM에 스크린샷을 던지는 방식이 아니라, 컨텍스트, 라우팅, 계획, 실행, 개입 지점을 갖춘 재사용 가능한 제어 루프로 모델을 감쌉니다.

## 🧵 Runtime Separation

MCP session/runtime가 중요한 이유는, 실제 실행 구조를 그대로 반영하기 때문입니다:

- `background runtime`
  - context 관찰과 turn tracking
  - strategy refresh와 백그라운드 추론
- `main-thread action runtime`
  - 현재 화면 routing
  - primitive action planning
  - 게임 창에 대한 실제 action execution
- `hitl runtime`
  - 외부 컨트롤러, dashboard, relay, 모바일 클라이언트
  - 실행 중인 시스템으로 lifecycle / strategy / control directive를 전달

이 레이어드 런타임의 핵심가치는 다음입니다:

- 무거운 background reasoning이 action loop를 막지 않음
- action loop는 interrupt 가능하면서도 예측 가능하게 유지됨
- HITL은 action thread 밖에 있으면서도 queue/gate를 통해 안전하게 개입 가능
- MCP session이 단순 상태 저장소가 아니라, 실제 runtime container처럼 동작함

## 🏗️ Architecture

### 4개 레이어

| 레이어 | 핵심 질문 | 대표 코드 | 상세 문서 |
|---|---|---|---|
| `Context` | 지금 화면과 게임 상태를 무엇으로 이해하고 있나? | `civStation/agent/modules/context/` | [Context README](civStation/agent/modules/context/README.md) |
| `Strategy` | 이 상태에서 무엇을 우선해야 하나? | `civStation/agent/modules/strategy/` | [Strategy README](civStation/agent/modules/strategy/README.md) |
| `Action` | 어떤 primitive가 이 화면을 처리하고, 다음 행동은 무엇인가? | `civStation/agent/modules/router/`, `civStation/agent/modules/primitive/` | [Router README](civStation/agent/modules/router/README.md), [Primitive README](civStation/agent/modules/primitive/README.md) |
| `HitL` | 사람이 어떻게 실행 중 개입할 수 있나? | `civStation/agent/modules/hitl/` | [HitL README](civStation/agent/modules/hitl/README.md) |

### 폴더 매핑

네. 지금 추상화된 모듈은 폴더 기준으로 직접 대응됩니다.

- `Context` -> `civStation/agent/modules/context/`
- `Strategy` -> `civStation/agent/modules/strategy/`
- `HitL` -> `civStation/agent/modules/hitl/`
- `Action`만 예외적으로 둘로 나뉩니다:
  `civStation/agent/modules/router/` + `civStation/agent/modules/primitive/`

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

## 🕹️ HitL 제어면

### 로컬 dashboard

- `http://127.0.0.1:8765`
- `POST /api/agent/start`
- `POST /api/agent/pause`
- `POST /api/agent/resume`
- `POST /api/agent/stop`
- `POST /api/directive`
- `POST /api/discuss`

### WebSocket

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

### 원격 컨트롤러

- [`minsing-jin/civ6_tacticall`](https://github.com/minsing-jin/civ6_tacticall.git)
- 모바일 QR 컨트롤러 + relay + bridge

## 🧩 MCP와 Skill 확장성

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
python -m civStation.mcp.server
```

문서:

- [MCP README](civStation/mcp/README.md)
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

- [Context README](civStation/agent/modules/context/README.md)
- [Strategy README](civStation/agent/modules/strategy/README.md)
- [Router README](civStation/agent/modules/router/README.md)
- [Primitive README](civStation/agent/modules/primitive/README.md)
- [HitL README](civStation/agent/modules/hitl/README.md)
- [MCP README](civStation/mcp/README.md)

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
