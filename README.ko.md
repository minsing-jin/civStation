# CivStation

`CivStation`은 Civilization VI용 레이어드 컴퓨터-유즈 스택입니다. 이 저장소는 에이전트를 하나의 거대한 블랙박스로 설명하지 않고, `Context`, `Strategy`, `Action`, `HitL`, 그리고 `MCP`라는 명확한 계층으로 설명하는 것을 목표로 합니다.

정식 GitHub 저장소:

- `https://github.com/minsing-jin/civStation`

주의:

- 저장소 이름은 `civStation`으로 바뀌었지만
- 현재 Python 패키지 이름은 `computer-use-test`
- 현재 Python 모듈 이름은 `computer_use_test`

## Language

- [README Hub](README.md)
- [English](README.en.md)
- [中文](README.zh.md)

## 한눈에 보기

### 4개 레이어

| 레이어 | 핵심 질문 | 대표 코드 | 상세 문서 |
|---|---|---|---|
| `Context` | 지금 화면과 게임 상태를 무엇으로 이해하고 있나? | `computer_use_test/agent/modules/context/` | [Context README](computer_use_test/agent/modules/context/README.md) |
| `Strategy` | 이 상태에서 무엇을 우선해야 하나? | `computer_use_test/agent/modules/strategy/` | [Strategy README](computer_use_test/agent/modules/strategy/README.md) |
| `Action` | 어떤 primitive가 이 화면을 처리하고, 다음 행동은 무엇인가? | `computer_use_test/agent/modules/router/`, `computer_use_test/agent/modules/primitive/` | [Router README](computer_use_test/agent/modules/router/README.md), [Primitive README](computer_use_test/agent/modules/primitive/README.md) |
| `HitL` | 사람이 어떻게 중간 개입하고 조작할 수 있나? | `computer_use_test/agent/modules/hitl/` | [HitL README](computer_use_test/agent/modules/hitl/README.md) |

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

## MCP에 대해

이 저장소는 내부 Python 모듈을 직접 건드리지 않아도 되도록, 같은 구조를 `Layered MCP`로도 노출합니다.

주요 도구 그룹:

- `context_*`
- `strategy_*`
- `action_*`
- `hitl_*`
- `workflow_*`
- `session_*`

핵심 문서:

- [MCP README](computer_use_test/mcp/README.md)
- [Layered MCP Tool Map](docs/layered_mcp.md)

MCP를 쓰는 이유:

- 내부 구현을 몰라도 바깥에서 제어 가능
- 세션 단위로 상태를 분리 가능
- `HitL`, `strategy`, `action`을 같은 계약으로 다룰 수 있음
- 외부 클라이언트, 자동화 스크립트, 에이전트 skill에서 공통 인터페이스로 사용 가능

## 확장성

### 1. MCP 확장성

이 저장소의 MCP는 단순 래퍼가 아니라, 어댑터 교체를 전제로 설계되어 있습니다.

확장 포인트:

- `action_router`
- `action_planner`
- `context_observer`
- `strategy_refiner`
- `action_executor`

관련 코드:

- [runtime.py](computer_use_test/mcp/runtime.py)
- [server.py](computer_use_test/mcp/server.py)
- [session.py](computer_use_test/mcp/session.py)

확장 방식:

1. `LayerAdapterRegistry`에 새 adapter를 등록
2. 세션 생성 시 `adapter_overrides`를 지정하거나
3. `session_config_update`로 세션별 override를 바꿔서 같은 MCP 계약 아래 다른 구현을 연결

즉, 바깥에서 보는 인터페이스는 유지하면서도 내부 라우터, 플래너, 컨텍스트 관찰기, 실행기를 갈아끼울 수 있습니다.

### 2. Skill 확장성

이 저장소는 `skills` 기반 에이전트 워크플로에도 잘 맞도록 되어 있습니다.

현재 보이는 skill 위치:

- `.codex/skills/`
- `.agents/skills/`

이미 들어있는 대표 예시:

- `.codex/skills/computer-use-mcp/SKILL.md`

권장 패턴:

1. skill은 내부 모듈을 직접 import하기보다 MCP를 안정적인 제어면으로 사용
2. 도메인별 skill을 별도 폴더로 분리
3. `SKILL.md`에 워크플로를 정의하고 필요하면 `scripts/`, `assets/`, `references/`를 함께 둠

예시 구조:

```text
.codex/skills/my-civ-skill/
├── SKILL.md
├── scripts/
└── references/
```

실무적으로는 이렇게 나누는 것이 좋습니다:

- `strategy-only` skill
- `plan-only` skill
- `hitl-ops` skill
- `evaluation` skill
- `dataset-collection` skill

즉, 이 저장소는 코드 자체의 확장성뿐 아니라, AI 운영자용 skill 레이어까지 확장 가능한 구조입니다.

## 저장소 구조

```text
computer_use_test/
├── agent/
│   ├── turn_runner.py
│   ├── turn_executor.py
│   └── modules/
│       ├── context/
│       ├── strategy/
│       ├── router/
│       ├── primitive/
│       └── hitl/
├── mcp/
├── utils/
└── evaluation/
```

## 빠른 시작

### 설치

```bash
make install
```

또는:

```bash
pip install -e ".[ui]"
```

### 환경 변수

```env
ANTHROPIC_API_KEY=...
GENAI_API_KEY=...
OPENAI_API_KEY=...
DISCORD_BOT_TOKEN=...
WHATSAPP_BOT_TOKEN=...
```

### 에이전트 실행

```bash
python -m computer_use_test.agent.turn_runner \
  --provider claude \
  --turns 20 \
  --strategy "Focus on science victory" \
  --status-ui \
  --status-port 8765
```

열기:

```text
http://localhost:8765
```

### Layered MCP 서버 실행

```bash
python -m computer_use_test.mcp.server
```

또는:

```bash
computer_use_test_mcp
```

## HitL 사용법

이 저장소에서 `HitL`은 에이전트가 돌아가는 동안, 사람이 외부 채널을 통해 중간에 개입하는 방식을 뜻합니다.

사용 방식은 3가지입니다:

1. 로컬 대시보드
2. HTTP/WebSocket 제어
3. `tacticall` 원격 폰 컨트롤러

### 1. 로컬 대시보드

사람이 직접 시작 버튼을 누르게 하려면:

```bash
python -m computer_use_test.agent.turn_runner \
  --provider claude \
  --turns 100 \
  --status-ui \
  --wait-for-start \
  --status-port 8765
```

사용 가능한 엔드포인트:

- `POST /api/agent/start`
- `POST /api/agent/pause`
- `POST /api/agent/resume`
- `POST /api/agent/stop`
- `POST /api/directive`
- `POST /api/discuss`

예시:

```bash
curl -X POST http://127.0.0.1:8765/api/agent/start
curl -X POST http://127.0.0.1:8765/api/agent/pause
curl -X POST http://127.0.0.1:8765/api/agent/resume
curl -X POST http://127.0.0.1:8765/api/agent/stop
```

```bash
curl -X POST http://127.0.0.1:8765/api/directive \
  -H "Content-Type: application/json" \
  -d '{"text":"Prioritize Campus and avoid war for the next few turns"}'
```

### 2. WebSocket 제어

내장 상태 서버 WebSocket:

```text
ws://127.0.0.1:8765/ws
```

보낼 수 있는 메시지:

```json
{ "type": "control", "action": "start" }
{ "type": "control", "action": "pause" }
{ "type": "control", "action": "resume" }
{ "type": "control", "action": "stop" }
{ "type": "command", "content": "Switch to culture victory and stop expanding" }
{ "type": "ping" }
```

### 3. 원격 폰 컨트롤러: `tacticall`

원격 `HitL` 컨트롤러는 별도 저장소 `tacticall`의 `controller/`에 있습니다.

- controller repo: [`minsing-jin/tacticall`](https://github.com/minsing-jin/tacticall)
- controller package: `controller/`

구조:

```text
Phone browser
  <-> tacticall relay server (/ws on 8787)
  <-> tacticall bridge.js on the host machine
  <-> local agent websocket (ws://127.0.0.1:8765/ws)
  <-> local discussion API (http://127.0.0.1:8765/api/discuss)
```

#### A. 로컬 에이전트 실행

```bash
python -m computer_use_test.agent.turn_runner \
  --provider claude \
  --turns 100 \
  --status-ui \
  --wait-for-start \
  --status-port 8765
```

#### B. relay/controller 실행

```bash
cd /Users/jinminseong/Desktop/tacticall/controller
npm install
npm start
```

열리는 주소:

```text
http://127.0.0.1:8787
ws://127.0.0.1:8787/ws
```

#### C. bridge 설정

```bash
cd /Users/jinminseong/Desktop/tacticall/controller
cp host-config.example.json host-config.json
```

예시:

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

- `tacticall/controller/host-config.example.json`의 기본 `localAgentUrl`은 `ws://localhost:8000/ws`
- 이 프로젝트와 연결하려면 `ws://127.0.0.1:8765/ws`로 바꿔야 함

#### D. bridge 실행

```bash
cd /Users/jinminseong/Desktop/tacticall/controller
npm run host
```

bridge가 하는 일:

1. relay에 host로 로그인
2. 이 저장소의 로컬 agent websocket에 연결
3. QR 코드를 출력

#### E. 휴대폰으로 QR 스캔

연결 후 조작:

- `start/pause/resume/stop` 버튼 -> WebSocket `control`
- 텍스트 명령 입력 -> WebSocket `command`
- discussion 패널 -> `discussion_query`
- bridge는 이를 `POST /api/discuss` 또는 로컬 WS로 전달
- 폰은 `status`, `agent_state`, `video_frame`, discussion 응답을 받음

### 서로 어떻게 조작되는가

#### lifecycle 제어

```text
phone/web UI -> control(start|pause|resume|stop)
-> bridge.js
-> ws://127.0.0.1:8765/ws
-> AgentGate
```

#### 전략 변경

```text
phone/web UI -> command("Focus on science")
-> bridge.js
-> ws://127.0.0.1:8765/ws
-> CommandQueue
-> turn_executor checkpoint
-> strategy override applied
```

#### discussion 모드

```text
phone/web UI -> discussion_query
-> bridge.js
-> POST http://127.0.0.1:8765/api/discuss
-> Strategy discussion engine
-> answer returned to phone
```

## MCP 사용 패턴

일반적인 외부 제어 흐름:

1. `session_create`
2. `context_get` 또는 `workflow_observe`
3. `strategy_refine`
4. `action_route` / `action_plan` 또는 `workflow_step`
5. `hitl_send`
6. `hitl_status`

예시:

- `hitl_send(session_id, directive_type="start")`
- `hitl_send(session_id, directive_type="pause")`
- `hitl_send(session_id, directive_type="resume")`
- `hitl_send(session_id, directive_type="stop")`
- `hitl_send(session_id, directive_type="change_strategy", payload="Avoid war and rush Campus")`

## 개발

```bash
make lint
make format
make check
make test
make coverage
```
