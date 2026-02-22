# Civ6 Computer Use Agent

Vision-Language Model(VLM) 기반 Civilization VI 자율 게임 에이전트.
스크린샷을 캡처하고 게임 상태를 분류(라우팅)한 뒤, 특화된 Primitive가 정규화 좌표로 액션을 생성하여 PyAutoGUI로 실행합니다.

---

## 아키텍처

### 핵심 플로우 (1 Turn)

```
스크린샷 캡처
    │
    ▼
Router VLM ─── 게임 상태 분류 ──▶ Primitive 선택
    │
    ▼
Planner VLM ─── 액션 생성 (정규화 좌표 0-1000)
    │
    ▼
execute_action() ─── 좌표 변환 ──▶ PyAutoGUI 실행
```

### 주요 모듈

```
computer_use_test/
├── agent/
│   ├── turn_runner.py          # CLI 진입점 (ConfigArgParse + config.yaml)
│   ├── turn_executor.py        # run_one_turn / run_multi_turn 실행 로직
│   └── modules/
│       ├── router/
│       │   └── primitive_registry.py   # Primitive 중앙 레지스트리
│       ├── hitl/
│       │   ├── command_queue.py        # 스레드-안전 directive 큐
│       │   ├── agent_gate.py           # 외부 제어 수신 게이트 (시작/정지/일시정지)
│       │   └── queue_listener.py       # stdin / voice 입력 리스너
│       ├── status_ui/
│       │   ├── server.py               # FastAPI 서버 (REST + WebSocket)
│       │   ├── state_bridge.py         # 에이전트↔UI 스레드 브리지
│       │   ├── websocket_manager.py    # WS 연결 관리 + 브로드캐스트
│       │   └── dashboard.py            # 실시간 대시보드 HTML/JS
│       ├── context/
│       │   ├── context_manager.py      # 게임 상태 컨텍스트
│       │   ├── context_updater.py      # 백그라운드 화면 분석
│       │   └── macro_turn_manager.py   # 게임 턴 경계 감지
│       └── strategy/
│           └── strategy_planner.py     # 전략 생성 / HITL 정제
└── utils/
    ├── llm_provider/           # Claude / Gemini / GPT / Mock 프로바이더
    └── screen.py               # 스크린샷 캡처 + 좌표 변환 (Retina 대응)
```

### Primitive 시스템

게임 상태를 10개의 전문 핸들러로 분류합니다.

| Primitive | 담당 |
|---|---|
| `unit_ops_primitive` | 유닛 이동·생산 |
| `popup_primitive` | 팝업·알림 처리 |
| `research_select_primitive` | 기술 연구 선택 |
| `city_production_primitive` | 도시 생산 설정 |
| `science_decision_primitive` | 과학 관련 결정 |
| `culture_decision_primitive` | 문화 관련 결정 |
| `governor_primitive` | 총독 배치·업그레이드 |
| `diplomatic_primitive` | 외교 처리 |
| `combat_primitive` | 전투 처리 |
| `policy_primitive` | 정책 카드 설정 |

새 Primitive 추가: `primitive_registry.py`에 항목 추가만 하면 Router 프롬프트가 자동 갱신됩니다.

### HITL (Human-in-the-Loop) 시스템

```
외부 웹 컨트롤러
    │  HTTP/WebSocket
    ▼
FastAPI (server.py)
    │
    ├── AgentGate ─── 시작/정지/일시정지 상태 관리
    │
    └── CommandQueue ─── Directive 큐
            │
            ▼
        turn_executor ─── 턴 실행 중 큐 체크
```

**Directive 우선순위**: `STOP` > `PRIMITIVE_OVERRIDE` > `PAUSE` > `CHANGE_STRATEGY`

---

## 설치

```bash
# 의존성 설치 + pre-commit 훅 설정
make install

# 또는 직접 설치
pip install -e ".[ui]"   # FastAPI + uvicorn 포함
```

### 환경변수

프로젝트 루트에 `.env` 파일 생성:

```env
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
OPENAI_API_KEY=sk-...      # 선택사항
```

---

## 실행 방법

### 기본 실행

```bash
# config.yaml 설정 사용 (기본값)
python -m computer_use_test.agent.turn_runner

# CLI로 직접 지정
python -m computer_use_test.agent.turn_runner \
    --provider claude \
    --turns 20 \
    --strategy "과학 승리에 집중"
```

### 실시간 대시보드 (상태 모니터링)

```bash
python -m computer_use_test.agent.turn_runner \
    --status-ui \
    --status-port 8765 \
    --turns 50
```

브라우저에서 `http://localhost:8765` 접속.

### 모바일 QR 연결

1. `--status-ui`로 에이전트 실행
2. PC 브라우저에서 `http://localhost:8765` 접속
3. 헤더 우측 상단 **"QR Connect"** 버튼 클릭
4. 휴대폰 카메라로 QR 코드 스캔 (같은 Wi-Fi 필수)

> QR 코드는 서버의 LAN IP(`http://192.168.x.x:8765`)를 자동 감지하여 생성합니다.
> QR 라이브러리를 CDN에서 로드하므로 처음 한 번은 인터넷 연결이 필요합니다.

### 외부 컨트롤러로 시작 제어

에이전트가 서버를 먼저 띄우고 외부 신호를 기다리는 모드:

```bash
python -m computer_use_test.agent.turn_runner \
    --status-ui \
    --wait-for-start \
    --turns 100
```

외부 컨트롤러에서 HTTP로 제어:

```bash
# 시작
curl -X POST http://192.168.x.x:8765/api/agent/start

# 일시정지
curl -X POST http://192.168.x.x:8765/api/agent/pause

# 재개
curl -X POST http://192.168.x.x:8765/api/agent/resume

# 정지
curl -X POST http://192.168.x.x:8765/api/agent/stop

# 현재 상태 조회
curl http://192.168.x.x:8765/api/agent/state
# → {"state": "running"}  # idle / running / paused / stopped
```

### 라우터/플래너 분리

```bash
python -m computer_use_test.agent.turn_runner \
    --router-provider gemini --router-model gemini-2.0-flash \
    --planner-provider claude --planner-model claude-sonnet-4-5
```

### config.yaml 수정으로 설정

```yaml
# config.yaml
provider: gemini
model: gemini-3-flash-preview
turns: 10
strategy: "과학 승리에 집중하고 정찰을 강화해."
status-ui: true
```

---

## API 엔드포인트 (status-ui)

| Method | Endpoint | 설명 |
|---|---|---|
| `GET` | `/` | 실시간 대시보드 |
| `GET` | `/api/status` | 에이전트 상태 스냅샷 (JSON) |
| `POST` | `/api/directive` | 텍스트 directive 전송 |
| `GET` | `/api/agent/state` | 에이전트 라이프사이클 상태 |
| `POST` | `/api/agent/start` | 에이전트 시작 |
| `POST` | `/api/agent/pause` | 일시정지 |
| `POST` | `/api/agent/resume` | 재개 |
| `POST` | `/api/agent/stop` | 정지 |
| `GET` | `/api/connection-info` | LAN IP + 접속 URL |
| `WS` | `/ws` | WebSocket 실시간 채널 |

**WebSocket 메시지 형식 (클라이언트 → 서버)**:

```json
{"mode": "high_level", "content": "문화 승리로 전략 변경"}
{"mode": "primitive", "content": "{\"action\":\"click\",\"x\":500,\"y\":300}"}
{"mode": "pause"}
{"mode": "resume"}
{"mode": "stop"}
```

---

## 개발

```bash
# Lint
make lint

# Format
make format

# 타입 체크 + Lint
make check

# 테스트
make test

# 커버리지
make coverage
```

---

## 좌표 정규화

VLM은 항상 `0 ~ normalizing_range` (기본 1000) 범위 좌표를 사용합니다.
실제 화면 좌표 변환은 `screen.py`의 `norm_to_real()`이 담당합니다.
Mac Retina 디스플레이의 논리/물리 픽셀 불일치도 자동으로 처리합니다.

---

## License

MIT