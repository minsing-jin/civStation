# Human-in-the-Loop

`HitL`은 부가 기능이 아니라 핵심 레이어 중 하나입니다.

## 이 레이어가 답하는 것

```text
How can a human supervise, interrupt, or redirect the agent while it is running?
```

## 제어 표면

### 로컬 dashboard

내장 FastAPI dashboard는 다음을 제공합니다.

- 브라우저 UI
- REST endpoints
- WebSocket connection
- screen/status streaming

### 직접 HTTP와 WebSocket 제어

가장 가벼운 외부 제어 경로입니다. 로컬 스크립트, 커스텀 dashboard, 운영 도구에 적합합니다.

### 원격 relay/controller

별도 `tacticall` controller 저장소를 통해 relay 기반 phone controller와 연결할 수 있습니다.

## directive priority

여러 directive가 동시에 쌓이면 런타임은 다음 우선순위로 처리합니다.

```text
STOP > PRIMITIVE_OVERRIDE > PAUSE > CHANGE_STRATEGY
```

이 순서는 긴급 중지의 신뢰성을 보장하는 안전장치입니다.

## 대표적인 개입

- start, pause, resume, stop
- 고수준 strategy 변경
- primitive override 강제
- 직접 command 주입
- 현재 run에 대한 discussion과 strategy finalize

## 핵심 파일

- `command_queue.py`
- `agent_gate.py`
- `turn_checkpoint.py`
- `status_ui/server.py`
- `status_ui/state_bridge.py`
- `relay/relay_client.py`

## 어떤 표면을 언제 쓰나

| 필요 | 가장 적합한 표면 |
| --- | --- |
| 로컬 수동 제어 | 내장 dashboard |
| 도구 기반 로컬 제어 | REST + WebSocket |
| 원격 모바일 제어 | relay + phone controller |
| 구조화된 외부 orchestration | MCP tools + `hitl_*` |

구체적인 endpoint 예시는 [제어와 디스커션](../guides/control-and-discussion.md)을 보세요.
