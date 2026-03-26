# 제어와 디스커션

이 페이지는 런타임 제어 표면을 운영자 관점에서 설명합니다.

## REST Endpoints

Agent lifecycle:

```text
GET  /api/agent/state
POST /api/agent/start
POST /api/agent/pause
POST /api/agent/resume
POST /api/agent/stop
```

Directive와 status:

```text
GET  /api/status
GET  /api/connection-info
POST /api/directive
```

Discussion:

```text
POST /api/discuss
POST /api/discuss/finalize
GET  /api/discuss/status
```

## WebSocket

기본 소켓:

```text
ws://127.0.0.1:8765/ws
```

예시 메시지:

```json
{ "type": "control", "action": "start" }
{ "type": "control", "action": "pause" }
{ "type": "control", "action": "resume" }
{ "type": "control", "action": "stop" }
{ "type": "command", "content": "Prioritize Campus and stop training settlers." }
```

## strategy directive 보내기

```bash
curl -X POST http://127.0.0.1:8765/api/directive \
  -H "Content-Type: application/json" \
  -d '{"text":"Focus on culture victory and avoid war for the next 10 turns."}'
```

`stop`, `pause`, `resume` 같은 quick command는 lifecycle directive로 해석됩니다.

## discussion mode

discussion이 활성화되어 있으면, 프로젝트는 strategy discussion session을 유지하고 그 결과를 최종 strategy update로 정리할 수 있습니다.

예시:

```bash
curl -X POST http://127.0.0.1:8765/api/discuss \
  -H "Content-Type: application/json" \
  -d '{
        "user_id":"operator",
        "message":"We are over-expanding. Tighten economy and tech first.",
        "mode":"in_game",
        "language":"ko"
      }'
```

## 원격 phone controller

원격 phone controller는 별도 `minsing-jin/tacticall` 저장소에 있습니다.

상위 흐름:

```text
Phone browser
  <-> relay server
  <-> host bridge
  <-> local agent websocket
  <-> local discussion API
```

로컬 브라우저 제어보다 더 유연한 모바일 제어가 필요할 때 relay 모드를 사용하세요.

## MCP 매핑

가장 가까운 MCP tools:

- `hitl_send`
- `hitl_status`

controller가 브라우저가 아니라 다른 agent나 skill일 때는 MCP가 더 적합합니다.
