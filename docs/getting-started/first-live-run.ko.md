# 첫 실전 실행

이 페이지는 프로세스를 어떻게 띄우느냐보다, 뜬 뒤에 무엇이 일어나는지에 초점을 둡니다.

## 추천 실행 명령

```bash
python -m civStation.agent.turn_runner \
  --provider gemini \
  --router-provider gemini \
  --planner-provider claude \
  --turns 100 \
  --strategy "Focus on science victory and reinforce scouting." \
  --status-ui \
  --wait-for-start \
  --status-port 8765
```

routing은 planning보다 싸고 빠른 경우가 많기 때문에 이런 분리가 실전적으로 유용합니다.

## 예상 동작

1. 프로세스가 부팅되고 provider를 초기화합니다.
2. status UI가 열릴 준비를 합니다.
3. `--wait-for-start`가 있으면 pre-start 상태에서 대기합니다.
4. start 후에는 각 턴이 observation, routing, planning, execution, checkpoint 처리 순으로 진행됩니다.

## 대시보드가 주는 것

- 라이브 status snapshot
- agent lifecycle control
- strategy 또는 자유 형식 directive
- discussion mode가 켜진 경우 discussion endpoints
- 외부 controller를 위한 WebSocket 채널

이 dashboard는 단순 모니터가 아니라 제어 표면입니다.

## 가장 빠르게 쓸 수 있는 명령

Lifecycle:

```bash
curl -X POST http://127.0.0.1:8765/api/agent/start
curl -X POST http://127.0.0.1:8765/api/agent/pause
curl -X POST http://127.0.0.1:8765/api/agent/resume
curl -X POST http://127.0.0.1:8765/api/agent/stop
```

Directive:

```bash
curl -X POST http://127.0.0.1:8765/api/directive \
  -H "Content-Type: application/json" \
  -d '{"text":"Switch to culture victory and stop expanding for now."}'
```

## 로그와 산출물

디버깅을 시작할 때 가장 먼저 볼 파일:

```text
.tmp/civStation/turn_runner_latest.log
```

이 경로는 프로젝트의 기본 latest-run log cache입니다. 실행이 이상하거나 멈췄을 때 첫 번째 확인 지점입니다.

## 운영 팁

- setup을 신뢰하기 전까지는 `--wait-for-start`를 켜 두세요.
- 개발 중에는 `--status-ui`를 끄지 마세요.
- 비용을 줄이면서 planner 품질을 유지하려면 router와 planner provider를 분리하세요.
- UI 제어만으로 부족하면 MCP 서버를 같이 띄우세요.
