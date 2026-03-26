# 에이전트 실행

실전에서 자주 쓰는 실행 패턴을 정리한 페이지입니다.

## 최소 로컬 실행

```bash
python -m civStation.agent.turn_runner \
  --provider gemini \
  --turns 50 \
  --strategy "Focus on science victory" \
  --status-ui
```

하나의 provider와 로컬 dashboard만으로 시작할 때 적합합니다.

## Router와 Planner 분리

```bash
python -m civStation.agent.turn_runner \
  --router-provider gemini \
  --planner-provider claude \
  --turns 100 \
  --strategy "Prioritize Campus and early scouting." \
  --status-ui
```

routing 비용을 낮추면서 planning 품질을 유지하고 싶을 때 적합합니다.

## 외부 start 신호를 기다리기

```bash
python -m civStation.agent.turn_runner \
  --provider claude \
  --status-ui \
  --wait-for-start
```

라이브 실험에서는 가장 안전한 기본값입니다.

## `config.yaml` 기반 실행

```bash
python -m civStation.agent.turn_runner --config config.yaml
```

저장소의 기본 config는 다음을 이미 포함합니다.

- provider와 model
- turns
- strategy
- HITL flags
- status UI
- image pipeline overrides

## Remote relay 모드

```bash
python -m civStation.agent.turn_runner \
  --provider gemini \
  --relay-url ws://127.0.0.1:8787/ws \
  --relay-token "$RELAY_TOKEN"
```

로컬 dashboard만으로는 부족하고 외부 relay로 제어해야 할 때 사용합니다.

## 자주 쓰는 런타임 플래그

| 필요 | 플래그 |
| --- | --- |
| turn 수 | `--turns` |
| strategy | `--strategy` |
| 로컬 dashboard | `--status-ui --status-port` |
| 안전한 부팅 | `--wait-for-start` |
| 모델 분리 | `--router-provider`, `--planner-provider`, `--turn-detector-provider` |
| prompt 언어 | `--prompt-language eng|kor` |
| 디버그 가시성 | `--debug context,turns` 또는 `--debug all` |

## 실전 기본값

- 개발 중에는 `--status-ui`를 유지
- setup을 신뢰하기 전에는 `--wait-for-start`를 유지
- strategy는 한 문장으로 명확하게 시작
- provider와 image 설정은 한 번에 하나씩만 바꾸기

전체 플래그는 [CLI 레퍼런스](../reference/cli.md)를 보세요.
