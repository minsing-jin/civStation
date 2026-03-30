# 실행 루프

라이브 루프의 중심은 `turn_executor.py`입니다.

## 상위 흐름

```text
turn_runner.py
  -> providers, HITL, status UI, knowledge, logging 설정
  -> run_multi_turn()
      -> run_one_turn()
          -> screen capture
          -> context update / read
          -> primitive routing
          -> action planning
          -> action execution
          -> outcome 기록
          -> checkpoints와 directives 처리
```

## 구체적인 단계

### 1. Observation

화면을 캡처하고, 필요하면 호출 지점별로 이미지를 전처리합니다.

### 2. Routing

router provider가 스크린샷을 `policy_primitive`, `city_production_primitive` 같은 primitive로 분류합니다.

### 3. Planning

planner provider가 그 primitive에 대한 다음 행동을 생성합니다. 단일 클릭일 수도 있고 multi-step process일 수도 있습니다.

### 4. Execution

행동은 normalized coordinate space로 표현되고, 실행 직전에 실제 화면 좌표로 변환됩니다.

### 5. Recording과 interrupt check

실행 결과를 기록하고 상태를 갱신한 뒤, queued directives가 pause, stop, override를 요구하는지 확인합니다.

## normalized coordinates가 중요한 이유

모델이 실제 화면 해상도를 직접 알 필요가 없습니다.

대신 `--range`가 정의하는 공유 좌표계를 사용하고, 런타임이 마지막에 실제 좌표로 변환합니다. 그래서 서로 다른 디스플레이에서도 재사용하기 쉬워집니다.

## 백그라운드 도우미

눈에 보이는 route-plan-execute 체인 외에도 다음 구성요소가 함께 작동합니다.

- `ContextUpdater`
- `TurnDetector`
- `StrategyUpdater`
- `InterruptMonitor`
- `TurnCheckpoint`

이 덕분에 모든 일을 하나의 동기식 모델 호출에 몰아넣지 않고도 루프를 반응성 있게 유지할 수 있습니다.

## 디버깅의 첫 지점

라이브 루프가 잘못 동작하면 가장 먼저 볼 로그:

```text
.tmp/civStation/turn_runner_latest.log
```

이 경로가 프로젝트의 latest run cache입니다.
