# Evaluation Framework

게임 에이전트의 행동을 정량적으로 평가하기 위한 프레임워크입니다.

## 아키텍처

```
evaluation/
├── dataset/                          # 데이터셋 수집 및 관리
│   ├── static_dataset_collector.py   # 스크린샷 + GT 수집 도구
│   └── static_screenshot/            # 예제 JSONL 데이터셋
├── evaluator/
│   ├── action_eval/                  # 액션 레벨 평가
│   │   ├── interfaces.py             # 기존 평가 인터페이스 (GroundTruth, EvalResult)
│   │   ├── base_static_primitive_evaluator.py  # 기존 프리미티브 평가기
│   │   ├── civ6_eval/                # Civ6 전용 5px tolerance 평가기
│   │   └── bbox_eval/               # ★ Bbox 기반 평가 프레임워크 (NEW)
│   └── strategy_eval/                # 전략 레벨 평가 (계획 중)
└── metric/                           # 메트릭 정의
```

## 평가 방식 비교

| 기능 | civ6_eval (기존) | bbox_eval (신규) |
|------|------------------|------------------|
| GT 포맷 | 포인트 좌표 (x, y) | 바운딩 박스 (x_min, y_min, x_max, y_max) |
| 허용 오차 | ±5px tolerance | 포인트가 bbox 내에 있으면 정답 |
| GT 세트 | 케이스당 1개 | 케이스당 N개 (다중 정답) |
| 에이전트 연동 | 내장 Router + Primitive 고정 | stdin/stdout 또는 VLM Provider |
| 메트릭 | primitive_match, action_sequence_match | strict_success, prefix_len, step_accuracy |
| 데이터 포맷 | JSON (배열) | JSONL (라인별 독립) |
| CLI | main.py | `python -m ...bbox_eval` |

## Quick Start

### Bbox 평가 (추천)

```python
from computer_use_test.evaluation.evaluator.action_eval.bbox_eval import (
    run_evaluation,
    MockAgentRunner,
    BuiltinAgentRunner,
    SubprocessAgentRunner,
)

# 1. Mock으로 빠른 테스트
report = run_evaluation("dataset.jsonl", MockAgentRunner())

# 2. 내장 VLM Provider
from computer_use_test.utils.llm_provider import create_provider
provider = create_provider("claude")
report = run_evaluation("dataset.jsonl", BuiltinAgentRunner(provider))

# 3. 외부 에이전트
report = run_evaluation("dataset.jsonl", SubprocessAgentRunner("python my_agent.py"))

# 결과 확인
print(f"Success: {report.aggregate.strict_success_rate:.1%}")
```

```bash
# CLI
python -m computer_use_test.evaluation.evaluator.action_eval.bbox_eval \
    --dataset dataset.jsonl --provider mock --verbose
```

자세한 문서는 [bbox_eval/README.md](evaluator/action_eval/bbox_eval/README.md)를 참고하세요.

### 기존 Civ6 평가

```python
from computer_use_test.evaluation.evaluator.action_eval.civ6_eval import Civ6StaticEvaluator
```

자세한 문서는 [civ6_eval/README.md](evaluator/action_eval/civ6_eval/README.md)를 참고하세요.

## 오픈소스 기여 가이드

### 커스텀 에이전트 추가

`BaseAgentRunner`를 상속하여 `run_case()` 메서드만 구현하면 됩니다:

```python
from computer_use_test.evaluation.evaluator.action_eval.bbox_eval import (
    BaseAgentRunner, AgentResponse, DatasetCase
)

class MyAgent(BaseAgentRunner):
    def run_case(self, case: DatasetCase) -> AgentResponse:
        # 여러분의 에이전트 로직
        return AgentResponse(actions=[...])
```

### 새로운 평가 메트릭 추가

`scorer.py`의 `compare_step()` 함수에 새로운 비교 로직을 추가할 수 있습니다.

### 새로운 게임 도메인 추가

1. JSONL 데이터셋 생성 (포맷은 bbox_eval README 참고)
2. `BaseAgentRunner` 구현
3. `run_evaluation()` 실행

프레임워크의 스키마와 스코어링은 게임에 무관하게 설계되어 있습니다.
