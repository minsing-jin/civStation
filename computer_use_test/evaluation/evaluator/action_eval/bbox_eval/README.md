# 📦 Bbox-Based Static Screenshot Evaluation

## 📚 Index

- [✨ 특징](#-특징)
- [🗂️ 패키지 구조](#-패키지-구조)
- [🚀 Quick Start](#-quick-start)
- [🧾 JSONL 데이터셋 포맷](#-jsonl-데이터셋-포맷)

**Bounding-box 기반 정적 스크린샷 평가 프레임워크**

게임 에이전트가 예측한 포인트 좌표 액션을 Ground Truth 바운딩 박스 타겟과 비교하여 평가합니다.

## ✨ 특징

- **Bounding Box GT**: 기존 ±5px tolerance 대신 바운딩 박스 영역으로 정답 정의
- **Multiple GT Sets**: 케이스당 여러 개의 허용 가능한 정답 액션 시퀀스 지원
- **외부 에이전트 연동**: stdin/stdout JSON 프로토콜로 임의의 에이전트 통합
- **내장 VLM Provider**: Claude, Gemini, GPT 등 기존 프로바이더 래핑
- **풍부한 메트릭**: `strict_success`, `prefix_len`, `step_accuracy`, 액션 타입별 breakdown

## 🗂️ 패키지 구조

```
bbox_eval/
├── __init__.py           # Public API (모든 export 여기서)
├── __main__.py           # python -m ... 지원
├── schema.py             # Pydantic 모델 (BBox, GT Actions, Results, Report)
├── scorer.py             # 스코어링 알고리즘
├── dataset_loader.py     # JSONL 데이터셋 로딩/검증
├── runner.py             # 평가 오케스트레이터
├── cli.py                # CLI 엔트리 포인트
├── agents/
│   ├── __init__.py       # Agent runner exports
│   ├── base.py           # BaseAgentRunner ABC
│   ├── subprocess_runner.py  # 외부 프로세스 에이전트
│   ├── builtin_adapter.py    # 내장 VLM 프로바이더 래퍼
│   └── mock_runner.py        # 테스트용 Mock
└── README.md
```

## 🚀 Quick Start

### 1. Python API

```python
from computer_use_test.evaluation.evaluator.action_eval.bbox_eval import (
    run_evaluation,
    MockAgentRunner,
)

# Mock 에이전트로 평가 실행
report = run_evaluation(
    dataset_path="path/to/dataset.jsonl",
    runner=MockAgentRunner(),
    verbose=True,
)

print(f"Total: {report.aggregate.total_cases}")
print(f"Success rate: {report.aggregate.strict_success_rate:.1%}")
print(f"Avg accuracy: {report.aggregate.avg_step_accuracy:.1%}")
```

### 2. CLI

```bash
# Mock 프로바이더로 테스트
python -m computer_use_test.evaluation.evaluator.action_eval.bbox_eval \
    --dataset path/to/dataset.jsonl \
    --provider mock --verbose

# Claude VLM으로 평가
python -m computer_use_test.evaluation.evaluator.action_eval.bbox_eval \
    --dataset path/to/dataset.jsonl \
    --provider claude \
    --model claude-4-5-sonnet-20241022 \
    --output results.json

# 외부 에이전트 (stdin/stdout)
python -m computer_use_test.evaluation.evaluator.action_eval.bbox_eval \
    --dataset path/to/dataset.jsonl \
    --agent-cmd "python my_agent.py" \
    --timeout 30
```

### 3. CLI 옵션

| 옵션 | 설명 |
|------|------|
| `--dataset` | JSONL 데이터셋 파일 경로 (필수) |
| `--provider` | 내장 VLM 프로바이더 (claude, gemini, gpt, mock) |
| `--agent-cmd` | 외부 에이전트 명령어 |
| `--model` | 프로바이더 모델 ID |
| `--timeout` | 케이스당 타임아웃 초 (기본: 10) |
| `--ignore-wait` | Wait 액션 무시 |
| `--output` | JSON 리포트 출력 경로 |
| `--verbose` | 상세 로깅 |

> `--provider`와 `--agent-cmd`는 상호 배타적입니다.

## 🧾 JSONL 데이터셋 포맷

각 줄이 하나의 평가 케이스입니다:

```json
{
  "case_id": "civ6_science_001",
  "instruction": "Select the Pottery technology for research",
  "screenshot_path": "screenshots/turn_10_science.png",
  "image_size": {"width": 1000, "height": 1000},
  "action_sets": [
    {
      "actions": [
        {"type": "click", "target_bbox": {"x_min": 80, "y_min": 180, "x_max": 120, "y_max": 220}, "button": "left"},
        {"type": "press", "keys": ["enter"]}
      ]
    },
    {
      "actions": [
        {"type": "click", "target_bbox": {"x_min": 75, "y_min": 175, "x_max": 125, "y_max": 225}, "button": "left"}
      ]
    }
  ],
  "metadata": {"turn": 10, "primitive": "science_decision_primitive"}
}
```

### GT Action 타입

| 타입 | 필드 | 설명 |
|------|------|------|
| `click` | `target_bbox`, `button` | 클릭 영역 |
| `double_click` | `target_bbox`, `button` | 더블클릭 영역 |
| `drag` | `start_bbox`, `end_bbox`, `button` | 드래그 시작/끝 영역 |
| `press` | `keys` | 키보드 입력 |
| `wait` | `duration` | 대기 |

에이전트 예측은 기존 `Action` 스키마 (포인트 좌표, 0-1000 정규화)를 사용합니다.

## 스코어링 알고리즘

### Step 비교

| GT 타입 | 정답 조건 | 진단 정보 |
|---------|----------|-----------|
| click/double_click | `pred.(x,y)` ∈ `gt.target_bbox` AND 버튼 일치 | bbox 중심까지 거리 |
| drag | `pred.start` ∈ `gt.start_bbox` AND `pred.end` ∈ `gt.end_bbox` | - |
| press | `pred.keys == gt.keys` (정확 일치) | Levenshtein 유사도 |
| wait | `ignore_wait=True`이면 항상 정답, 아니면 타입 일치 | - |

### Sequence 비교

1. `ignore_wait=True`이면 Wait 액션 필터링
2. 쌍을 이루는 step들을 비교
3. **prefix_len**: index 0부터 연속 정답 수
4. **step_accuracy**: 정답 step 수 / GT 액션 수
5. **strict_success**: 길이 동일 AND 모든 step 정답

### Best GT Set 선택

여러 GT action_set 중 예측과 가장 잘 맞는 것을 선택:
`strict_success (desc) > prefix_len (desc) > step_accuracy (desc)`

## 커스텀 에이전트 연동

### 방법 1: Python 클래스 (BaseAgentRunner 상속)

```python
from computer_use_test.evaluation.evaluator.action_eval.bbox_eval import (
    BaseAgentRunner,
    AgentResponse,
    DatasetCase,
    run_evaluation,
)
from computer_use_test.agent.models.schema import ClickAction


class MyGameAgent(BaseAgentRunner):
    def __init__(self, model_path: str):
        self.model = load_my_model(model_path)

    def run_case(self, case: DatasetCase) -> AgentResponse:
        # 스크린샷 분석
        prediction = self.model.predict(
            image_path=case.screenshot_path,
            instruction=case.instruction,
        )
        # Action 객체로 변환
        actions = [
            ClickAction(type="click", x=prediction.x, y=prediction.y, button="left")
        ]
        return AgentResponse(actions=actions, meta={"model": "my_model"})


# 평가 실행
report = run_evaluation("dataset.jsonl", MyGameAgent("model.pt"))
```

### 방법 2: 외부 프로세스 (stdin/stdout)

에이전트를 별도 프로세스로 실행합니다:

**my_agent.py:**
```python
import sys, json

# stdin에서 요청 읽기
request = json.loads(sys.stdin.read())
instruction = request["instruction"]
screenshot = request["screenshot_path"]
size = request["image_size"]

# 에이전트 로직
actions = [{"type": "click", "x": 100, "y": 200, "button": "left"}]

# stdout으로 응답
print(json.dumps({"actions": actions, "meta": {"agent": "my_agent"}}))
```

```bash
python -m computer_use_test.evaluation.evaluator.action_eval.bbox_eval \
    --dataset dataset.jsonl \
    --agent-cmd "python my_agent.py" \
    --timeout 30
```

## 리포트 형식

```json
{
  "aggregate": {
    "total_cases": 100,
    "strict_success_rate": 0.75,
    "avg_step_accuracy": 0.88,
    "avg_prefix_len": 1.5,
    "error_count": 2,
    "timeout_count": 1,
    "per_action_type": [
      {"action_type": "click", "total": 120, "correct": 105, "accuracy": 0.875},
      {"action_type": "press", "total": 50, "correct": 48, "accuracy": 0.96}
    ]
  },
  "cases": [...],
  "config": {...},
  "timestamp": "2026-02-27T10:00:00"
}
```

## 데이터셋 검증

```python
from computer_use_test.evaluation.evaluator.action_eval.bbox_eval import validate_dataset

valid_count, errors = validate_dataset("dataset.jsonl")
print(f"Valid: {valid_count}, Errors: {len(errors)}")
for err in errors:
    print(f"  {err}")
```

## 테스트

```bash
# 모든 bbox eval 테스트
pytest tests/evaluator/civ6_eval/test_bbox_schema.py -v
pytest tests/evaluator/civ6_eval/test_bbox_scorer.py -v
pytest tests/evaluator/civ6_eval/test_dataset_loader.py -v
pytest tests/evaluator/civ6_eval/test_agent_runner.py -v
pytest tests/evaluator/civ6_eval/test_bbox_eval_integration.py -v

# 전체 테스트 (기존 테스트 포함)
pytest tests/evaluator/civ6_eval/ -v
```

## 다른 게임으로 확장

이 프레임워크는 Civ6에 국한되지 않습니다. 다른 게임에 적용하려면:

1. **JSONL 데이터셋 생성**: 게임 스크린샷 + GT 바운딩 박스 어노테이션
2. **에이전트 구현**: `BaseAgentRunner`를 상속하거나 stdin/stdout 프로토콜 사용
3. **평가 실행**: `run_evaluation()` 또는 CLI로 실행

스키마와 스코어링은 게임에 무관하게 동작합니다.
