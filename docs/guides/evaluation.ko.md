# 평가

이 저장소는 라이브 런타임일 뿐 아니라 action-level evaluation framework도 포함합니다.

## 주요 평가 영역

```text
civStation/evaluation/
  dataset/
  evaluator/
  metric/
```

## 두 가지 평가 트랙

| 트랙 | 언제 쓰나 | 핵심 아이디어 |
| --- | --- | --- |
| `bbox_eval` | 일반 action evaluation과 multi-answer datasets | action이 허용된 bbox 안에 들어오면 정답 |
| `civ6_eval` | 오래된 Civ6 전용 point-tolerance 흐름 | 고정 좌표와 tolerance window 사용 |

## 추천 경로: `bbox_eval`

Programmatic example:

```python
from civStation.evaluation.evaluator.action_eval.bbox_eval import (
    BuiltinAgentRunner,
    MockAgentRunner,
    SubprocessAgentRunner,
    run_evaluation,
)

report = run_evaluation("dataset.jsonl", MockAgentRunner())
```

CLI example:

```bash
python -m civStation.evaluation.evaluator.action_eval.bbox_eval \
  --dataset dataset.jsonl \
  --provider mock \
  --verbose
```

## fixtures와 integration tests

관련 파일:

- `tests/evaluator/civ6_eval/fixtures/ground_truth.json`
- `tests/evaluator/civ6_eval/fixtures/sample_bbox_dataset.jsonl`
- `tests/evaluator/civ6_eval/fixtures/screenshots/README.md`

스크린샷 fixture 디렉터리는 버전 관리에는 비워 둔 채, 로컬에서만 실제 이미지를 넣어 쓰는 구조입니다.

## 연구/논문 산출물

논문 관련 validation artifacts는 다음 경로 아래에 있습니다.

```text
paper/arxiv/results/
```

이 파일들은 leaderboard용 벤치마크가 아니라 paper draft를 지원하는 산출물입니다.

## 언제 evaluation을 돌려야 하나

- primitive logic을 바꾼 뒤
- parser나 action schema를 바꾼 뒤
- 이미지 전처리 기본값을 바꾼 뒤
- routing이나 planning 개선을 주장하기 전에

더 넓은 테스트 표면은 [테스트와 품질](../development/testing-and-quality.md)을 보세요.
