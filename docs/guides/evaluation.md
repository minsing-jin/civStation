# Evaluation

The repository is not only a live agent runtime. It also includes an evaluation framework for action-level testing.

## Main Evaluation Areas

```text
civStation/evaluation/
  dataset/
  evaluator/
  metric/
```

## Two Evaluation Tracks

| Track | When to use it | Key idea |
| --- | --- | --- |
| `bbox_eval` | general action evaluation and multi-answer datasets | an action is correct if it lands inside an accepted bbox |
| `civ6_eval` | older Civ6-specific point-tolerance flow | fixed point targets with a tolerance window |

## Recommended Path: `bbox_eval`

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

## Fixtures and Integration Tests

Relevant files:

- `tests/evaluator/civ6_eval/fixtures/ground_truth.json`
- `tests/evaluator/civ6_eval/fixtures/sample_bbox_dataset.jsonl`
- `tests/evaluator/civ6_eval/fixtures/screenshots/README.md`

The screenshot fixture folder intentionally stays empty in version control. Real screenshots can be added locally for integration tests.

## Research and Paper Artifacts

The repository also contains paper-oriented validation artifacts under:

```text
paper/arxiv/results/
```

Those files are supporting artifacts for the paper draft, not leaderboard-style benchmark claims.

## When to Reach for Evaluation

- after changing primitive logic
- after changing parser or action schema behavior
- after changing image preprocessing defaults
- before claiming a routing or planning improvement

Use [Testing and Quality](../development/testing-and-quality.md) for the broader test matrix.
