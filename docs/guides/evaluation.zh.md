# 评测

这个仓库不仅包含实时代理运行时，也包含 action-level evaluation framework。

## 主要评测区域

```text
civStation/evaluation/
  dataset/
  evaluator/
  metric/
```

## 两条评测路线

| 路线 | 适用场景 | 核心思路 |
| --- | --- | --- |
| `bbox_eval` | 通用动作评测与 multi-answer datasets | 动作落在允许 bbox 内即算正确 |
| `civ6_eval` | 旧版 Civ6 专用 point-tolerance 流程 | 使用固定坐标和 tolerance window |

## 推荐路径：`bbox_eval`

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

## fixtures 与 integration tests

相关文件：

- `tests/evaluator/civ6_eval/fixtures/ground_truth.json`
- `tests/evaluator/civ6_eval/fixtures/sample_bbox_dataset.jsonl`
- `tests/evaluator/civ6_eval/fixtures/screenshots/README.md`

截图 fixture 目录在版本控制中保持为空，真实截图只在本地添加。

## 研究/论文产物

与论文相关的 validation artifacts 位于：

```text
paper/arxiv/results/
```

这些文件不是 leaderboard 式基准，而是用来支持 paper draft 的材料。

## 什么时候该跑评测

- 修改 primitive logic 之后
- 修改 parser 或 action schema 之后
- 修改图像预处理默认值之后
- 在声称 routing 或 planning 有改进之前

更完整的测试矩阵请看 [测试与质量](../development/testing-and-quality.md)。
