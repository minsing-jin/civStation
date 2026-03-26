# 城市生产蓝色地块放置后续实现计划

这份设计说明描述了在 `city production placement` 之后出现可购买蓝色地块时的后续处理流程。

## 核心目标

- 处理带有蓝色可购买地块的放置界面
- 在最终 build confirmation 前，对同一地块执行 deterministic re-click
- 在 placement prompt 中显式加入 current gold、adjacency 与 strategy reasoning

## 设计摘要

- 在 `CityProductionProcess` 中增加轻量级 post-placement resolver stage
- 在 short-term memory 中保存最近一次 placement click
- 根据界面仍处于 placement 还是进入 confirm 阶段来分支

## 实现重点

- 用 `tests/agent/modules/primitive/test_multi_step_process.py` 锁定行为
- 在 `civStation/agent/modules/memory/short_term_memory.py` 中加入 follow-up state
- 在 `civStation/agent/modules/primitive/multi_step_process.py` 中加入后续 stage

原文包含更细的 task-by-task 计划，这一页是中文摘要。
