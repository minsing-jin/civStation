# 总督秘密结社观察与追踪实现计划

这份文档是关于 `governor_primitive` 的 observation scroll、secret-society 分支与 Rich trace 行为改进计划。

## 目标

- 强制 governor primitive 至少执行一次 downward observation scroll
- 增加 secret-society appointment branch，并允许合流到 promotion
- 用 tests 锁定 governor 相关 Rich trace 行为

## 架构

- 所有逻辑保持在 `GovernorProcess` 内部
- observation、branch selection 与 branch merge 都由代码显式控制
- 不增加第二套 trace 系统，而是复用 `turn_executor.py` 的 runtime trace feed

## 实现重点

- 增加 governor observation regression tests
- 引入至少一次 downward scan 的规则
- 只有在 no-new-candidates / end-of-list 确认后才进入选择阶段

这一页是计划摘要；原文包含更细的测试驱动实施步骤。
