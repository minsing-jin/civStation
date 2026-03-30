# Turn Runner Raw Log Cache 设计

这份文档讨论的是：在保留 `turn_runner` 的 Rich live 输出体验的同时，额外保留一个可供外部 coding agent 阅读的 plain-text 最新运行日志。

## 问题

- 现在的 Rich 输出对操作员很友好
- 但失败后缺少可回看的 raw text log
- 关键上下文散落在终端状态和 traceback 中

## 目标

- 每次运行只保留一个最新日志文件
- 同时保存 Python logging records 与 uncaught traceback
- 不改变现有 Rich UX

## 推荐方案

最合适的方式是组合：

- root logger file handler
- `sys.excepthook`

## 设计摘要

- 新增 `civStation/utils/run_log_cache.py`
- 用 deterministic 的方式解析 latest-run temp cache path
- 每次运行开始时覆盖文件，结束时清理

这一页是设计摘要；原文包含更详细的方案比较。
