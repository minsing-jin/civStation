# 统一运行日志捕获设计

这份文档说明如何把 `turn_runner` 的 `logging`、Rich、`print` 与 uncaught traceback 一并写入同一个最新运行日志文件。

## 目标

- 把终端可见的重要输出统一写入 latest-run 文件
- 新运行开始时覆盖旧内容
- 保持终端上的现有输出体验不变

## 核心设计

- `RunLogSession` 持有单个 latest-run 文件
- 保留 root `FileHandler`
- 用 tee stream wrapper 包裹 `sys.stdout` 与 `sys.stderr`，同时写入原始流和文件
- 会话结束时恢复 stdout、stderr、excepthook 与 logger handler

## 为什么需要它

原有 raw-log cache 对 logging 和 uncaught exception 足够好，但不能稳定保留 Rich 输出和 `print()`。

这一页是设计摘要；原文有更完整的 current state 与 restore 语义说明。
