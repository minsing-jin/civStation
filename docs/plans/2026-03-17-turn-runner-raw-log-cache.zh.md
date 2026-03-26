# Turn Runner Raw Log Cache 实现计划

这份实现计划描述了如何给 `turn_runner` 增加一个 opt-in 的 raw run log cache。

## 目标

- 每次运行覆盖同一个最新日志文件
- 同时捕获 plain logging 与 uncaught tracebacks
- 保持原有 Rich output 不变

## 架构

- 新 utility module 持有 temp-file path、logger file handler 与 temporary `sys.excepthook` wrapper
- `turn_runner.main()` 在启动时 opt-in
- 在 cleanup 中总是 teardown

## 实现重点

- 在 `tests/utils/test_run_log_cache.py` 中增加 regression tests
- 引入 `RunLogSession` 之类的 session object
- 把 root logger 与 traceback capture 统一管理

原文包含逐任务的实现步骤，这一页是中文摘要。
