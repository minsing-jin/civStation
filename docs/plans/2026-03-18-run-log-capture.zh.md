# 运行日志捕获实现计划

这份实现计划描述的是：如何把 `turn_runner` 运行期间的重要输出统一写入单个最新日志文件。

## 目标

- 把 `logging`、stdout、stderr 与 traceback 写入同一个 latest-run 文件
- 每次新运行覆盖同一路径

## 架构

- `RunLogSession` 同时持有 root logging handler 与 stdout/stderr tee
- 在 close 时恢复所有 monkeypatch 与 handlers

## 实现重点

- 在 `tests/utils/test_run_log_cache.py` 中增加 print/stderr capture regression
- 在 `civStation/utils/run_log_cache.py` 中实现 tee stream wrapper
- 验证 `session.close()` 会恢复 stdout/stderr

这一页是实现摘要，原文包含从 failing tests 开始的详细执行步骤。
