# 测试与质量

项目已经具备分层测试表面。最好先运行能够证明改动正确性的最小测试。

## 主要命令

```bash
make lint
make format
make check
make test
make coverage
make docs-build
```

使用 `just`：

```bash
just qa
just test
just coverage
```

## 测试区域

| 目录 | 覆盖内容 |
| --- | --- |
| `tests/agent/` | turn loop、runtime modules、checkpoints |
| `tests/evaluator/` | bbox 与 Civ6 evaluation logic |
| `tests/utils/` | providers、parsers、screen helpers、run-log cache |
| `tests/mcp/` | layered MCP server behavior |
| `tests/rough_test/` | exploratory/heavier tests 与 reports |

## integration marker

pytest 配置定义了 `integration` marker：

```bash
pytest -m "not integration"
```

适合做更快的本地验证。

## CI

当前主测试 workflow 会：

- 运行 Ruff checks
- 在 Python `3.12` 与 `3.13` 上测试
- 安装音频相关依赖所需的 Linux system packages

docs workflow 会在 PR 中以 strict mode 构建站点，并在推送到 `main` 或 `master` 时发布到 GitHub Pages。

## 合并前应确认

- 为改动模块运行最小且有针对性的测试
- 对重要 runtime 变更跑完整 `pytest`
- docs 或 nav 改动后执行 `make docs-build`
- 触及 HitL 或 status UI 时实际验证 dashboard 流程
