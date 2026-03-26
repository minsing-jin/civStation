# 参与贡献

这一页是 `CONTRIBUTING.md` 的实用版。

## setup

```bash
make install
```

如果你也要修改 docs：

```bash
uv pip install -e ".[docs,test]"
```

## 日常命令

```bash
make lint
make format
make check
make test
make coverage
make docs-build
```

或使用 `just`：

```bash
just qa
just docs-build
```

## 贡献原则

- 保持改动小且可逆
- 行为变化时同步更新 docs
- 逻辑变化时同步添加或更新 tests
- 尽量沿着现有分层扩展，而不是绕过架构

## 好的 PR 范围

- primitive 改进并附带对应 tests
- 新的 MCP adapter 或 tool 改进
- HitL surface 修复并验证 dashboard/API
- 让文档与当前代码一致，而不是写成“未来愿景”

## 在哪里提问题

- bugs/regressions：GitHub issues
- feature proposals：GitHub issues 或边界明确的 PR
- docs 缺口：通常直接提 PR 最快

## 本地 docs 工作流

```bash
make docs-serve
```

这样可以在浏览器中直接审阅文档变化。
