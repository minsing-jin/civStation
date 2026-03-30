# 项目结构

这是一张快速地图，用来帮助你找到重要内容所在的位置。

## 顶层目录

```text
civStation/
docs/
paper/
scripts/
tests/
.agents/
.codex/
```

## 运行时代码

```text
civStation/
  agent/
    turn_runner.py
    turn_executor.py
    models/
    modules/
      context/
      strategy/
      router/
      primitive/
      hitl/
      knowledge/
      memory/
  mcp/
  utils/
  evaluation/
```

### `agent/`

实时运行时入口与 orchestration logic。

### `mcp/`

layered MCP facade、session model、runtime config、serialization helpers 与 tool registration。

### `utils/`

共享基础设施，包括：

- provider implementations
- image preprocessing
- screen capture 与 execution
- logging 与 run-log cache
- prompt helpers
- chat app integration

### `evaluation/`

action evaluation datasets、runners、scoring logic 与 metrics。

## 文档与设计说明

### `docs/`

人类可读文档、主题覆盖和 docs build config 现在都在这里：

- `docs/mkdocs.yml`
- `docs/assets/`
- `docs/overrides/`
- `docs/plans/`

### `paper/`

论文草稿源文件、bibliography 与 validation artifacts。

## 测试

```text
tests/
  agent/
  evaluator/
  utils/
  mcp/
  rough_test/
```

- `agent/` 覆盖 turn loop 与 runtime modules
- `evaluator/` 覆盖 bbox 与 Civ6 evaluation
- `utils/` 覆盖底层 helpers
- `mcp/` 覆盖 layered server
- `rough_test/` 保存 exploratory/heavier test material

## skill 目录

```text
.agents/skills/
.codex/skills/
```

这里是项目专用与共享 agent workflows 的 skill roots。

## 新人最先打开的文件

1. `README.md`
2. `civStation/agent/turn_runner.py`
3. `civStation/agent/turn_executor.py`
4. `civStation/mcp/server.py`
5. `civStation/agent/modules/*/README.md`
