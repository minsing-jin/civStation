# 架构指南

这一页把根目录级别的架构说明收进 docs 门户中。

## 高层数据流

```text
turn_runner.py
  -> provider setup
  -> HITL setup
  -> logging and run sessions
  -> run_multi_turn()

run_multi_turn()
  -> run_one_turn()
      -> capture screenshot
      -> route primitive
      -> plan action
      -> execute action
      -> update context and checkpoints
```

## 核心运行时文件

| 文件 | 角色 |
| --- | --- |
| `civStation/agent/turn_runner.py` | CLI 与 runtime wiring |
| `civStation/agent/turn_executor.py` | observe、route、plan、execute loop |
| `civStation/mcp/server.py` | layered MCP facade |
| `civStation/utils/image_pipeline.py` | per-site 图像预处理 |
| `civStation/utils/llm_provider/` | provider implementations |

## 为什么要这样拆分

- routing 与 planning 是不同问题
- context 必须比单次点击活得更久
- strategy 应该可以在不重写整段 prompt 的情况下更新
- human control 必须能安全中断循环
- MCP 需要在不依赖内部 imports 的情况下暴露架构

## 目录视图

```text
civStation/
  agent/
    modules/
      context/
      strategy/
      router/
      primitive/
      hitl/
      knowledge/
      memory/
  evaluation/
  mcp/
  utils/
```

## 建议搭配阅读

- [心智模型](../concepts/mental-model.md)
- [分层](../concepts/layers.md)
- [执行循环](../concepts/execution-loop.md)
