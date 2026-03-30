# Computer-use 提供商设计

这份文档讨论如何把 OpenAI 与 Anthropic 的 computer-use provider 接入到现有 `BaseVLMProvider` 契约中。

## 目标

- 不替换当前 router/executor architecture
- 增加 `openai-computer` 与 `anthropic-computer` provider
- 在 planner-side `analyze()` 中使用 provider-specific computer-use API/tool

## 核心决定

- 不另起一个 autonomous agent loop
- 而是在 planner adapter 这一层接入 computer-use provider

## 原因

因为代码库已经拥有：

1. screenshot capture
2. primitive routing
3. action planning
4. local execution

因此最好的集成点不是新循环，而是 planner-side `analyze()`。

这页是设计摘要，原文对不同方案与最终选择有更完整说明。
