# Computer-use 提供商实现计划

这份实现计划描述如何把 OpenAI / Anthropic computer-use provider 接入现有 provider factory 与 planner `analyze()` 路径。

## 目标

- 让 `openai-computer` 与 `anthropic-computer` 可以从 provider factory 中选择
- 不破坏当前 router 与 multi-action flow
- 仅在 `analyze()` 中使用 computer-use API，其余路径保持原样

## 架构

- 增加 shared action-mapping helper
- 实现 `OpenAIComputerVLMProvider` 与 `AnthropicComputerVLMProvider`
- 继承现有 GPT / Claude provider
- 保持 `_send_to_api()` 与 `analyze_multi()` 走既有 JSON path

## 实现重点

- 增加 action translation tests
- 增加 coordinate normalization tests
- 验证 `create_provider()` 注册成功

这页是实现摘要；原文包含逐阶段的测试与实现计划。
