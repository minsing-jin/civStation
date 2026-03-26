# 历史实现总结

这一页用于把旧的 implementation summary 保留在 docs 门户中。

## 那份总结当时覆盖了什么

原始总结主要针对项目较早阶段：

- static evaluator tests 的 pytest migration
- VLM provider integration
- primitive 级别的 provider support
- prompt module 整理
- 面向 evaluator 的文档

## 当时的重要里程碑

### 测试迁移

它记录了 evaluator tests 向 pytest 的迁移，包括 tolerance、parsing 与 integration coverage。

### provider integration

它还记录了以下支持的加入：

- Claude
- Gemini
- GPT
- provider factory helpers

### primitive integration

那也是多个 primitive 开始接入 optional VLM provider 与 custom prompt 的阶段。

## 为什么保留它

这份文档适合作为项目考古材料。它能帮助理解在当前 layered runtime 与 MCP surface 成形之前，evaluator 与 provider stack 是如何演进的。

## 当前的标准文档

如果你要看最新用法，请优先阅读：

- [提供商与图像管线](../guides/providers-and-image-pipeline.md)
- [评测](../guides/evaluation.md)
- [项目结构](../reference/project-layout.md)
