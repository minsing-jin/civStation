# 为什么是 CivStation

CivStation 是为那些不满足于“把 bot 跑起来然后祈祷”的人做的。

## 核心想法

Civilization VI 不是单次点击环境。它是一个长时程、带持久状态的策略游戏，某个局部上看似正确的动作，几回合之后可能仍然是全局错误。

因此，CivStation 同时是：

- 一个让 VLM 实际游玩 Civ6 的实用运行时
- 一个用于 long-horizon visual agents 的可控研究与工程基座

## README 真正在强调什么

README 的哲学非常明确：

- 系统应当可检查，而不是黑盒
- 人类 strategy 必须留在回路中
- 运行时职责要按责任拆开
- 同一架构必须能通过 MCP 对外开放
- 评测必须存在于实时游戏之外

这些不是实现细节，而是产品选择。

## 为什么项目要这样组织

- `Layered by design`：context、strategy、action、HitL 的失败方式不同
- `Human-steerable`：strategy drift 必须能在运行中修正
- `MCP-first`：外部工具不该依赖不稳定内部实现
- `Runtime-separated`：后台推理不该阻塞动作执行
- `Operator-friendly`：dashboard、WebSocket、mobile controller 都是一等表面

## 一句话总结

CivStation 不是“一个会点 Civ6 的 VLM”。

它是一套让 VLM 游玩 Civilization VI，同时让整个 run 在多个回合中保持可观察、可干预、可对齐的系统。
