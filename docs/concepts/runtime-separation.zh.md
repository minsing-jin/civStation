# 运行时分离

README 并没有把 CivStation 描述成“一个循环负责所有事”。它描述的是一个把不同类型工作拆到不同运行时 lane 中的系统。

## 三条运行时 lane

### Background runtime

适合在 action loop 旁边运行的工作：

- context observation
- turn tracking
- strategy refresh
- background reasoning

### Main-thread action runtime

这一条必须保持 deterministic 且 interruptible：

- route 当前屏幕
- plan primitive action
- 在 game window 上执行动作

### HitL runtime

这一条把控制放在 action thread 之外：

- dashboard
- WebSocket control
- relay 与 mobile controller
- strategy 与 lifecycle directives

## 为什么重要

如果把这些全塞进一个阻塞循环里：

- 昂贵的 reasoning 会拖住 action execution
- interruption 会变得脆弱
- strategy refinement 会来得太晚
- external control 会退化成附加功能

运行时分离让 CivStation 更像一个可操作系统，而不是一个聪明但脆弱的 demo。
