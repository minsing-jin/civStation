# 执行循环

实时循环的中心是 `turn_executor.py`。

## 高层流程

```text
turn_runner.py
  -> 设置 providers、HITL、status UI、knowledge、logging
  -> run_multi_turn()
      -> run_one_turn()
          -> capture screen
          -> update / read context
          -> route primitive
          -> plan action
          -> execute action
          -> record outcome
          -> 处理 checkpoints 与 directives
```

## 具体步骤

### 1. Observation

运行时先抓取屏幕，需要时再按调用位置执行图像预处理。

### 2. Routing

router provider 会把截图分类成 `policy_primitive`、`city_production_primitive` 等 primitive。

### 3. Planning

planner provider 为该 primitive 生成下一步动作。它可能是一次点击，也可能是一段 multi-step process。

### 4. Execution

动作先以 normalized coordinate space 表示，真正执行前再转换为实际屏幕坐标。

### 5. Recording 与 interrupt 检查

系统会记录结果、更新状态，并检查队列中的 directives 是否要求 pause、stop 或 override。

## 为什么 normalized coordinates 很重要

模型不需要直接知道真实屏幕分辨率。

它只需要工作在由 `--range` 定义的共享坐标空间中，运行时再在最后一步转成真实坐标。这使系统更容易跨显示器复用。

## 后台辅助组件

除了可见的 route-plan-execute 链路之外，还有这些组件一起工作：

- `ContextUpdater`
- `TurnDetector`
- `StrategyUpdater`
- `InterruptMonitor`
- `TurnCheckpoint`

这让循环保持响应性，而不必把所有事情都塞进一次同步模型调用里。

## 调试起点

当实时循环行为异常时，先看这个日志：

```text
.tmp/civStation/turn_runner_latest.log
```

这就是项目的 latest run cache 路径。
