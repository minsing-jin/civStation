# 第一次实机运行

这一页关注的不是“怎么启动进程”，而是“启动之后会发生什么”。

## 推荐启动命令

```bash
python -m civStation.agent.turn_runner \
  --provider gemini \
  --router-provider gemini \
  --planner-provider claude \
  --turns 100 \
  --strategy "Focus on science victory and reinforce scouting." \
  --status-ui \
  --wait-for-start \
  --status-port 8765
```

这样拆分很实用，因为 routing 通常比 planning 更便宜也更快。

## 预期流程

1. 进程启动并初始化 providers。
2. status UI 准备就绪。
3. 如果启用了 `--wait-for-start`，代理会停在 pre-start 状态。
4. 一旦收到 start，每个 turn 会经过 observation、routing、planning、execution 与 checkpoint 处理。

## 仪表盘提供什么

- 实时状态快照
- agent 生命周期控制
- strategy 或自由文本 directive
- 启用 discussion mode 时的讨论接口
- 供外部 controller 使用的 WebSocket 通道

它不是一个被动监视器，而是控制表面的一部分。

## 最快可用的命令

Lifecycle:

```bash
curl -X POST http://127.0.0.1:8765/api/agent/start
curl -X POST http://127.0.0.1:8765/api/agent/pause
curl -X POST http://127.0.0.1:8765/api/agent/resume
curl -X POST http://127.0.0.1:8765/api/agent/stop
```

Directive:

```bash
curl -X POST http://127.0.0.1:8765/api/directive \
  -H "Content-Type: application/json" \
  -d '{"text":"Switch to culture victory and stop expanding for now."}'
```

## 日志与产物

调试时第一件事是看这个文件：

```text
.tmp/civStation/turn_runner_latest.log
```

这是项目默认的 latest-run log cache 路径。运行异常、卡住或行为不对时，先看这里。

## 操作建议

- 在你完全信任环境之前，保持 `--wait-for-start` 开启。
- 开发阶段不要关闭 `--status-ui`。
- 如果想降成本又不牺牲 planner 质量，就拆分 router 与 planner provider。
- 当 UI 控制不够时，把 MCP server 一起跑起来。
