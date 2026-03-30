# 快速开始

这是从 clone 到可控运行的最短路径。

## 1. 安装

```bash
make install
```

## 2. 设置 API Key

```env
ANTHROPIC_API_KEY=...
GENAI_API_KEY=...
OPENAI_API_KEY=...
```

## 3. 启动带状态 UI 的代理

```bash
python -m civStation.agent.turn_runner \
  --provider claude \
  --turns 100 \
  --strategy "Focus on science victory" \
  --status-ui \
  --wait-for-start \
  --status-port 8765
```

这个命令会：

- 启动实时 turn loop
- 启用内置 dashboard 与 control API
- 不会立刻行动，而是等待显式的 start 信号
- 在运行过程中保留可见且可修改的 strategy

## 4. 打开仪表盘

```text
http://127.0.0.1:8765
```

你可以在这里 start、pause、resume、stop，并发送 directive。

## 5. 可选：运行 layered MCP server

在另一个终端中：

```bash
python -m civStation.mcp.server
```

当你希望外部工具或 skill 通过 MCP 控制架构，而不是直接 import 内部 Python 模块时使用它。

## 6. 可选：使用默认 `config.yaml`

仓库已经包含一个项目级配置文件。你可以直接使用它，并只覆盖真正需要修改的项：

```bash
python -m civStation.agent.turn_runner \
  --config config.yaml \
  --provider gemini \
  --status-ui
```

CLI 参数优先级高于 `config.yaml`。

## 下一步

- 查看 [第一次实机运行](first-live-run.md) 了解操作员视角
- 查看 [运行代理](../guides/running-the-agent.md) 了解更多启动模式
- 查看 [分层 MCP](../concepts/layered-mcp.md) 了解如何接入其它系统
