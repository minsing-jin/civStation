# CLI 参考

主要运行时 CLI：

```bash
python -m civStation.agent.turn_runner --help
```

parser 基于 ConfigArgParse，因此既能读取 CLI 参数，也能读取 `config.yaml`。

## Provider 配置

| 参数组 | 用途 |
| --- | --- |
| `--provider`, `--model` | 默认 provider/model |
| `--router-provider`, `--router-model` | 只覆盖 router |
| `--planner-provider`, `--planner-model` | 只覆盖 planner |
| `--turn-detector-provider`, `--turn-detector-model` | 只覆盖 turn detection |

当前 help 输出中支持的 provider：

```text
claude, gemini, gpt, openai, openai-computer, anthropic-computer, mock
```

## 执行参数

| 参数 | 含义 |
| --- | --- |
| `--turns` | 回合数 |
| `--range` | normalized coordinate range |
| `--delay-action` | 动作间延迟 |
| `--delay-turn` | 回合间延迟 |
| `--prompt-language` | primitive prompt 语言 |
| `--debug` | 如 `context`、`turns`、`all` 等 debug 选项 |

## Strategy 与 HITL

| 参数 | 含义 |
| --- | --- |
| `--strategy` | 高层 strategy 文本 |
| `--hitl` | 启用 human-in-the-loop 模式 |
| `--autonomous` | 启用 autonomous 模式 |
| `--hitl-mode` | interrupt mode，目前为 `async` |

## Chat App Integration

| 参数族 | 含义 |
| --- | --- |
| `--chatapp` | `original`、`discord` 或 `whatsapp` |
| `--discord-*` | Discord token、channel、user 配置 |
| `--whatsapp-*` | WhatsApp token、user 配置 |
| `--enable-discussion` | 启用 strategy discussion engine |

## Knowledge Retrieval

| 参数 | 含义 |
| --- | --- |
| `--knowledge-index` | 本地 Civopedia index 路径 |
| `--enable-web-search` | 在可用时启用 Tavily web search |

## Control API 与 Status UI

| 参数 | 含义 |
| --- | --- |
| `--status-ui` | 启用 dashboard 与 control API |
| `--control-api` | 不启用完整 dashboard，只启用 lifecycle API |
| `--status-port` | UI/API 绑定端口 |
| `--wait-for-start` | 等待外部 start 信号 |

## Image Pipeline 参数

以下每个 site 都有相同形态的图像参数：

- `router`
- `planner`
- `context`
- `turn-detector`

每个 site：

```text
--{site}-img-preset
--{site}-img-max-long-edge
--{site}-img-ui-filter
--{site}-img-color
--{site}-img-encode
--{site}-img-jpeg-quality
```

## Relay 参数

| 参数 | 含义 |
| --- | --- |
| `--relay-url` | relay server 的 WebSocket URL |
| `--relay-token` | relay auth token |

## 典型命令

快速本地运行：

```bash
python -m civStation.agent.turn_runner \
  --provider gemini \
  --turns 50 \
  --status-ui
```

以控制为中心的运行：

```bash
python -m civStation.agent.turn_runner \
  --router-provider gemini \
  --planner-provider claude \
  --status-ui \
  --wait-for-start
```
