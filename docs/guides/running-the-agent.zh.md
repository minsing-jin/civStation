# 运行代理

这里整理的是最常用、最实用的启动模式。

## 最小本地运行

```bash
python -m civStation.agent.turn_runner \
  --provider gemini \
  --turns 50 \
  --strategy "Focus on science victory" \
  --status-ui
```

适合用单一 provider 加本地 dashboard 起步。

## 拆分 Router 与 Planner

```bash
python -m civStation.agent.turn_runner \
  --router-provider gemini \
  --planner-provider claude \
  --turns 100 \
  --strategy "Prioritize Campus and early scouting." \
  --status-ui
```

适合希望降低 routing 成本，同时保留 planning 质量的场景。

## 等待外部 start 信号

```bash
python -m civStation.agent.turn_runner \
  --provider claude \
  --status-ui \
  --wait-for-start
```

这是实时实验时最安全的默认值。

## 基于 `config.yaml` 运行

```bash
python -m civStation.agent.turn_runner --config config.yaml
```

仓库中的默认 config 已经包含：

- provider 与 model
- turns
- strategy
- HITL flags
- status UI
- image pipeline overrides

## Remote relay 模式

```bash
python -m civStation.agent.turn_runner \
  --provider gemini \
  --relay-url ws://127.0.0.1:8787/ws \
  --relay-token "$RELAY_TOKEN"
```

当控制来源应该是外部 relay，而不仅是本地 dashboard 时使用。

## 常用运行参数

| 需求 | 参数 |
| --- | --- |
| 回合数 | `--turns` |
| strategy | `--strategy` |
| 本地 dashboard | `--status-ui --status-port` |
| 安全启动 | `--wait-for-start` |
| 模型拆分 | `--router-provider`, `--planner-provider`, `--turn-detector-provider` |
| prompt 语言 | `--prompt-language eng|kor` |
| debug 可见性 | `--debug context,turns` 或 `--debug all` |

## 实用默认建议

- 开发时保持 `--status-ui`
- 在你信任环境前保持 `--wait-for-start`
- strategy 先从一句清晰的话开始
- provider 与 image 设置一次只改一个维度

完整参数请看 [CLI 参考](../reference/cli.md)。
