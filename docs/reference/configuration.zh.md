# 配置

运行时会从 `config.yaml` 读取默认值，而 CLI 参数会覆盖这些默认值。

## 项目配置示例

仓库当前的 config 大致如下：

```yaml
provider: gemini
model: gemini-3-flash-preview
router-provider: gemini
router-model: gemini-3-flash-preview
planner-provider: gemini
planner-model: gemini-3-flash-preview
turns: 100
range: 10000
prompt-language: eng
debug: "all"
router-img-max-long-edge: 1024
strategy: "과학 승리에 집중하고 정찰을 강화해."
hitl: true
autonomous: true
hitl-mode: async
chatapp: original
control-api: true
status-ui: true
```

要看精确的当前默认值，请直接查看仓库根目录里的真实文件。

## 优先级

```text
CLI flags > config.yaml > parser defaults
```

也就是说，`config.yaml` 负责稳定基线，而实验性的差异由 CLI 覆盖。

## 主要部分

### provider 设置

- `provider`
- `model`
- `router-provider`
- `router-model`
- `planner-provider`
- `planner-model`
- `turn-detector-provider`
- `turn-detector-model`

### 执行设置

- `turns`
- `range`
- `delay-action`
- `delay-turn`
- `prompt-language`
- `debug`

### strategy 与 HITL

- `strategy`
- `hitl`
- `autonomous`
- `hitl-mode`

### chat app integration

- `chatapp`
- `discord-token`
- `discord-channel`
- `discord-user`
- `whatsapp-token`
- `whatsapp-phone-number-id`
- `whatsapp-user`
- `enable-discussion`

### knowledge retrieval

- `knowledge-index`
- `enable-web-search`

### status UI 与 control

- `control-api`
- `status-ui`
- `status-port`
- `wait-for-start`

### image pipeline overrides

按 site 划分：

- `router-img-*`
- `planner-img-*`
- `context-img-*`
- `turn-detector-img-*`

### relay

- `relay-url`
- `relay-token`

## 最佳实践

- 仓库级 `config.yaml` 保持保守
- 实验差异尽量通过 CLI overrides 表达
- image pipeline 的调优一次只改一个维度
- 分享 run setup 时明确说明非默认 provider split
