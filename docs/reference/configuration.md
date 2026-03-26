# Configuration

The runtime reads defaults from `config.yaml` and lets CLI flags override them.

## Example Project Config

The repository currently ships a config that looks like this in spirit:

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

See the real file at the repository root for the exact current defaults.

## Precedence Rules

```text
CLI flags > config.yaml > parser defaults
```

This makes `config.yaml` a stable baseline and the CLI the place for experiment-specific overrides.

## Key Sections

### Provider settings

Use these to choose or split models:

- `provider`
- `model`
- `router-provider`
- `router-model`
- `planner-provider`
- `planner-model`
- `turn-detector-provider`
- `turn-detector-model`

### Execution settings

- `turns`
- `range`
- `delay-action`
- `delay-turn`
- `prompt-language`
- `debug`

### Strategy and HITL

- `strategy`
- `hitl`
- `autonomous`
- `hitl-mode`

### Chat app integration

- `chatapp`
- `discord-token`
- `discord-channel`
- `discord-user`
- `whatsapp-token`
- `whatsapp-phone-number-id`
- `whatsapp-user`
- `enable-discussion`

### Knowledge retrieval

- `knowledge-index`
- `enable-web-search`

### Status UI and control

- `control-api`
- `status-ui`
- `status-port`
- `wait-for-start`

### Image pipeline overrides

Per site:

- `router-img-*`
- `planner-img-*`
- `context-img-*`
- `turn-detector-img-*`

### Relay

- `relay-url`
- `relay-token`

## Best Practice

- keep the repo-level `config.yaml` conservative
- use CLI overrides for experiments
- isolate image pipeline tweaks so you can compare runs cleanly
- document any non-default provider split when sharing a run setup
