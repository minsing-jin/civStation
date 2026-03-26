# Running the Agent

These are the launch patterns that matter most in practice.

## Minimal Local Run

```bash
python -m civStation.agent.turn_runner \
  --provider gemini \
  --turns 50 \
  --strategy "Focus on science victory" \
  --status-ui
```

Use this when you want one provider for everything and a local dashboard.

## Split Router and Planner

```bash
python -m civStation.agent.turn_runner \
  --router-provider gemini \
  --planner-provider claude \
  --turns 100 \
  --strategy "Prioritize Campus and early scouting." \
  --status-ui
```

Use this when you want lower routing cost and better planning quality.

## Wait for External Start

```bash
python -m civStation.agent.turn_runner \
  --provider claude \
  --status-ui \
  --wait-for-start
```

This is the safest default for live experimentation.

## Run From `config.yaml`

```bash
python -m civStation.agent.turn_runner --config config.yaml
```

The repository config already includes defaults for:

- provider and model
- turn count
- strategy
- HITL flags
- status UI
- image pipeline overrides

## Remote Relay Mode

```bash
python -m civStation.agent.turn_runner \
  --provider gemini \
  --relay-url ws://127.0.0.1:8787/ws \
  --relay-token "$RELAY_TOKEN"
```

Use this when control should come from an external relay instead of only the local dashboard.

## Common Runtime Flags

| Need | Flags |
| --- | --- |
| turn count | `--turns` |
| strategy | `--strategy` |
| local dashboard | `--status-ui --status-port` |
| safe boot | `--wait-for-start` |
| split models | `--router-provider`, `--planner-provider`, `--turn-detector-provider` |
| prompt language | `--prompt-language eng|kor` |
| debug visibility | `--debug context,turns` or `--debug all` |

## Good Operational Defaults

- keep `--status-ui` on during development
- keep `--wait-for-start` on until you trust the setup
- start with one clear strategy sentence
- change one subsystem at a time when tuning providers or image settings

Use [CLI Reference](../reference/cli.md) for the full flag groups.
