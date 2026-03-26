# CLI Reference

The main runtime CLI is:

```bash
python -m civStation.agent.turn_runner --help
```

The parser is built with ConfigArgParse, which means flags can come from both CLI arguments and `config.yaml`.

## Provider Configuration

| Flag group | Purpose |
| --- | --- |
| `--provider`, `--model` | default provider/model |
| `--router-provider`, `--router-model` | override the router only |
| `--planner-provider`, `--planner-model` | override the planner only |
| `--turn-detector-provider`, `--turn-detector-model` | override turn detection only |

Supported provider names in the current help output:

```text
claude, gemini, gpt, openai, openai-computer, anthropic-computer, mock
```

## Execution Parameters

| Flag | Meaning |
| --- | --- |
| `--turns` | number of turns |
| `--range` | normalized coordinate range |
| `--delay-action` | delay between actions |
| `--delay-turn` | delay between turns |
| `--prompt-language` | primitive prompt language |
| `--debug` | debug feature list such as `context`, `turns`, or `all` |

## Strategy and HITL

| Flag | Meaning |
| --- | --- |
| `--strategy` | high-level strategy text |
| `--hitl` | enable human-in-the-loop mode |
| `--autonomous` | enable autonomous mode |
| `--hitl-mode` | interrupt mode, currently `async` |

## Chat App Integration

| Flag family | Meaning |
| --- | --- |
| `--chatapp` | `original`, `discord`, or `whatsapp` |
| `--discord-*` | Discord token, channel, and user controls |
| `--whatsapp-*` | WhatsApp token and user controls |
| `--enable-discussion` | enable strategy discussion engine |

## Knowledge Retrieval

| Flag | Meaning |
| --- | --- |
| `--knowledge-index` | path to a local Civopedia index |
| `--enable-web-search` | enable Tavily-backed web search if available |

## Control API and Status UI

| Flag | Meaning |
| --- | --- |
| `--status-ui` | enable the dashboard and control API |
| `--control-api` | enable lifecycle API without the full dashboard |
| `--status-port` | bind port for the UI/API |
| `--wait-for-start` | wait for an external start signal |

## Image Pipeline Flags

The runtime exposes the same set of image flags for each site:

- `router`
- `planner`
- `context`
- `turn-detector`

Per site:

```text
--{site}-img-preset
--{site}-img-max-long-edge
--{site}-img-ui-filter
--{site}-img-color
--{site}-img-encode
--{site}-img-jpeg-quality
```

## Relay Flags

| Flag | Meaning |
| --- | --- |
| `--relay-url` | WebSocket URL of the relay server |
| `--relay-token` | relay auth token |

## Typical Commands

Quick local run:

```bash
python -m civStation.agent.turn_runner \
  --provider gemini \
  --turns 50 \
  --status-ui
```

High-control run:

```bash
python -m civStation.agent.turn_runner \
  --router-provider gemini \
  --planner-provider claude \
  --status-ui \
  --wait-for-start
```
