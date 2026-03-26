# Providers and Image Pipeline

Provider choice and image preprocessing are two of the biggest quality and cost levers in the project.

## Supported Provider Names

From `civStation.utils.llm_provider.create_provider()`:

| Provider flag | Meaning | Default model |
| --- | --- | --- |
| `claude` | Anthropic VLM provider | `claude-4-5-sonnet-20241022` |
| `gemini` | Google GenAI VLM provider | `gemini-3-flash-preview` |
| `gpt` | OpenAI VLM provider | `gpt-4o` |
| `openai` | alias of `gpt` | `gpt-4o` |
| `openai-computer` | OpenAI computer-use style provider | `gpt-5.4` |
| `anthropic-computer` | Claude computer-use style provider | inherits Claude default |
| `mock` | deterministic fake provider for testing | `mock-vlm` |

## Practical Selection Guide

| Need | Suggested setup |
| --- | --- |
| cheapest end-to-end experimentation | `gemini` or `mock` |
| stronger planner quality | `claude` for planning |
| direct OpenAI vision planning | `gpt` |
| tool-native computer-use experiments | `openai-computer` or `anthropic-computer` |
| tests with no API calls | `mock` |

## Split Providers by Role

The CLI exposes independent provider slots for:

- router
- planner
- turn detector

This is important because they do different work. A fast cheap router can still pair well with a stronger planner.

Example:

```bash
python -m civStation.agent.turn_runner \
  --router-provider gemini \
  --planner-provider claude \
  --turn-detector-provider gemini
```

## Image Pipeline Sites

The runtime can preprocess images differently for each call site:

- `router`
- `planner`
- `context`
- `turn-detector`

Each site has its own `--{site}-img-*` flags.

## Built-In Presets

From `civStation/utils/image_pipeline.py`:

- `router_default`
- `planner_default`
- `context_default`
- `turn_detector_default`
- `planner_high_quality`
- `observation_fast`
- `policy_tab_check_fast`
- `city_production_followup_fast`
- `city_production_placement_fast`

## Main Image Controls

| Flag suffix | Meaning |
| --- | --- |
| `img-preset` | named preset |
| `img-max-long-edge` | resize limit |
| `img-ui-filter` | UI filtering mode |
| `img-color` | color policy |
| `img-encode` | transport encoding simulation |
| `img-jpeg-quality` | JPEG quality override |

Example:

```bash
python -m civStation.agent.turn_runner \
  --provider gemini \
  --router-img-preset router_default \
  --planner-img-preset planner_high_quality \
  --context-img-max-long-edge 1280
```

## Why This Matters

- routing often benefits from aggressive simplification
- planning usually needs more detail
- turn detection may need different tradeoffs than the main planner
- cost and latency change materially with resize and encoding choices

Treat image preprocessing as a first-class tuning surface, not as invisible plumbing.
