# Architecture Guide

This page brings the root-level architecture notes into the docs portal so the historical design narrative lives under `docs/`.

## High-Level Data Flow

```text
turn_runner.py
  -> provider setup
  -> HITL setup
  -> logging and run sessions
  -> run_multi_turn()

run_multi_turn()
  -> run_one_turn()
      -> capture screenshot
      -> route primitive
      -> plan action
      -> execute action
      -> update context and checkpoints
```

## Core Runtime Files

| File | Role |
| --- | --- |
| `civStation/agent/turn_runner.py` | CLI and runtime wiring |
| `civStation/agent/turn_executor.py` | observe, route, plan, execute loop |
| `civStation/mcp/server.py` | layered MCP facade |
| `civStation/utils/image_pipeline.py` | per-site image preprocessing |
| `civStation/utils/llm_provider/` | provider implementations |

## Why the Runtime Is Split This Way

- routing and planning are different problems
- context should outlive a single click
- strategy should be editable without rewriting the whole prompt
- human control should be able to interrupt the loop safely
- MCP should expose the architecture without forcing internal imports

## Directory View

```text
civStation/
  agent/
    modules/
      context/
      strategy/
      router/
      primitive/
      hitl/
      knowledge/
      memory/
  evaluation/
  mcp/
  utils/
```

## Read This With

- [Mental Model](../concepts/mental-model.md)
- [Layers](../concepts/layers.md)
- [Execution Loop](../concepts/execution-loop.md)
