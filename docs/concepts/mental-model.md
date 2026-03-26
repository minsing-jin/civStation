# Architecture Overview

CivStation is a VLM gameplay architecture for Civilization VI. It is built to let vision-language models act on the Civ6 UI while a human keeps refining the strategic intent through HitL so the run stays aligned over long horizons.

## The Problem It Solves

Most VLM gameplay demos stop at this pattern:

```text
screenshot -> model -> click
```

That is fine for one-off demos. It becomes hard to debug, steer, or extend once you want:

- persistent state
- long-horizon strategy beyond the next click
- human strategy refinement while the run is live
- model specialization across routing, planning, and observation
- external orchestration through MCP

## The CivStation Answer

CivStation turns VLM gameplay into an architecture:

```text
screen
  -> context
  -> strategy
  -> action routing
  -> action planning
  -> execution
  -> human intervention and strategy refinement when needed
```

Each part has a clear responsibility and a folder in the codebase.

## Two Ways to Think About the Project

### As a VLM gameplay runtime

You run `turn_runner.py`, point it at a Civ6 game screen, and let the stack route the current UI state, plan an action, execute it locally, and keep moving turn by turn.

### As a human-steered long-horizon system

A human can keep upgrading the strategy through HitL while the run is active. That is how the system avoids drifting away from the operator's intent over long horizons.

### As a layered platform

You can also use the MCP server, sessions, resources, and prompts as a stable external contract for skills, controllers, or higher-level orchestration.

## Why the Architecture Is Split

The split is not cosmetic. Each layer fails in a different way.

- `Context` answers what the agent knows.
- `Strategy` answers what should matter next over multiple turns.
- `Action` answers what should be done on the current screen.
- `HitL` answers how a person can refine, interrupt, or redirect the run.

This is the main architectural difference between CivStation and a monolithic prompt-driven agent.

## Where the Core Loop Lives

- CLI entry point: `civStation/agent/turn_runner.py`
- pure execution loop: `civStation/agent/turn_executor.py`
- MCP facade: `civStation/mcp/server.py`

If you only remember one idea, remember this: CivStation is built so VLMs can play Civ6 while humans keep the long-horizon strategy aligned through structured intervention.
