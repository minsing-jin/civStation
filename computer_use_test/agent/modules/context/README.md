# Context Layer

The `Context` layer answers one question:

```text
What does the agent currently know about the game?
```

## Responsibility

This layer maintains the shared state that the rest of the system reads from:

- `GlobalContext`: turn number, era, economy, war state, unit summaries
- `HighLevelContext`: strategic notes, threats, opportunities, current strategy snapshot
- `PrimitiveContext`: short-term action history and local execution state

## Main Files

- `context_manager.py`
  Singleton state hub used by the agent loop
- `context_updater.py`
  Background screenshot analyzer that updates situation summaries, threats, and opportunities
- `turn_detector.py`
  Dedicated background turn-number detector
- `macro_turn_manager.py`
  Detects higher-level turn boundaries and summaries
- `global_context.py`
  Broad game-state facts
- `high_level_context.py`
  Strategy-facing context
- `primitive_context.py`
  Execution-local context and action history

## How It Fits The Stack

```text
Screenshot
  -> ContextUpdater / TurnDetector
  -> ContextManager
  -> Strategy and Action read from that shared state
```

## MCP Mapping

The layered MCP server exposes this layer as:

- `context_get`
- `context_update`
- `context_observe`

See also:

- [MCP README](../../../mcp/README.md)
- [Layered MCP doc](../../../../docs/layered_mcp.md)
