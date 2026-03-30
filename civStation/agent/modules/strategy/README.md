# 🧠 Strategy Layer

## 📚 Index

- [🎯 Responsibility](#-responsibility)
- [📁 Main Files](#-main-files)
- [🔄 How It Fits The Stack](#-how-it-fits-the-stack)
- [🕹️ HitL Connection](#-hitl-connection)
- [🔌 MCP Mapping](#-mcp-mapping)

The `Strategy` layer answers:

```text
Given the current state and human intent, what should the agent optimize for next?
```

## 🎯 Responsibility

This layer converts free-form guidance into a structured plan that downstream primitives can follow.

A strategy is not just plain text. It becomes a `StructuredStrategy` with:

- `text`
- `victory_goal`
- `current_phase`
- `primitive_directives`
- optional `primitive_hint`

## 📁 Main Files

- `strategy_planner.py`
  Refines human input or generates strategy autonomously
- `strategy_updater.py`
  Background strategy refresh worker
- `strategy_schemas.py`
  `StructuredStrategy`, `VictoryType`, parsing helpers
- `prompts/strategy_prompts.py`
  Prompt templates for refine/update/autonomous flows

## 🔄 How It Fits The Stack

```text
Context
  + Human input
  -> StrategyPlanner
  -> StructuredStrategy
  -> Action layer reads it during planning
```

## 🕹️ HitL Connection

This is the main place where high-level human guidance lands.

Examples:

- "Focus on science victory"
- "Avoid war for the next 10 turns"
- "Prioritize Campus over military"

Those instructions are refined into a structured strategy instead of being used as raw text forever.

## 🔌 MCP Mapping

- `strategy_get`
- `strategy_set`
- `strategy_refine`

See also:

- [MCP README](../../../mcp/README.md)
- [HitL README](../hitl/README.md)
