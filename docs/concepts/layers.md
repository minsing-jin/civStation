# Layers

The high-level architecture is simple once you map the abstractions to folders.

## Layer Map

| Layer | Core question | Main code | Primary outputs |
| --- | --- | --- | --- |
| `Context` | What is on the screen and what does the agent currently know? | `civStation/agent/modules/context/` | situation summaries, turn data, local state |
| `Strategy` | Given current state and human intent, what should matter next? | `civStation/agent/modules/strategy/` | `StructuredStrategy` |
| `Action` | Which primitive should handle this screen, and what action should it take? | `civStation/agent/modules/router/`, `civStation/agent/modules/primitive/` | routed primitive + normalized action |
| `HitL` | How can a human supervise, interrupt, or redirect the run? | `civStation/agent/modules/hitl/` | lifecycle control, directives, dashboard state |

## Context

The context layer is the shared memory surface for the rest of the system.

It combines:

- `GlobalContext`
- `HighLevelContext`
- `PrimitiveContext`

Key files:

- `context_manager.py`
- `context_updater.py`
- `turn_detector.py`
- `macro_turn_manager.py`

## Strategy

The strategy layer takes free-form guidance and turns it into structured intent.

The main artifact is `StructuredStrategy`, which includes:

- `text`
- `victory_goal`
- `current_phase`
- `primitive_directives`
- optional `primitive_hint`

Key files:

- `strategy_planner.py`
- `strategy_updater.py`
- `strategy_schemas.py`
- `prompts/strategy_prompts.py`

## Action

The action layer is deliberately split in two.

### Router

The router selects the primitive for the current screen.

Key files:

- `primitive_registry.py`
- `router.py`
- `base_router.py`

### Primitive

The primitive layer plans the actual executable action or action sequence.

Key files:

- `multi_step_process.py`
- `runtime_hooks.py`
- `base_primitive.py`
- `primitives.py`

## HitL

The human-in-the-loop layer is where runtime control lives.

Key files:

- `agent_gate.py`
- `command_queue.py`
- `turn_checkpoint.py`
- `status_ui/server.py`
- `relay/relay_client.py`

## Folder Mapping

This mapping is literal, not metaphorical:

- `Context` lives in `civStation/agent/modules/context/`
- `Strategy` lives in `civStation/agent/modules/strategy/`
- `HitL` lives in `civStation/agent/modules/hitl/`
- `Action` spans `router/` and `primitive/`

That split is intentional because classification and action generation are different problems.
