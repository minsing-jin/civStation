# Extending the Stack

The project is designed to be extended layer by layer.

## Add a Primitive

Start here:

- `civStation/agent/modules/router/primitive_registry.py`
- `civStation/agent/modules/primitive/`

Typical flow:

1. define the primitive in the registry
2. add or update prompt logic
3. handle planning or multi-step behavior
4. add tests
5. document the new behavior

## Add or Swap an MCP Adapter

Start here:

- `civStation/mcp/runtime.py`
- `civStation/mcp/server.py`

Use adapter overrides when you want a different router, planner, context observer, strategy refiner, or executor without changing the public MCP surface.

## Extend HITL

Start here:

- `civStation/agent/modules/hitl/command_queue.py`
- `civStation/agent/modules/hitl/status_ui/server.py`
- `civStation/agent/modules/hitl/relay/relay_client.py`

Be careful with directive priority and lifecycle semantics. Those are operator safety issues, not just convenience behavior.

## Extend Providers or Image Handling

Start here:

- `civStation/utils/llm_provider/`
- `civStation/utils/image_pipeline.py`

This is the right place for model-specific behavior, preprocessing presets, and transport-specific image tuning.

## Extend Skills

Project-facing skills live under:

- `.codex/skills/`
- `.agents/skills/`

The recommended pattern is:

1. keep the skill thin
2. use MCP as the stable control plane
3. keep reusable workflow instructions in `SKILL.md`
4. keep helper scripts next to the skill when needed

## Extend Evaluation

Start here:

- `civStation/evaluation/evaluator/action_eval/bbox_eval/`
- `civStation/evaluation/evaluator/action_eval/civ6_eval/`

Use `bbox_eval` when you want a more reusable and less game-specific action evaluator.
