# Router Layer

The router is the first half of the `Action` layer.

It answers:

```text
Which primitive should handle the current screen?
```

## Responsibility

The router classifies the current screenshot into a primitive such as:

- `unit_ops_primitive`
- `city_production_primitive`
- `research_select_primitive`
- `policy_primitive`
- `popup_primitive`

The router does not decide the final click itself.
It decides which action specialist should take over.

## Main Files

- `primitive_registry.py`
  Single source of truth for primitive criteria, prompts, priorities, and metadata
- `router.py`
  Router implementation
- `base_router.py`
  Shared router interface

## Why This Folder Matters

If you add, remove, or redefine primitives, start here.

`primitive_registry.py` drives:

- router prompt generation
- primitive priority
- primitive prompt lookup
- HITL-only primitive registration such as forced `war_primitive`

## Flow

```text
Screenshot
  -> route_primitive()
  -> primitive name
  -> Primitive layer plans action
```

## MCP Mapping

- `action_route`
- `action_route_and_plan`
- `workflow_decide`
- `workflow_step`

See also:

- [Primitive README](../primitive/README.md)
- [MCP README](../../../mcp/README.md)
