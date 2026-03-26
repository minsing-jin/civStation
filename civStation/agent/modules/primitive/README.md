# ⚙️ Primitive Layer

## 📚 Index

- [🎯 Responsibility](#-responsibility)
- [📁 Main Files](#-main-files)
- [🔄 Execution Model](#-execution-model)
- [🕹️ HitL Connection](#-hitl-connection)
- [📍 Related Runtime Entry Points](#-related-runtime-entry-points)
- [🔌 MCP Mapping](#-mcp-mapping)

The primitive system is the second half of the `Action` layer.

It answers:

```text
Given the selected primitive, what action should be executed next?
```

## 🎯 Responsibility

This layer turns a routed screen into executable UI actions.

Depending on the primitive, that can be:

- a single action
- a multi-step scripted flow
- an observation-assisted multi-step flow
- a HITL-forced override path

## 📁 Main Files

- `multi_step_process.py`
  Main multi-step execution engine
- `runtime_hooks.py`
  Runtime hooks and integration helpers
- `primitives.py`
  Older primitive classes kept for reference
- `base_primitive.py`
  Shared primitive interface

## 🔄 Execution Model

```text
route_primitive()
  -> primitive selected
  -> plan_action()
  -> normalized action(s)
  -> execute_action()
```

The action output uses normalized coordinates, which are converted to real screen coordinates before execution.

## 🕹️ HitL Connection

This layer is where lower-level intervention matters most.

Examples:

- force a specific primitive with `PRIMITIVE_OVERRIDE`
- inject a specific micro-level directive
- pause between multi-step checkpoints
- stop before final execution

The queue is checked before planning and during checkpoints in the turn executor.

## 📍 Related Runtime Entry Points

- `civStation/agent/turn_executor.py`
  observe -> route -> plan -> execute

## 🔌 MCP Mapping

- `action_plan`
- `action_execute`
- `action_route_and_plan`
- `workflow_act`
- `workflow_step`

See also:

- [Router README](../router/README.md)
- [HitL README](../hitl/README.md)
