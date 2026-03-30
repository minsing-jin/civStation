# MCP Tools

The layered MCP server groups tools by responsibility.

## Session Tools

- `adapter_list`
- `session_create`
- `session_list`
- `session_get`
- `session_export`
- `session_import`
- `session_delete`
- `session_config_get`
- `session_config_update`

Use these to create isolated sessions, inspect them, and swap adapters or runtime config.

## Context Tools

- `context_get`
- `context_update`
- `context_observe`

## Strategy Tools

- `strategy_get`
- `strategy_set`
- `strategy_refine`

## Memory Tools

- `memory_get`
- `memory_start_task`
- `memory_update`
- `memory_reset`

These are especially useful for multi-step primitive flows that need task-local state.

## Action Tools

- `action_route`
- `action_plan`
- `action_execute`
- `action_route_and_plan`

## Workflow Tools

- `workflow_observe`
- `workflow_decide`
- `workflow_act`
- `workflow_step`

Use workflow tools when you want common observe-decide-act bundles instead of manual tool sequencing.

## HitL Tools

- `hitl_send`
- `hitl_status`

## Resources

- `civ6://sessions`
- `civ6://sessions/{session_id}/state`
- `civ6://sessions/{session_id}/context`
- `civ6://sessions/{session_id}/memory`

## Prompt Templates

- `strategy_only_workflow`
- `plan_only_workflow`
- `full_orchestration_workflow`
- `relay_controlled_workflow`

## Typical Sequences

### Strategy only

1. `session_create`
2. `context_get`
3. `memory_get`
4. `strategy_refine`
5. `strategy_get`

### Plan only

1. `session_create`
2. `workflow_observe`
3. `action_route`
4. `action_plan`

### Full orchestration

1. `session_create`
2. `workflow_observe`
3. `workflow_decide`
4. `workflow_act` or `workflow_step(execute=true)`
5. `hitl_status`
