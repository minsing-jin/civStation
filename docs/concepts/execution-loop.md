# Execution Loop

The live loop is centered on `turn_executor.py`.

## High-Level Flow

```text
turn_runner.py
  -> setup providers, HITL, status UI, knowledge, logging
  -> run_multi_turn()
      -> run_one_turn()
          -> capture screen
          -> update / read context
          -> route primitive
          -> plan action
          -> execute action
          -> record outcome
          -> handle checkpoints and directives
```

## The Concrete Steps

### 1. Observation

The runtime captures the screen and optionally preprocesses the image for the relevant call site.

### 2. Routing

The router provider classifies the screenshot into a primitive such as `policy_primitive` or `city_production_primitive`.

### 3. Planning

The planner provider generates the next action for that primitive. This can be a single click or a multi-step process.

### 4. Execution

Actions are expressed in normalized coordinate space and translated into real screen coordinates before execution.

### 5. Recording and interruption checks

The executor records what happened, updates state, and checks whether queued directives require pausing, stopping, or overriding the default flow.

## Why Normalized Coordinates Matter

The models do not need to know the real screen resolution.

Instead they operate in a shared coordinate space, controlled by `--range`, and the runtime converts the output into real coordinates later. This is one of the reasons the system is easier to retarget across displays.

## Background Helpers

The turn loop is not only the visible route-plan-execute chain.

It also relies on background or adjacent helpers:

- `ContextUpdater`
- `TurnDetector`
- `StrategyUpdater`
- `InterruptMonitor`
- `TurnCheckpoint`

These keep the loop responsive without collapsing everything into one synchronous model call.

## Debugging Anchor

When the live loop does something wrong, the first log to inspect is:

```text
.tmp/civStation/turn_runner_latest.log
```

That is the latest run cache used by the project.
