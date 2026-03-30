# First Live Run

This page is about what happens after the process starts, not just how to launch it.

## Recommended Launch Command

```bash
python -m civStation.agent.turn_runner \
  --provider gemini \
  --router-provider gemini \
  --planner-provider claude \
  --turns 100 \
  --strategy "Focus on science victory and reinforce scouting." \
  --status-ui \
  --wait-for-start \
  --status-port 8765
```

That split is useful because routing is cheaper than planning and often benefits from a faster model.

## What You Should Expect

1. The process boots and loads providers.
2. The status UI becomes available.
3. The agent waits in a pre-start state if `--wait-for-start` is enabled.
4. Once started, each turn goes through observation, routing, planning, execution, and checkpoint handling.

## What the Dashboard Gives You

- live status snapshots
- agent lifecycle control
- strategy or free-form directives
- discussion endpoints when discussion mode is enabled
- a WebSocket channel for external controllers

The dashboard is not just a pretty monitor. It is part of the control surface.

## The Fastest Useful Commands

Lifecycle:

```bash
curl -X POST http://127.0.0.1:8765/api/agent/start
curl -X POST http://127.0.0.1:8765/api/agent/pause
curl -X POST http://127.0.0.1:8765/api/agent/resume
curl -X POST http://127.0.0.1:8765/api/agent/stop
```

Directive:

```bash
curl -X POST http://127.0.0.1:8765/api/directive \
  -H "Content-Type: application/json" \
  -d '{"text":"Switch to culture victory and stop expanding for now."}'
```

## Logs and Artifacts

When you debug a live run, start here first:

```text
.tmp/civStation/turn_runner_latest.log
```

That file is the default latest-run log cache path used by the project. It is the first place to inspect when a run goes wrong, stalls, or behaves unexpectedly.

## Operator Tips

- Start with `--wait-for-start` until you trust the setup.
- Keep `--status-ui` on during development. It exposes far more state than raw terminal logs.
- Split router and planner providers when you want lower cost without giving up planner quality.
- Use the MCP server when you want structured external orchestration instead of UI-only control.
