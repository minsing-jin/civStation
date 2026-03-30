# Quickstart

This is the shortest path from clone to a live, controllable run.

## 1. Install

```bash
git clone https://github.com/minsing-jin/civStation.git
cd civStation
uv sync
```

`uv sync` creates the repo-local `.venv`.
`uv run ...` works without activation, but if you want to enter it directly:

```bash
source .venv/bin/activate
```

## 2. Set API Keys

```env
ANTHROPIC_API_KEY=...
GENAI_API_KEY=...
OPENAI_API_KEY=...
```

## 3. Print the Operator Guide

```bash
uv run civstation
```

This prints the preflight checklist:

- keep Civ6 visible on the main monitor
- keep the actual game screen focused
- avoid covering the game with the dashboard
- prefer a phone or secondary device for control

## 4. Start the Agent With the Status UI

```bash
uv run civstation run \
  --provider gemini \
  --model gemini-3-flash \
  --turns 100 \
  --strategy "Focus on science victory" \
  --status-ui \
  --wait-for-start \
  --status-port 8765
```

What this command does:

- starts the live turn loop
- enables the built-in dashboard and control API
- waits for an explicit start signal instead of acting immediately
- keeps the strategy visible and editable while the run is live
- shows the operator checklist before the run begins

## 5. Open the Dashboard

```text
http://127.0.0.1:8765
```

From there you can start, pause, resume, stop, and send directives.

## 6. Optional: Run the Layered MCP Server

In a second terminal:

```bash
uv run civstation mcp
```

Use this when you want external tools or skills to drive the architecture through MCP instead of importing Python internals directly.

## 7. Optional: Use the Default `config.yaml`

The repository already ships a project-level config file. You can run the agent with those defaults and override only what matters:

```bash
uv run civstation run \
  --config config.yaml \
  --provider gemini \
  --status-ui
```

Command-line flags override `config.yaml`.

## Where to Go Next

- Read [First Live Run](first-live-run.md) if you want the operator workflow.
- Read [Running the Agent](../guides/running-the-agent.md) if you want more launch patterns.
- Read [Layered MCP](../concepts/layered-mcp.md) if you plan to integrate this into another agent system.
