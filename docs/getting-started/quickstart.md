# Quickstart

This is the shortest path from clone to a live, controllable run.

## 1. Install

```bash
make install
```

## 2. Set API Keys

```env
ANTHROPIC_API_KEY=...
GENAI_API_KEY=...
OPENAI_API_KEY=...
```

## 3. Start the Agent With the Status UI

```bash
python -m civStation.agent.turn_runner \
  --provider claude \
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

## 4. Open the Dashboard

```text
http://127.0.0.1:8765
```

From there you can start, pause, resume, stop, and send directives.

## 5. Optional: Run the Layered MCP Server

In a second terminal:

```bash
python -m civStation.mcp.server
```

Use this when you want external tools or skills to drive the architecture through MCP instead of importing Python internals directly.

## 6. Optional: Use the Default `config.yaml`

The repository already ships a project-level config file. You can run the agent with those defaults and override only what matters:

```bash
python -m civStation.agent.turn_runner \
  --config config.yaml \
  --provider gemini \
  --status-ui
```

Command-line flags override `config.yaml`.

## Where to Go Next

- Read [First Live Run](first-live-run.md) if you want the operator workflow.
- Read [Running the Agent](../guides/running-the-agent.md) if you want more launch patterns.
- Read [Layered MCP](../concepts/layered-mcp.md) if you plan to integrate this into another agent system.
