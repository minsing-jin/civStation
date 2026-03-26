# Project Layout

This page is the fast map of where the important things live.

## Top-Level Folders

```text
civStation/
docs/
paper/
scripts/
tests/
.agents/
.codex/
```

## Runtime Code

```text
civStation/
  agent/
    turn_runner.py
    turn_executor.py
    models/
    modules/
      context/
      strategy/
      router/
      primitive/
      hitl/
      knowledge/
      memory/
  mcp/
  utils/
  evaluation/
```

### `agent/`

The live runtime entry points and orchestration logic.

### `mcp/`

The layered MCP facade, session model, runtime config, serialization helpers, and tool registration.

### `utils/`

Shared infrastructure such as:

- provider implementations
- image preprocessing
- screen capture and execution
- logging and run-log cache
- prompt helpers
- chat app integration

### `evaluation/`

Action evaluation datasets, runners, scoring logic, and related metrics.

## Documentation and Design Notes

### `docs/`

Human-facing documentation, theme overrides, and docs build config. The docs stack now lives under:

- `docs/mkdocs.yml`
- `docs/assets/`
- `docs/overrides/`
- `docs/plans/`

### `paper/`

Paper draft sources, bibliography, and validation artifacts used for the research/documentation side of the project.

## Tests

```text
tests/
  agent/
  evaluator/
  utils/
  mcp/
  rough_test/
```

- `agent/` covers the turn loop and module behavior
- `evaluator/` covers bbox and Civ6 evaluation flows
- `utils/` covers lower-level helpers
- `mcp/` covers the layered server
- `rough_test/` stores exploratory or heavier test material

## Skill Folders

```text
.agents/skills/
.codex/skills/
```

These are the skill roots for project-specific and shared agent workflows.

## First Files to Open

If you are new to the repo, open these in order:

1. `README.md`
2. `civStation/agent/turn_runner.py`
3. `civStation/agent/turn_executor.py`
4. `civStation/mcp/server.py`
5. `civStation/agent/modules/*/README.md`
