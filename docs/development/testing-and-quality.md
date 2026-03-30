# Testing and Quality

The project already has a layered test surface. Use the smallest test that proves your change.

## Main Commands

```bash
make lint
make format
make check
make test
make coverage
make docs-build
```

With `just`:

```bash
just qa
just test
just coverage
```

## Test Areas

| Folder | What it covers |
| --- | --- |
| `tests/agent/` | turn loop, runtime modules, checkpoints |
| `tests/evaluator/` | bbox and Civ6 evaluation logic |
| `tests/utils/` | providers, parsers, screen helpers, run-log cache |
| `tests/mcp/` | layered MCP server behavior |
| `tests/rough_test/` | exploratory or heavier tests and reports |

## Integration Marker

The pytest config defines an `integration` marker:

```bash
pytest -m "not integration"
```

Use that when you want a faster local pass.

## CI

The main test workflow currently:

- runs Ruff checks
- tests on Python `3.12` and `3.13`
- installs Linux system packages required by audio-related dependencies

The docs workflow builds the MkDocs site in strict mode on pull requests and deploys it on pushes to `main` or `master`.

## What to Verify Before Merging

- the smallest targeted test for the changed module
- a full `pytest` run for non-trivial runtime changes
- `make docs-build` when docs or nav changed
- the live dashboard flow when touching HitL or status UI code
