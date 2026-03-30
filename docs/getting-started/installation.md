# Installation

Start with the concrete path that gets you a runnable local environment.

## Requirements

- Python `3.10+`
- A local Civilization VI environment the agent can observe and control
- API keys for any model provider you plan to use
- Screen capture and input permissions for your operating system

## Recommended Runtime Environment

For live gameplay, use the host machine's local `uv` / `.venv` environment.

Do not treat Docker as the default runtime for real Civ6 sessions.

Why:

- CivStation captures the host desktop and Civ6 window directly.
- CivStation sends mouse and keyboard actions back to the real local desktop session.
- macOS permissions such as `Screen Recording` and `Accessibility` must be granted on the host.

Docker is reasonable for non-GUI tasks such as docs builds, linting, and tests that do not need the real game window. It is not the recommended runtime for playing Civ6 through CivStation.

If you plan to use voice or run the full dashboard stack on Linux, system packages such as PortAudio may be required. The CI job installs `portaudio19-dev` and `gcc` on Ubuntu for that reason.

## Core Install

For normal project work and live runs:

```bash
git clone https://github.com/minsing-jin/civStation.git
cd civStation
uv sync
```

`uv sync` creates or updates the repo-local `.venv`.
`uv run ...` works without activation, but if you want to enter the environment directly:

```bash
source .venv/bin/activate
```

If you want the editable development workflow with test dependencies and `pre-commit`, you can still use:

```bash
make install
```

## Install With Docs Support

If you want to serve or build this documentation site locally:

```bash
uv pip install -e ".[docs,test]"
```

Or use the built-in shortcuts later:

```bash
make docs-serve
make docs-build
```

## Environment Variables

Set only the providers you actually use.

```env
ANTHROPIC_API_KEY=...
GENAI_API_KEY=...
OPENAI_API_KEY=...
RELAY_TOKEN=...
```

Provider aliases are documented in [Providers and Image Pipeline](../guides/providers-and-image-pipeline.md).

## Verify the Install

Check the CLI:

```bash
uv run civstation
uv run civstation run --help
```

Check the MCP server entry point:

```bash
uv run civstation mcp --help
```

The MCP server runs on stdio by default, so it will wait for a client instead of opening a browser page.

## Next Step

Run the minimal loop in [Quickstart](quickstart.md).
