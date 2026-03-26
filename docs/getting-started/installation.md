# Installation

Start with the concrete path that gets you a runnable local environment.

## Requirements

- Python `3.10+`
- A local Civilization VI environment the agent can observe and control
- API keys for any model provider you plan to use
- Screen capture and input permissions for your operating system

If you plan to use voice or run the full dashboard stack on Linux, system packages such as PortAudio may be required. The CI job installs `portaudio19-dev` and `gcc` on Ubuntu for that reason.

## Core Install

For normal project work:

```bash
make install
```

That installs the project in editable mode with test dependencies and sets up `pre-commit`.

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
python -m civStation.agent.turn_runner --help
```

Check the MCP server entry point:

```bash
python -m civStation.mcp.server
```

The MCP server runs on stdio by default, so it will wait for a client instead of opening a browser page.

## Next Step

Run the minimal loop in [Quickstart](quickstart.md).
