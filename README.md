# CivStation

Layered Civilization VI computer-use stack with `Context`, `Strategy`, `Action`, `HitL`, and a layered MCP interface.

Canonical GitHub repository:

- `https://github.com/minsing-jin/civStation`

Current package and module names are still:

- Python package: `computer-use-test`
- Python module: `computer_use_test`

## Languages

- [한국어 README](README.ko.md)
- [English README](README.en.md)
- [中文 README](README.zh.md)

## Quick Links

- [Context README](computer_use_test/agent/modules/context/README.md)
- [Strategy README](computer_use_test/agent/modules/strategy/README.md)
- [Router README](computer_use_test/agent/modules/router/README.md)
- [Primitive README](computer_use_test/agent/modules/primitive/README.md)
- [HitL README](computer_use_test/agent/modules/hitl/README.md)
- [MCP README](computer_use_test/mcp/README.md)
- [Layered MCP Tool Map](docs/layered_mcp.md)

## Summary

CivStation is easiest to understand as a layered control system:

- `Context`: what the agent currently knows
- `Strategy`: what the agent should optimize for
- `Action`: how the next UI action is chosen and executed
- `HitL`: how a human can supervise and intervene
- `MCP`: a stable external contract over the same layers

For setup, HITL control, MCP workflows, and extensibility:

- [한국어](README.ko.md)
- [English](README.en.md)
- [中文](README.zh.md)
