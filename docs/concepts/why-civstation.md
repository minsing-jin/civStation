# Why CivStation

CivStation exists for people who want more than "run the bot and hope."

## The Core Idea

Civilization VI is not a one-click environment. It is a long-horizon, stateful strategy game where locally valid actions can still be globally poor many turns later.

That makes CivStation useful as both:

- a practical VLM runtime for playing Civ6
- a controllable research and engineering substrate for long-horizon visual agents

## What the README Is Actually Claiming

The README makes a few clear philosophical claims:

- the system should be inspectable instead of opaque
- human strategy should stay in the loop
- runtime work should be split by responsibility
- the same architecture should be accessible through MCP
- evaluation should exist outside live gameplay

Those are product choices, not just implementation details.

## Why the Project Is Structured This Way

- `Layered by design`: context, strategy, action, and HitL fail differently
- `Human-steerable`: strategy drift should be correctable during a live run
- `MCP-first`: external tools should not have to import unstable internals
- `Runtime-separated`: background reasoning should not block action execution
- `Operator-friendly`: the dashboard, WebSocket, and mobile controller are first-class

## The Short Version

CivStation is not just "a VLM that clicks Civ6."

It is a system for letting VLMs play Civilization VI while keeping the run observable, steerable, and aligned over many turns.
