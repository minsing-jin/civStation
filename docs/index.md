<div class="hero-grid" markdown>

<div class="hero-panel" markdown>

<span class="hero-kicker">Overview</span>

<div class="language-switch" markdown>

<a class="md-button md-button--primary" href="./">English</a>
<a class="md-button" href="ko/">한국어</a>
<a class="md-button" href="zh/">中文</a>

</div>

<h1 class="landing-title">CivStation</h1>

Runtime, tooling, and control surfaces for letting VLMs play Civilization VI.

<p class="landing-lead">CivStation is built for running vision-language models against the Civilization VI UI. It combines screenshot observation, primitive routing, action planning, local execution, evaluation, and human override into one working system.</p>

[Install and run](getting-started/installation.md){ .md-button .md-button--primary }
[Run a live session](getting-started/first-live-run.md){ .md-button }
[Read the architecture](concepts/mental-model.md){ .md-button }

</div>

</div>

<div class="ornament-rule"></div>

## Start Here

<div class="grid cards" markdown>

-   **Quickstart**

    ---

    Install the project, set API keys, run the agent with the status UI, and open the dashboard.

    [Go to quickstart](getting-started/quickstart.md)

-   **First live run**

    ---

    Learn the runtime flow, what the dashboard exposes, and where logs and run artifacts land.

    [Run a live session](getting-started/first-live-run.md)

-   **Architecture overview**

    ---

    Understand how the VLM runtime, HitL strategy refinement, routing, planning, and execution fit together.

    [Read the concepts](concepts/mental-model.md)

</div>

## What CivStation Actually Covers

<div class="grid cards" markdown>

-   **Live VLM gameplay**

    ---

    Run a VLM against the Civ6 screen, route the current UI state into a primitive, plan the next action, and execute it locally.

-   **Operator control**

    ---

    Pause, resume, stop, change strategy, override primitives, and discuss the next move while the run is live.

-   **Layered MCP**

    ---

    Expose the same architecture through sessions, prompts, resources, adapter overrides, and workflow tools instead of depending on internal imports.

-   **Evaluation**

    ---

    Test routing and action quality with bbox-style evaluation and related fixtures without having to rely only on live gameplay runs.

</div>

<div class="ornament-rule"></div>

## Documentation Layout

| Section | What you get |
| --- | --- |
| `Getting Started` | Installation, quickstart, first live run |
| `Concepts` | architecture overview, layers, execution loop, HITL, MCP |
| `Guides` | Operating the agent, control surfaces, providers, evaluation |
| `Reference` | CLI flags, `config.yaml`, MCP tool list, folder map |
| `Development` | Contribution workflow, testing, extension points, release notes |
| `Appendix` | legacy pages, historical summaries, design notes |

## Canonical Names

Use these names consistently when you document, script, or integrate the project:

- Product name: `CivStation`
- GitHub repository: `minsing-jin/civStation`
- Python package: `civStation`
- Python module: `civStation`

## Recommended Reading Order

1. Start with [Quickstart](getting-started/quickstart.md).
2. Read [Architecture Overview](concepts/mental-model.md) and [Layers](concepts/layers.md).
3. Use [Running the Agent](guides/running-the-agent.md) and [Control and Discussion](guides/control-and-discussion.md) for day-to-day operation.
4. Use [MCP Tools](reference/mcp-tools.md) and [Extending the Stack](development/extending-the-stack.md) when building on top of the project.
