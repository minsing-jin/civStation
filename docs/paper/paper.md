---
title: "CivStation: A layered computer-use and evaluation stack for Civilization VI agents"
tags:
  - Python
  - computer-use agents
  - game AI
  - Civilization VI
  - Model Context Protocol
  - human-in-the-loop
authors:
  - name: Minsing Jin
    affiliation: "1"
affiliations:
  - name: "TODO: add author affiliation for submission"
    index: 1
date: 25 March 2026
bibliography: paper.bib
---

# Summary

CivStation is an open-source Python stack for building, steering, and evaluating computer-use agents for *Sid Meier's Civilization VI*. Instead of treating the agent as one opaque prompt loop, CivStation decomposes gameplay into four explicit layers: `Context`, `Strategy`, `Action`, and `HitL` (human-in-the-loop). The repository combines a live turn executor, a layered Model Context Protocol (MCP) server, and a static evaluation framework for screenshot-based action benchmarking. This architecture is designed for researchers and practitioners who need more than end-to-end automation: they need inspectable intermediate state, controllable execution, and reusable interfaces for tooling and experiments.

At the action level, CivStation routes each screenshot to a specialized primitive and then plans UI actions in normalized coordinates before execution. At the supervision level, the same stack exposes pause, resume, stop, strategy redirection, and primitive overrides through a dashboard, HTTP endpoints, WebSocket control, and MCP tools. At the evaluation level, the project includes both a legacy Civ6-specific evaluator and a newer bounding-box-based evaluator that supports multiple acceptable ground-truth action sets and external agent integration. The result is a software package that can serve as a research scaffold for turn-based game agents, a practical control plane for operator-guided play, and an experimentation harness for action-level evaluation.

# Statement of need

Computer-use agents are increasingly evaluated on browser tasks, desktop interaction, and software engineering workflows, but many long-horizon interactive domains remain under-served by current open-source tooling. Turn-based strategy games are a useful example. They combine visual state interpretation, deferred planning, localized UI actions, and frequent need for human correction. A Civilization VI agent must reason about game state, route into the right UI mode, plan actions in different screen contexts, and recover from ambiguous or changing interface states across many turns.

Existing codebases often optimize for one part of this problem at the expense of the others. A monolithic agent can be convenient for demos, but it is harder to inspect, steer, and benchmark. Conversely, a narrow evaluator may make scoring easier while leaving real execution and control workflows out of scope. CivStation addresses this gap by packaging three needs into one system:

First, it offers a layered runtime for live computer use in a game environment. The runtime separates context observation, high-level strategy refinement, low-level action routing/planning, and human control so that each concern can be developed and tested independently.

Second, it exposes the runtime through MCP. MCP defines a client-server pattern for exposing tools, resources, and prompts to AI applications [@mcp_architecture]. In CivStation, this means the same internal layers can be used from external clients without importing internal Python modules directly. The MCP surface is session-based, supports import/export of state, and allows per-session adapter overrides.

Third, it provides evaluation software for static screenshot benchmarks. The bounding-box evaluator enables more realistic ground truth than exact-pixel matching and supports external agents through a simple process interface, which lowers the barrier to comparing models and prompting strategies.

This combination is useful to researchers studying game agents, computer-use interfaces, or operator-supervised autonomy, and to developers who need a controllable environment for iterative debugging rather than a single end-to-end black box.

# State of the field

Several open-source systems address adjacent parts of the computer-use landscape. Browser Use focuses on browser automation and making websites accessible to AI agents, with task execution, browser infrastructure, and persistent sessions centered on web workflows [@browser_use]. OpenHands provides a broader AI-driven development platform with an SDK, CLI, local GUI, and cloud deployment model for software-engineering agents [@openhands_docs; @openhands_repo]. These systems are valuable general-purpose agent platforms, but their primary abstractions are not tailored to turn-based game-state control or to action-level game evaluation.

On the research side, Voyager demonstrates how a large language model can support open-ended embodied control in Minecraft through curriculum generation, a growing skill library, and iterative self-improvement [@voyager]. Voyager is an important reference for long-horizon game agents, but it targets lifelong skill acquisition in a 3D embodied sandbox rather than a layered operator-facing control surface for UI interaction in a turn-based strategy game.

CivStation differs from these systems in three ways. First, it is centered on a specific but demanding domain: Civilization VI computer use. Second, it makes intermediate layers explicit instead of collapsing them into a single agent abstraction. Third, it couples live execution with an evaluation harness inside the same repository. To the best of our knowledge from the repository materials and web sources reviewed for this draft, there is not an equally focused open-source stack that combines layered game control, MCP exposure, human intervention paths, and screenshot-based action evaluation for Civilization VI in one package.

# Software design

The core design choice in CivStation is to separate the agent into layers with different responsibilities. The `Context` layer stores global game state, high-level notes, threats, opportunities, and primitive-local execution history. The `Strategy` layer converts free-form human guidance or autonomous updates into a structured plan that downstream primitives can consume. The `Action` layer is divided into routing and primitive-specific planning, and the `HitL` layer handles lifecycle control and directives.

This decomposition matters because Civilization VI presents qualitatively different interface states. A research-selection screen, a policy-card screen, a city-production prompt, and a unit movement prompt require different logic and different prompting context. CivStation therefore uses a primitive registry to route screenshots into specialized handlers. In the repository snapshot used for this draft, the registry contains 12 routable primitives and 2 human-forced primitives, including research, culture, religion, policy, diplomacy, voting, governor, city production, and unit operations. This keeps the active prompt smaller and makes failure analysis easier than in a monolithic action generator.

The project also normalizes coordinates before execution. Vision-language models emit actions in a fixed coordinate space, and the runtime converts these into device-specific screen coordinates immediately before execution. This allows the same plan representation to be reused across display configurations while isolating platform-specific details to the execution boundary.

Another important decision is the layered MCP server. CivStation exposes 29 MCP tools in the current implementation, spanning session management, context, strategy, memory, action, human control, and full workflow orchestration. Sessions isolate context and recent artifacts so that external tools can inspect, replay, or branch agent state without sharing one global mutable process. Adapter overrides make it possible to swap routers, planners, observers, refiners, or executors per session while preserving one external contract. This is a practical application of MCP's client-server design for context exchange [@mcp_architecture].

Finally, the evaluation subsystem complements the live runtime. The legacy evaluator checks Civ6 primitive selection and action generation with coordinate tolerance, while the newer `bbox_eval` subsystem supports bounding-box targets, multiple valid action sequences, detailed sequence metrics, and external agent runners. This makes CivStation useful not only for playing the game, but also for reproducible action-level comparisons across agent implementations.

# Research impact statement

The repository examined for this draft is still alpha-stage software, but it already shows concrete signs of research utility. The package is distributed under the MIT license, contains 123 Python source files and 48 files under `tests/`, and includes continuous integration for linting and test execution across Python 3.12 and 3.13. In the local environment used for this draft, targeted tests for the layered MCP server (`3` tests) and the bounding-box scorer (`34` tests) passed under Python 3.11.11. The full test suite did not run in the base interpreter because required dependencies such as `python-dotenv`, `Pillow`, and `google-genai` were not installed there; this is an environment limitation rather than direct evidence about project correctness.

The near-term research significance of CivStation comes from its combination of concerns. Researchers can use the same codebase to: 1) run a live Civilization VI agent with structured human supervision, 2) expose that runtime to external tooling through MCP, and 3) benchmark action-level behavior on static screenshots. This integrated setup is well suited for studies of agent controllability, primitive decomposition, prompt specialization, intervention timing, and evaluation methodology in visual interactive environments.

The software should also be useful beyond Civilization VI. The `bbox_eval` framework and the layered MCP pattern are not intrinsically game-specific, and they provide reusable patterns for domains where visual state, discrete UI modes, and partial human control all matter. In that sense, CivStation is both a domain application and a reference architecture for layered computer-use research software.

# AI usage disclosure

This draft paper was prepared with the assistance of a generative AI coding and writing system that read repository materials, synthesized code structure, and produced a first-pass manuscript. All technical claims, metadata, and references should be reviewed and corrected by the human authors before submission. The repository materials reviewed for this draft do not clearly disclose whether generative AI was also used in software implementation or documentation authoring; the authors should update this section to describe any such use and how AI-generated outputs were verified.

# Acknowledgements

No external funding information was available in the repository materials reviewed for this draft. The authors should update this section with funding, institutional support, or contributor acknowledgements as appropriate for submission.

# References
