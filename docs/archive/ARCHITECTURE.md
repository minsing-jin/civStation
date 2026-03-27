# Architecture Guide

This document describes the internal structure of the Civ6 Computer-Use Agent
for contributors who want to extend, debug, or build on top of the project.

For a quick-start overview, see [README.md](README.md).

---

## High-Level Data Flow

```
┌────────────────────────────────────────────────────────────────────┐
│                        turn_runner.py (CLI)                        │
│  parse args → setup providers → setup HITL → call run_one_turn()  │
└────────────────────────┬───────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────── turn_executor.py ──────────────────────────────┐
│                                                                    │
│  1. capture_screen_pil()  ─────────────────  PIL Image             │
│                                                  │                 │
│  2. route_primitive(router_provider, image)       │                 │
│       └─ VLM classifies screenshot               │                 │
│       └─ Returns: RouterResult(primitive, turn#)  │                 │
│                                                  │                 │
│  3. plan_action(planner_provider, image, ...)     │                 │
│       └─ VLM generates action (normalized 0-1000)│                 │
│       └─ Returns: AgentAction                    │                 │
│                                                  │                 │
│  4. execute_action(action, screen_w, screen_h)   │                 │
│       └─ norm_to_real() → PyAutoGUI              │                 │
│                                                  │                 │
│  5. Record in ContextManager                     │                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
civStation/
├── __init__.py                    # Package entry: __version__, create_provider
│
├── agent/                         # Agent core
│   ├── __init__.py                # Re-exports: run_one_turn, Action, AgentPlan, etc.
│   ├── turn_runner.py             # CLI entry point (ConfigArgParse + config.yaml)
│   ├── turn_executor.py           # Pure execution: route → plan → execute → record
│   │
│   ├── models/
│   │   ├── __init__.py            # Re-exports all Pydantic action models
│   │   └── schema.py             # ClickAction, DragAction, KeyPressAction, AgentPlan...
│   │
│   └── modules/
│       ├── __init__.py            # Namespace only (no heavy imports)
│       │
│       ├── router/                # Screenshot → Primitive classification
│       │   ├── __init__.py        # Re-exports: PrimitiveRouter, RouterResult, PRIMITIVE_REGISTRY
│       │   ├── base_router.py     # Abstract PrimitiveRouter base class
│       │   ├── router.py          # Civ6Router, Civ6MockRouter (filename-based for tests)
│       │   └── primitive_registry.py  # Single source of truth for all primitives
│       │
│       ├── primitive/             # Game-state specialized handlers
│       │   ├── __init__.py        # Re-exports: BasePrimitive, UnitOpsPrimitive, ...
│       │   ├── base_primitive.py  # Abstract BasePrimitive base class
│       │   └── primitives.py      # Concrete primitives (UnitOps, Popup, Research, ...)
│       │
│       ├── context/               # Game state tracking across turns
│       │   ├── __init__.py        # Re-exports: ContextManager, MacroTurnManager, ...
│       │   ├── context_manager.py # Singleton holding all game context
│       │   ├── context_updater.py # Background VLM analysis of screenshots
│       │   ├── global_context.py  # Turn number, resources, era, etc.
│       │   ├── primitive_context.py   # Per-primitive context (recent actions, etc.)
│       │   └── macro_turn_manager.py  # Detects in-game turn boundaries
│       │
│       ├── strategy/              # High-level strategy generation
│       │   ├── __init__.py        # Re-exports: StrategyPlanner, VictoryType, ...
│       │   ├── strategy_planner.py    # LLM-powered strategy refinement
│       │   └── prompts/           # Strategy prompt templates
│       │
│       ├── knowledge/             # RAG-based knowledge augmentation
│       │   ├── __init__.py        # Re-exports: KnowledgeManager, DocumentRetriever, ...
│       │   ├── knowledge_manager.py   # Orchestrates document + web retrieval
│       │   ├── document_retriever.py  # Civopedia JSON index search
│       │   └── web_search_retriever.py    # Tavily web search integration
│       │
│       └── hitl/                  # Human-in-the-Loop control system
│           ├── __init__.py        # Re-exports: CommandQueue, AgentGate, RelayClient, ...
│           ├── command_queue.py   # Thread-safe directive queue (STOP/PAUSE/CHANGE_STRATEGY)
│           ├── agent_gate.py      # External lifecycle control (start/stop/pause)
│           ├── input_manager.py   # Abstracts input sources (stdin, chatapp, relay)
│           ├── queue_listener.py  # Stdin listener for local HITL
│           ├── turn_checkpoint.py # Inter-turn checkpoint prompts
│           ├── status_ui/
│           │   ├── server.py      # FastAPI server (REST + WebSocket + Dashboard)
│           │   ├── state_bridge.py    # Thread-safe agent ↔ UI bridge
│           │   ├── websocket_manager.py   # WS connection pool + broadcast
│           │   ├── screen_streamer.py     # Real-time screenshot streaming via WS
│           │   └── dashboard.py   # Built-in HTML/JS dashboard
│           └── relay/
│               └── relay_client.py    # Remote HITL via external relay server
│
├── evaluation/                    # Evaluation framework
│   ├── __init__.py                # Re-exports: BaseEvaluator, run_evaluation, ...
│   ├── evaluator/
│   │   ├── __init__.py            # Re-exports: BaseEvaluator, EvalResult, ...
│   │   └── action_eval/
│   │       ├── __init__.py        # Re-exports base classes + within_tolerance
│   │       ├── base_static_primitive_evaluator.py  # Abstract evaluator + GroundTruth/EvalResult
│   │       ├── interfaces.py      # Re-exports + within_tolerance() helper
│   │       ├── civ6_eval/
│   │       │   ├── __init__.py    # Civ6StaticEvaluator, load_ground_truth_from_json
│   │       │   ├── civ6_static_evaluator.py   # Civ6-specific comparison logic
│   │       │   └── main.py        # CLI runner + ground truth JSON loader
│   │       └── bbox_eval/         # Bounding-box evaluation framework
│   │           ├── __init__.py    # Full re-exports: run_evaluation, schemas, agents, ...
│   │           ├── runner.py      # evaluate_case, run_evaluation
│   │           ├── scorer.py      # Levenshtein, aggregate metrics
│   │           ├── schema.py      # GTAction, BBox, CaseResult, EvalReport, ...
│   │           ├── dataset_loader.py  # JSONL dataset loading + validation
│   │           ├── cli.py         # Typer CLI
│   │           ├── __main__.py    # python -m entrypoint
│   │           └── agents/        # Agent runners (builtin, subprocess, mock)
│   ├── dataset/                   # Dataset collection (placeholder)
│   └── metric/                    # Evaluation metrics (placeholder)
│
└── utils/                         # Shared utilities
    ├── __init__.py                # Re-exports: load_env_variable
    ├── utils.py                   # API key loader with provider mapping
    ├── screen.py                  # Screenshot capture, resize_for_vlm, execute_action
    ├── rich_logger.py             # Rich-based structured terminal logging
    ├── llm_provider/
    │   ├── __init__.py            # Factory: create_provider, get_available_providers
    │   ├── base.py                # BaseVLMProvider (abstract), MockVLMProvider
    │   ├── claude.py              # ClaudeVLMProvider (Anthropic API)
    │   ├── gemini.py              # GeminiVLMProvider (Google GenAI API)
    │   ├── gpt.py                 # GPTVLMProvider (OpenAI API)
    │   └── parser.py              # JSON parsing: parse_action_json, parse_to_agent_plan
    ├── prompts/
    │   ├── __init__.py            # Re-exports all prompt templates
    │   ├── primitive_prompt.py    # Korean prompt templates for each primitive
    │   └── action_prompt.py       # Live agent action format template
    ├── chatapp/                   # Chat app integrations (Discord, WhatsApp)
    └── debug/                     # Debug utilities (TurnValidator, context logging)
```

---

## Key Design Decisions

### 1. Primitive-Based Architecture

Instead of a single monolithic prompt, the agent classifies each screenshot into
one of 10 specialized **primitives** (unit operations, popup handling, research
selection, etc.). Each primitive has its own prompt template optimized for that
game situation.

**Why:** Civ6 has very different UI states. A single prompt would be too long and
ambiguous. Specialized prompts produce better actions and are easier to iterate on.

**Extension point:** Add a new entry to `PRIMITIVE_REGISTRY` in
`primitive_registry.py`. The router prompt, primitive name list, and prompt lookup
all auto-update.

### 2. Normalized Coordinates (0-1000)

The VLM always outputs coordinates in a `0-normalizing_range` space (default 1000).
`screen.py` handles the conversion to actual screen pixels, including Mac Retina
display mismatch where screenshot pixels ≠ PyAutoGUI coordinates.

**Why:** This decouples the VLM from screen resolution. The same model output works
on any display.

### 3. Separate Router and Planner Providers

The CLI supports `--router-provider` and `--planner-provider` flags to use different
VLM models for routing (cheap/fast model) vs. planning (expensive/accurate model).

**Why:** Routing is a simple classification task — a fast model like `gemini-2.0-flash`
works well. Planning requires deeper reasoning and benefits from a stronger model.

### 4. Image Optimization for VLM Inference

Screenshots are downscaled to `VLM_MAX_LONG_EDGE` (default 1280px) and encoded as
JPEG (quality 80) before being sent to VLMs. This reduces image tokens by ~75%
while keeping game-UI text readable.

Configuration: Adjust `VLM_MAX_LONG_EDGE` and `VLM_JPEG_QUALITY` in `screen.py`.
Disable per-provider: `create_provider(..., resize_for_vlm=False)`.

### 5. Two JSON Formats

| Context | Format | Parser |
|---------|--------|--------|
| Live agent (turn_executor) | Single flat action: `{"action": "click", ...}` | `parse_action_json()` |
| Static evaluation (primitives) | Actions array: `{"actions": [...]}` | `parse_to_agent_plan()` |

### 6. Context System

`ContextManager` is a singleton that tracks:
- **GlobalContext:** Turn number, era, resources, science/culture output
- **PrimitiveContext:** Recent actions per primitive
- **HighLevelContext:** Strategy, threats, opportunities

`ContextUpdater` runs VLM analysis in a background thread to extract game state
from screenshots without blocking the main turn loop.

### 7. HITL (Human-in-the-Loop)

The agent supports multiple control channels:
- **Web UI** — FastAPI server with REST + WebSocket + HTML dashboard
- **Chat Apps** — Discord/WhatsApp bot integration
- **Remote Relay** — WebSocket relay for headless deployment

All channels feed into `CommandQueue`, which is checked at the start of each turn.
Directive priority: `STOP > PRIMITIVE_OVERRIDE > PAUSE > CHANGE_STRATEGY`.

---

## Evaluation Framework

### Static Evaluation (Primitive-Based)

```
GroundTruth(screenshot, expected_primitive, expected_actions)
    │
    ▼
BaseEvaluator.evaluate_single(gt)
    ├── Router.route(screenshot) → selected_primitive
    ├── Primitive.generate_plan_and_action(screenshot) → AgentPlan
    └── _compare(gt, selected, plan) → EvalResult
```

### BBox Evaluation

The `bbox_eval` package supports bounding-box-based evaluation where ground truth
actions define target regions (bounding boxes) rather than exact pixel coordinates.
Supports multiple acceptable GT action sets per case and external agent integration.

```bash
python -m civStation.evaluation.evaluator.action_eval.bbox_eval \
    --dataset dataset.jsonl --provider mock --verbose
```

---

## Adding a New VLM Provider

1. Create `utils/llm_provider/my_provider.py`
2. Subclass `BaseVLMProvider`
3. Implement: `_send_to_api()`, `_build_image_content()`,
   `_build_pil_image_content()`, `_build_text_content()`, `get_provider_name()`
4. Register in `utils/llm_provider/__init__.py` → `create_provider()` factory

---

## Tests

```
tests/
├── evaluator/                 # Evaluation tests (primary test suite)
│   └── civ6_eval/
│       ├── test_bbox_eval_integration.py  # BBox evaluation integration tests
│       ├── test_evaluation_integration.py # Static evaluation integration tests
│       ├── test_tolerance.py   # Coordinate tolerance unit tests
│       └── test_json_parsing.py    # VLM response JSON parsing tests
├── agent/modules/             # Agent module unit tests
│   ├── hitl/
│   │   ├── test_command_queue.py   # CommandQueue thread safety
│   │   ├── test_relay_client.py    # RelayClient message routing
│   │   └── test_state_bridge.py    # AgentStateBridge snapshot
│   └── context/
│       └── test_macro_turn_manager.py  # MacroTurnManager turn detection
└── rough_test/                # Experimental / research tests
```

Run: `pytest tests/ -v` (or `make test`)
