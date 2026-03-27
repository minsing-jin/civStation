# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Civilization VI game agent evaluation system that uses Vision-Language Models (VLMs) to control the game via computer vision and action primitives. The agent captures screenshots, routes to the appropriate primitive, generates actions with normalized coordinates, and executes them via PyAutoGUI.

## Commands

```bash
# Install
make install              # Install deps + pre-commit hooks

# Lint & Format
make lint                 # Ruff check only
make format               # Ruff format + fix
make check                # Lint + type check (ty)
ruff check --fix          # Auto-fix lint issues

# Test
make test                 # Run all tests
pytest tests/evaluator/civ6_eval/ -v                    # All evaluator tests
pytest tests/evaluator/civ6_eval/test_tolerance.py -v   # Single test file
pytest -m "not integration"                              # Skip integration tests

# Coverage
make coverage

# Run the agent
python -m civStation.agent.turn_runner --provider claude
python -m civStation.agent.turn_runner --provider gemini \
    --router-model gemini-2.0-flash --planner-model gemini-2.5-pro
```

## Architecture

### Core Flow (One Turn)

```
Screenshot → Router (VLM) → Select Primitive → Primitive (VLM) → AgentPlan → Execute via PyAutoGUI
```

1. `capture_screen_pil()` captures screenshot + logical resolution
2. Router classifies game state → selects one of 6 primitives
3. Primitive analyzes screenshot → generates action with normalized coords (0-1000)
4. `execute_action()` converts normalized → logical screen → PyAutoGUI

### Key Modules

- **`agent/turn_runner.py`** — Orchestrates the one-turn execution flow. Supports separate VLM providers for routing vs planning.
- **`agent/models/schema.py`** — Pydantic models using discriminated unions for type-safe actions (`ClickAction`, `KeyPressAction`, `DragAction`, etc.) and `AgentPlan`.
- **`utils/llm_provider/base.py`** — `BaseVLMProvider` abstract class. Subclasses only implement `_send_to_api()` and content builders. Shared logic: `call_vlm()`, `analyze()`, `parse_to_agent_plan()`, `_parse_action_json()`.
- **`utils/llm_provider/__init__.py`** — Factory: `create_provider(name, api_key, model)` creates Claude/Gemini/GPT/Mock providers.
- **`utils/screen.py`** — `capture_screen_pil()`, `execute_action()`, `norm_to_real()`. Handles Mac Retina coordinate mismatch.
- **`utils/prompts/primitive_prompt.py`** — Prompt templates (Korean) for each primitive. `JSON_FORMAT_INSTRUCTION` is a template string formatted with `normalizing_range` at call time via `get_primitive_prompt(name, normalizing_range)`.
- **`utils/prompts/action_prompt.py`** — System prompt template for the live agent's coordinate-normalized action format.
- **`agent/modules/router/primitive_registry.py`** — Central registry defining all primitives (criteria, prompts, priorities). Auto-generates `ROUTER_PROMPT` and `PRIMITIVE_NAMES`.

### Primitive System

The agent uses a **primitive-based architecture** where game states are classified into specialized handlers:

- **Primitive Registry** (`agent/modules/router/primitive_registry.py`) — Single source of truth defining all primitives with criteria, prompts, and priorities.
- **Available Primitives**: `unit_ops_primitive`, `popup_primitive`, `governor_primitive`, `research_select_primitive`, `city_production_primitive`, `science_decision_primitive`, `culture_decision_primitive`, `diplomatic_primitive`, `combat_primitive`, `policy_primitive`.
- Adding new primitives: Add entry to `PRIMITIVE_REGISTRY` with `criteria`, `prompt`, and `priority`. Everything else auto-updates.

### High-Level Strategy

The system supports passing a `--strategy` (or `-s`) flag to `turn_runner.py` to guide decision-making:
- Strategy text is injected into primitive prompts via `high_level_strategy` parameter.
- Default strategy (if not provided): "과학 승리를 목표로 함" (Science Victory).
- Strategy influences primitive decisions (e.g., governor selection, policy cards, tech tree choices).

### Two JSON Formats

The codebase uses two distinct JSON response formats:

1. **Live agent** (`action_prompt.py` / `_parse_action_json`): Single flat action object — `{"action": "click", "x": 500, "y": 300, ...}`
2. **Static evaluation** (`primitive_prompt.py` / `parse_to_agent_plan`): Actions array — `{"reasoning": "...", "actions": [{"type": "click", ...}]}`

`turn_runner.py` uses the live agent format via `provider.analyze()`.

### Evaluator

- **`evaluator/civ6/static_eval/`** — Static evaluation comparing predicted actions against ground truth with ±5 pixel tolerance.
- `GroundTruth` → `BaseEvaluator.evaluate_single()` → `EvalResult` (primitive_match, action_sequence_match).
- Test fixtures in `tests/evaluator/civ6_eval/fixtures/` (ground_truth.json, screenshots/).

## Context System

- **Context Manager** (`agent/modules/primitive_context/context_manager.py`) — Currently minimal, planned to track game state across turns.
- **Temporal Context** — Primitives can receive `context` parameter with game state info (turn number, city stats, production items, etc.).
- Context is passed to primitive prompts via `get_primitive_prompt(context=...)`.

## Conventions

- **Ruff config**: line-length 120, rules E/W/F/I/B/UP. Pre-commit runs `ruff check --fix` and `ruff format`.
- **Prompts are in Korean** to match the Korean version of Civ6 game UI.
- **Coordinate normalization**: VLM always works with 0-`normalizing_range` (default 1000). Conversion to screen coords happens only in `screen.py`.
- **Environment variables**: `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `OPENAI_API_KEY` (loaded via dotenv).
- **pytest testpaths**: `tests/evaluator`. Integration tests are marked with `@pytest.mark.integration`.
- **Provider factory**: Use `create_provider(provider_name, model)` from `utils/llm_provider/__init__.py` to instantiate VLM providers.
- **Mac Retina screens**: `screen.py` handles logical vs physical pixel coordinate mismatch.