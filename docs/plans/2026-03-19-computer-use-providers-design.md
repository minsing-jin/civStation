# Computer-Use Providers Design

**Date:** 2026-03-19

**Goal:** Add OpenAI and Anthropic computer-use providers that fit the existing `BaseVLMProvider` contract without replacing the current router / executor architecture.

## Context

The codebase already owns the runtime loop:

1. capture screenshot
2. route to primitive
3. plan next action
4. execute the action locally

That means the best integration point is the planner-side `analyze()` call, not a separate autonomous agent loop.

## Decision

Implement two new providers:

- `openai-computer`
- `anthropic-computer`

These providers will:

- reuse the existing GPT / Claude `_send_to_api()` behavior for router calls and JSON-based parsing flows
- override only `analyze()` to call the provider-specific computer-use API/tool
- keep `analyze_multi()` on the normal JSON path so existing multi-action primitives continue to work

## Why this shape

### Option 1: Replace the whole agent loop

Rejected. The repository already has stable screenshot capture, normalization, execution, HITL, and primitive routing. Replacing that loop would be a much larger architecture change than the user asked for.

### Option 2: Add a planner-side computer-use adapter

Accepted. This fits the current abstraction with the smallest behavioral change and preserves existing primitives.

### Option 3: Wrap browser-use / macOS-use as providers

Deferred. Those libraries are higher-level agent loops. They are possible future integrations, but they are a worse fit than native OpenAI / Anthropic computer-use tools for this repo.

## Data flow

For normal router and JSON flows:

- `route_primitive()` and `analyze_multi()` continue using `_send_to_api()` exactly as today.

For planner single-action flows:

1. `provider.analyze()` preprocesses the PIL image with the existing image pipeline.
2. The provider calls the external computer-use API/tool with the processed image and prompt.
3. The first executable tool action is translated into `AgentAction`.
4. Pixel coordinates from the tool response are normalized back into the repo's `0..normalizing_range` coordinate contract.

## Constraints

- Do not change executor semantics.
- Do not require absolute screen metadata in the provider interface.
- Do not break multi-action primitives such as policy drag bundles.
- Keep provider creation compatible with existing CLI/config wiring.

## Files to touch

- Create: `computer_use_test/utils/llm_provider/openai_computer.py`
- Create: `computer_use_test/utils/llm_provider/anthropic_computer.py`
- Create: `computer_use_test/utils/llm_provider/computer_use_actions.py`
- Modify: `computer_use_test/utils/llm_provider/__init__.py`
- Modify: `README.md`
- Test: `tests/utils/test_openai_computer_provider.py`
- Test: `tests/utils/test_anthropic_computer_provider.py`

## Testing strategy

- unit-test action translation from OpenAI / Anthropic tool payloads
- unit-test normalization from pixel coordinates to repo-normalized coordinates
- unit-test that router-style `_send_to_api()` behavior remains inherited and unchanged
- unit-test provider factory registration

## Non-goals

- no full external agent loop
- no browser-use / Skyvern / Stagehand integration in this change
- no changes to unrelated dirty files already present in the branch
