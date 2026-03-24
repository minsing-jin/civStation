# Computer-Use Providers Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add OpenAI and Anthropic computer-use providers that can be selected through the existing provider factory and used for planner-style `analyze()` calls without breaking current router or multi-action flows.

**Architecture:** Introduce a shared action-mapping helper for computer-use tool payloads, then implement `OpenAIComputerVLMProvider` and `AnthropicComputerVLMProvider` by subclassing the current GPT / Claude providers. Only `analyze()` is overridden to use computer-use APIs; `_send_to_api()` and `analyze_multi()` stay on the existing text+image JSON path.

**Tech Stack:** Python, pytest, existing `BaseVLMProvider` abstraction, OpenAI SDK, Anthropic SDK

---

### Task 1: Add failing action-mapping and factory tests

**Files:**
- Create: `tests/utils/test_openai_computer_provider.py`
- Create: `tests/utils/test_anthropic_computer_provider.py`

**Step 1: Write the failing tests**

- Add tests for OpenAI tool action -> `AgentAction` translation.
- Add tests for Anthropic tool-use input -> `AgentAction` translation.
- Add tests for coordinate normalization using the processed image size.
- Add tests proving the new providers are registered in `create_provider()`.

**Step 2: Run test to verify it fails**

Run: `pytest tests/utils/test_openai_computer_provider.py tests/utils/test_anthropic_computer_provider.py -q`
Expected: FAIL because the providers and action mappers do not exist yet.

**Step 3: Write minimal implementation**

- Add only the smallest helper / provider code needed for the first tests to pass.

**Step 4: Run test to verify it passes**

Run the same pytest command.

### Task 2: Implement shared computer-use action mapping

**Files:**
- Create: `computer_use_test/utils/llm_provider/computer_use_actions.py`

**Step 1: Add OpenAI action translation helpers**

- Map `click`, `double_click`, `move`, `drag`, `scroll`, `keypress`, `type`, and `wait`.
- Convert pixel coordinates into normalized coordinates.

**Step 2: Add Anthropic action translation helpers**

- Map `left_click`, `right_click`, `double_click`, `mouse_move`, `left_click_drag`, `scroll`, `key`, `type`, and `wait`.
- Reject or ignore unsupported tool payloads with clear errors.

**Step 3: Add safe extraction helpers**

- Support both dict-like fixtures and SDK object-like responses in tests.

**Step 4: Run focused tests**

Run: `pytest tests/utils/test_openai_computer_provider.py tests/utils/test_anthropic_computer_provider.py -q`

### Task 3: Implement OpenAI computer-use provider

**Files:**
- Create: `computer_use_test/utils/llm_provider/openai_computer.py`
- Modify: `computer_use_test/utils/llm_provider/__init__.py`

**Step 1: Subclass the existing GPT provider**

- Reuse normal `_send_to_api()` routing behavior from `GPTVLMProvider`.
- Override `analyze()` only.

**Step 2: Build the computer-use request**

- Preprocess the image with the existing pipeline.
- Call `client.responses.create(...)` with the computer-use tool and the user image+instruction input.

**Step 3: Translate the first executable computer action**

- Parse the response output.
- Convert the action to `AgentAction`.
- Preserve retry behavior on parse / validation failure.

**Step 4: Run focused tests**

Run: `pytest tests/utils/test_openai_computer_provider.py -q`

### Task 4: Implement Anthropic computer-use provider

**Files:**
- Create: `computer_use_test/utils/llm_provider/anthropic_computer.py`
- Modify: `computer_use_test/utils/llm_provider/__init__.py`

**Step 1: Subclass the existing Claude provider**

- Reuse normal `_send_to_api()` routing behavior from `ClaudeVLMProvider`.
- Override `analyze()` only.

**Step 2: Build the beta computer-use request**

- Preprocess the image with the existing pipeline.
- Call `client.beta.messages.create(...)` with the beta computer tool and user image+instruction content.

**Step 3: Translate the first tool-use block**

- Extract the `tool_use` block for the computer tool.
- Convert its `input` payload to `AgentAction`.

**Step 4: Run focused tests**

Run: `pytest tests/utils/test_anthropic_computer_provider.py -q`

### Task 5: Register providers and document usage

**Files:**
- Modify: `computer_use_test/utils/llm_provider/__init__.py`
- Modify: `README.md`

**Step 1: Register provider names**

- Add canonical names for OpenAI and Anthropic computer providers.
- Keep aliases if helpful, but prefer one canonical display name per vendor.

**Step 2: Document planner usage**

- Add an example using a normal router provider plus computer-use planner provider.

**Step 3: Run broader checks**

Run: `pytest tests/utils/test_openai_computer_provider.py tests/utils/test_anthropic_computer_provider.py tests/agent/test_turn_runner.py -q`

### Task 6: Final verification and delivery

**Files:**
- Modify: `README.md`
- Modify: `computer_use_test/utils/llm_provider/__init__.py`
- Create: `computer_use_test/utils/llm_provider/openai_computer.py`
- Create: `computer_use_test/utils/llm_provider/anthropic_computer.py`
- Create: `computer_use_test/utils/llm_provider/computer_use_actions.py`
- Create: `tests/utils/test_openai_computer_provider.py`
- Create: `tests/utils/test_anthropic_computer_provider.py`

**Step 1: Run full targeted verification**

Run: `pytest tests/utils/test_openai_computer_provider.py tests/utils/test_anthropic_computer_provider.py tests/agent/test_turn_runner.py -q`
Run: `ruff check computer_use_test/utils/llm_provider/__init__.py computer_use_test/utils/llm_provider/computer_use_actions.py computer_use_test/utils/llm_provider/openai_computer.py computer_use_test/utils/llm_provider/anthropic_computer.py tests/utils/test_openai_computer_provider.py tests/utils/test_anthropic_computer_provider.py`

**Step 2: Commit**

```bash
git add docs/plans/2026-03-19-computer-use-providers-design.md docs/plans/2026-03-19-computer-use-providers.md README.md computer_use_test/utils/llm_provider/__init__.py computer_use_test/utils/llm_provider/computer_use_actions.py computer_use_test/utils/llm_provider/openai_computer.py computer_use_test/utils/llm_provider/anthropic_computer.py tests/utils/test_openai_computer_provider.py tests/utils/test_anthropic_computer_provider.py
git commit -m "feat: add OpenAI and Anthropic computer-use providers"
```
