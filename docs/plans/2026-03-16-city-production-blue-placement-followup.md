# City Production Blue Placement Followup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a city-production placement follow-up flow that handles blue purchasable tiles by re-clicking the purchased tile before the final build confirmation.

**Architecture:** Extend `CityProductionProcess` with a lightweight post-placement resolver stage that classifies the post-click UI as still-placement or confirm. Persist the last placement click in short-term memory so the process can deterministically re-click the same tile after a blue-tile purchase, while strengthening placement prompts to require current-gold, adjacency, and high-level-strategy reasoning.

**Tech Stack:** Python, pytest, Ruff, existing multi-step process and short-term memory modules.

---

### Task 1: Lock behavior with tests

**Files:**
- Modify: `tests/agent/modules/primitive/test_multi_step_process.py`
- Test: `tests/agent/modules/primitive/test_multi_step_process.py`

**Step 1: Write the failing tests**

Add tests for:
- blue-tile purchase follow-up transitioning from `production_place` to a re-click stage after the placement screen remains open
- deterministic re-click using the same coordinates before moving to `production_place_confirm`
- placement prompt/stage guidance explicitly mentioning current gold, adjacency bonus, and blue-tile re-click behavior

**Step 2: Run test to verify it fails**

Run: `pytest tests/agent/modules/primitive/test_multi_step_process.py -k "blue_tile or placement_stage_prompt" -v`
Expected: FAIL because the new placement follow-up stage and prompt text do not exist yet.

### Task 2: Implement placement follow-up state

**Files:**
- Modify: `civStation/agent/modules/memory/short_term_memory.py`
- Modify: `civStation/agent/modules/primitive/multi_step_process.py`

**Step 1: Write minimal implementation**

Add:
- placement-follow-up state to `ShortTermMemory` and checkpoint restore/capture
- a fast placement follow-up classifier in `CityProductionProcess`
- new stages for `resolve_placement_followup` and deterministic `production_place_reclick`
- state transitions that store the original placement click and re-click the same tile once when the post-click UI stays on the placement map

**Step 2: Run targeted tests to verify they pass**

Run: `pytest tests/agent/modules/primitive/test_multi_step_process.py -k "blue_tile or placement_stage_prompt" -v`
Expected: PASS

### Task 3: Verify the full regression slice

**Files:**
- Modify: `civStation/utils/prompts/primitive_prompt.py`
- Modify: `civStation/agent/modules/primitive/multi_step_process.py`
- Modify: `civStation/agent/modules/memory/short_term_memory.py`
- Test: `tests/agent/modules/primitive/test_multi_step_process.py`

**Step 1: Run broader verification**

Run: `pytest tests/agent/modules/primitive/test_multi_step_process.py -v`
Expected: PASS

**Step 2: Run static checks**

Run: `ruff check civStation/agent/modules/memory/short_term_memory.py civStation/agent/modules/primitive/multi_step_process.py civStation/utils/prompts/primitive_prompt.py tests/agent/modules/primitive/test_multi_step_process.py`
Expected: PASS

Run: `ruff format --check civStation/agent/modules/memory/short_term_memory.py civStation/agent/modules/primitive/multi_step_process.py civStation/utils/prompts/primitive_prompt.py tests/agent/modules/primitive/test_multi_step_process.py`
Expected: PASS

### Task 4: Commit and publish

**Files:**
- Modify: Git history / PR metadata

**Step 1: Commit**

```bash
git add docs/plans/2026-03-16-city-production-blue-placement-followup.md tests/agent/modules/primitive/test_multi_step_process.py civStation/agent/modules/memory/short_term_memory.py civStation/agent/modules/primitive/multi_step_process.py civStation/utils/prompts/primitive_prompt.py
git commit -m "feat: support blue-tile city placement follow-up"
```

**Step 2: Push and update PR**

```bash
git push -u origin Feature/#56
gh api repos/minsing-jin/civStation/pulls/57 -X PATCH -f body='...'
```
