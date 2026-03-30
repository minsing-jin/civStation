# Governor Secret Society Observation Trace Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `governor_primitive` force at least one downward observation scroll, add a secret-society appointment branch that can merge into promotion, and lock governor Rich trace behavior into tests.

**Architecture:** Keep all governor execution logic inside `GovernorProcess` so observation, branch selection, and branch merging remain code-owned. Reuse the shared runtime trace feed in `turn_executor.py`, adding governor-focused regression coverage instead of a second trace system.

**Tech Stack:** Python, pytest, Rich live logger, short-term memory choice catalog

---

### Task 1: Governor observation regression

**Files:**
- Modify: `tests/agent/modules/primitive/test_multi_step_process.py`
- Modify: `civStation/agent/modules/primitive/multi_step_process.py`

**Step 1: Write the failing tests**

- Add a test that the first governor observation does not terminate scanning even when `end_of_list=true`.
- Add a test that governor observation completes only after at least one downward scroll and then no new candidates are found.

**Step 2: Run test to verify it fails**

Run: `pytest tests/agent/modules/primitive/test_multi_step_process.py -k "governor and observation" -v`

Expected: FAIL because governor currently allows immediate `choose_from_memory`.

**Step 3: Write minimal implementation**

- Track the minimum one-pass downward scan rule in `GovernorProcess.consume_observation()`.
- Only allow `choose_from_memory` after at least one successful downward scroll and a no-new-candidates/end-of-list confirmation.

**Step 4: Run test to verify it passes**

Run: `pytest tests/agent/modules/primitive/test_multi_step_process.py -k "governor and observation" -v`

Expected: PASS

### Task 2: Secret-society governor branch

**Files:**
- Modify: `tests/agent/modules/primitive/test_multi_step_process.py`
- Modify: `civStation/agent/modules/primitive/multi_step_process.py`

**Step 1: Write the failing tests**

- Add a test that secret-society best choice with active `임명` goes to a dedicated secret-society branch.
- Add a test that secret-society best choice with active `진급` still goes to the normal promote branch.
- Add a test that the secret-society branch does `appoint click -> esc -> esc -> post-check`, and merges into promote when a green promote button is visible.

**Step 2: Run test to verify it fails**

Run: `pytest tests/agent/modules/primitive/test_multi_step_process.py -k "governor and secret" -v`

Expected: FAIL because no secret-society branch exists today.

**Step 3: Write minimal implementation**

- Add secret-society branch stages to `GovernorProcess`.
- Route best-choice decisions by candidate note and active action type.
- Add a post-appointment classifier that merges to promote or ends cleanly.

**Step 4: Run test to verify it passes**

Run: `pytest tests/agent/modules/primitive/test_multi_step_process.py -k "governor and secret" -v`

Expected: PASS

### Task 3: Governor trace regression

**Files:**
- Modify: `tests/agent/test_turn_executor.py`
- Modify: `civStation/agent/turn_executor.py` if needed

**Step 1: Write the failing test**

- Add a governor multi-step test that asserts Rich receives `observe`, `plan`, `exec`, and `stage` trace lines during the governor scan flow.

**Step 2: Run test to verify it fails**

Run: `pytest tests/agent/test_turn_executor.py -k "governor and trace" -v`

Expected: FAIL if governor trace is not being surfaced consistently.

**Step 3: Write minimal implementation**

- If needed, ensure governor uses the shared primitive trace feed the same way as other multi-step primitives.

**Step 4: Run test to verify it passes**

Run: `pytest tests/agent/test_turn_executor.py -k "governor and trace" -v`

Expected: PASS
