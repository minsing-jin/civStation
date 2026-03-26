# Run Log Capture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** `turn_runner` 실행 중 터미널에 보이는 주요 출력을 최신 로그 파일 하나에 모두 기록하고, 새 실행 시 같은 파일을 덮어쓴다.

**Architecture:** `RunLogSession`이 root logging handler와 stdout/stderr tee를 함께 소유한다. 출력 경로를 한 세션 객체에 집중시켜 close 시점에 모두 원복하고, 기존 deterministic latest-log overwrite 동작은 유지한다.

**Tech Stack:** Python stdlib (`logging`, `sys`, `traceback`, `io` semantics), pytest

---

### Task 1: Add failing tests for stream tee capture

**Files:**
- Modify: `tests/utils/test_run_log_cache.py`
- Modify: `civStation/utils/run_log_cache.py`

**Step 1: Write the failing test**

Add tests that assert:

- `print("hello")` is written to the latest run log file
- `sys.stderr.write("oops\n")` is written to the latest run log file
- `session.close()` restores the previous `sys.stdout` and `sys.stderr`

**Step 2: Run test to verify it fails**

Run: `pytest tests/utils/test_run_log_cache.py -q`

Expected: FAIL because `RunLogSession` does not yet tee stdout/stderr.

**Step 3: Write minimal implementation**

Implement a small tee stream wrapper inside `civStation/utils/run_log_cache.py` and install it in `RunLogSession.__init__`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/utils/test_run_log_cache.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/utils/test_run_log_cache.py civStation/utils/run_log_cache.py docs/plans/2026-03-18-run-log-capture-design.md docs/plans/2026-03-18-run-log-capture.md
git commit -m "feat: capture runtime stdout and stderr in latest run log"
```

### Task 2: Verify overwrite and exception behavior still hold

**Files:**
- Modify: `tests/utils/test_run_log_cache.py`
- Modify: `civStation/utils/run_log_cache.py`

**Step 1: Write or adjust focused tests**

Keep the existing overwrite and exception-hook tests green after the stdout/stderr tee change.

**Step 2: Run targeted tests**

Run: `pytest tests/utils/test_run_log_cache.py -q`

Expected: PASS for root logging, overwrite, stdout/stderr tee, traceback, restoration.

**Step 3: Refactor only if needed**

If stream wrapper code is noisy, extract a tiny helper class without changing behavior.

**Step 4: Re-run tests**

Run: `pytest tests/utils/test_run_log_cache.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/utils/test_run_log_cache.py civStation/utils/run_log_cache.py
git commit -m "test: harden run log capture behavior"
```
