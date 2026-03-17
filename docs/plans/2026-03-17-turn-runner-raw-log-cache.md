# Turn Runner Raw Log Cache Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an opt-in raw run log cache for `computer_use_test.agent.turn_runner` that overwrites on each new run and captures plain logging plus uncaught tracebacks without altering existing Rich output.

**Architecture:** A new utility module owns the temp-file path, root logger file handler, and temporary `sys.excepthook` wrapper. `turn_runner.main()` opts into that utility at startup and always tears it down in cleanup so the behavior remains isolated to that entrypoint.

**Tech Stack:** Python standard library (`logging`, `pathlib`, `sys`, `tempfile`, `traceback`), `pytest`

---

### Task 1: Add util regression tests

**Files:**
- Create: `tests/utils/test_run_log_cache.py`
- Test: `tests/utils/test_run_log_cache.py`

**Step 1: Write the failing test**

```python
def test_session_writes_root_logger_records(tmp_path):
    session = run_log_cache.start_run_log_session(base_dir=tmp_path)
    ...
    assert "hello" in session.path.read_text()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/utils/test_run_log_cache.py -v`
Expected: FAIL because the module or API does not exist yet.

**Step 3: Write minimal implementation**

```python
class RunLogSession:
    ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/utils/test_run_log_cache.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/utils/test_run_log_cache.py computer_use_test/utils/run_log_cache.py
git commit -m "test: cover turn runner raw log cache"
```

### Task 2: Wire `turn_runner` into the new util

**Files:**
- Modify: `computer_use_test/agent/turn_runner.py`
- Modify: `computer_use_test/utils/__init__.py`
- Test: `tests/utils/test_run_log_cache.py`

**Step 1: Write the failing test**

```python
def test_second_session_overwrites_previous_contents(tmp_path):
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/utils/test_run_log_cache.py -v`
Expected: FAIL because overwrite or cleanup behavior is incomplete.

**Step 3: Write minimal implementation**

```python
run_log_session = start_run_log_session()
...
run_log_session.close()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/utils/test_run_log_cache.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add computer_use_test/agent/turn_runner.py computer_use_test/utils/__init__.py
git commit -m "feat: cache raw turn runner logs"
```

### Task 3: Verify traceback capture and full integration

**Files:**
- Modify: `tests/utils/test_run_log_cache.py`
- Modify: `computer_use_test/utils/run_log_cache.py`

**Step 1: Write the failing test**

```python
def test_exception_hook_appends_traceback(tmp_path):
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/utils/test_run_log_cache.py -v`
Expected: FAIL because uncaught traceback text is not persisted yet.

**Step 3: Write minimal implementation**

```python
def _handle_uncaught_exception(...):
    traceback.print_exception(..., file=self._stream)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/utils/test_run_log_cache.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/utils/test_run_log_cache.py computer_use_test/utils/run_log_cache.py
git commit -m "test: cover raw traceback capture"
```

### Task 4: Final verification and push

**Files:**
- Modify: `computer_use_test/utils/run_log_cache.py`
- Modify: `computer_use_test/agent/turn_runner.py`
- Test: `tests/utils/test_run_log_cache.py`

**Step 1: Run focused verification**

Run: `pytest tests/utils/test_run_log_cache.py -v`
Expected: all tests pass.

**Step 2: Run broader regression check**

Run: `pytest tests/agent/test_turn_executor.py -q`
Expected: existing agent tests still pass.

**Step 3: Review diff**

Run: `git diff -- computer_use_test/agent/turn_runner.py computer_use_test/utils/run_log_cache.py tests/utils/test_run_log_cache.py`

**Step 4: Commit**

```bash
git add docs/plans/2026-03-17-turn-runner-raw-log-cache-design.md docs/plans/2026-03-17-turn-runner-raw-log-cache.md computer_use_test/agent/turn_runner.py computer_use_test/utils/run_log_cache.py computer_use_test/utils/__init__.py tests/utils/test_run_log_cache.py
git commit -m "feat: cache raw turn runner logs"
```

**Step 5: Push**

Run: `git push origin HEAD`
