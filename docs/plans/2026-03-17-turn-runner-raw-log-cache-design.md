# Turn Runner Raw Log Cache Design

**Problem**

`python -m civStation.agent.turn_runner` currently favors Rich live output for operator visibility, but it does not preserve a plain-text execution log that external coding agents can inspect after a failure. When a run crashes, the useful context is spread across terminal state and traceback output.

**Goal**

Add an isolated utility that captures raw run logs for each `turn_runner` execution without changing the existing Rich UX. The log should be overwritten on every new run so only the latest execution is kept in a temporary cache file.

**Requirements**

- Preserve existing Rich live dashboard behavior.
- Capture standard Python logging records in plain text.
- Capture uncaught exception tracebacks in the same file.
- Overwrite the file at the start of every new `turn_runner` run.
- Keep the logic in a separate utility module so other code paths are unaffected unless they opt in.

**Options**

1. Root logger file handler only.
   This is simple, but uncaught exceptions written directly to `stderr` could be missed or fragmented.

2. File handler plus `sys.excepthook`.
   This captures both logging records and top-level unhandled tracebacks while leaving console behavior intact. It stays narrow because it only activates when `turn_runner` opts in.

3. Redirect global `stdout` and `stderr` into a file.
   This is broader than needed and risks changing console behavior or interfering with Rich live output.

**Recommendation**

Use option 2. It meets the debugging need without changing the terminal UI or broadening scope into unrelated streams.

**Design**

- Add `civStation/utils/run_log_cache.py`.
- Expose a small session object that:
  - resolves a temp-cache path for the latest run,
  - opens the file in write mode to guarantee overwrite semantics,
  - attaches a plain `logging.FileHandler` to the root logger,
  - installs a wrapper `sys.excepthook` that appends uncaught tracebacks,
  - restores the previous hook and removes the handler during cleanup.
- Initialize the session near the start of `turn_runner.main()` so startup failures are included.
- Close the session in `main()` cleanup regardless of success or failure.
- Keep the cache path deterministic for the latest run, for example under the system temp directory in a `civStation` subdirectory.

**Testing**

- Verify starting a session creates or truncates the target file.
- Verify root logger messages are written to the file.
- Verify a second session overwrites prior contents.
- Verify the custom exception hook writes traceback text and preserves the previous hook behavior contract.

**Non-Goals**

- Rotating or retaining historical log files.
- Capturing arbitrary subprocess output.
- Changing log levels or existing Rich dashboard rendering.
