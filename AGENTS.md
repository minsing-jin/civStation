# AGENTS.md

Project-specific instructions for Codex in this repository.

## Debugging Workflow

- When the user asks about a bug, failure, regression, runtime error, or something "not working" in this project, read `./.tmp/civStation/turn_runner_latest.log` first before diagnosing.
- This log path matches the default cache produced by `civStation.utils.run_log_cache.get_run_log_cache_path()`.
- If the log file is missing, empty, or clearly stale for the current issue, say that explicitly and then continue with code inspection, tests, or other debugging steps.
- This rule applies only to this repository.
