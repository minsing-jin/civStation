"""Utilities for caching the latest raw turn-runner log to a temp file."""

from __future__ import annotations

import logging
import sys
import tempfile
import traceback
from pathlib import Path
from types import TracebackType

_CACHE_DIRNAME = "computer_use_test"
_CACHE_FILENAME = "turn_runner_latest.log"
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def get_run_log_cache_path(base_dir: Path | str | None = None) -> Path:
    """Return the deterministic cache path for the latest turn-runner log."""
    root = Path(base_dir) if base_dir is not None else Path(tempfile.gettempdir()) / _CACHE_DIRNAME
    return root / _CACHE_FILENAME


class RunLogSession:
    """Owns one root file handler and exception hook for the current run."""

    def __init__(self, path: Path):
        self.path = path
        self._root_logger = logging.getLogger()
        self._previous_hook = sys.excepthook
        self._handler = logging.FileHandler(self.path, mode="w", encoding="utf-8")
        self._handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        self._handler.setLevel(logging.NOTSET)
        self._closed = False

        self._root_logger.addHandler(self._handler)
        sys.excepthook = self._handle_uncaught_exception

    def _handle_uncaught_exception(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType | None,
    ) -> None:
        stream = self._handler.stream
        if stream is not None:
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=stream)
            stream.flush()

        previous_hook = self._previous_hook
        if previous_hook is not None:
            previous_hook(exc_type, exc_value, exc_traceback)

    def close(self) -> None:
        """Detach the file handler and restore the prior exception hook."""
        if self._closed:
            return

        sys.excepthook = self._previous_hook
        self._root_logger.removeHandler(self._handler)
        self._handler.close()
        self._closed = True


def start_run_log_session(base_dir: Path | str | None = None) -> RunLogSession:
    """Start capturing the latest raw turn-runner log in a temp cache file."""
    path = get_run_log_cache_path(base_dir=base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    return RunLogSession(path=path)
