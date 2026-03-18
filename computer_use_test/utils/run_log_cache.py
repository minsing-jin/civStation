"""Utilities for caching the latest raw turn-runner log to a temp file."""

from __future__ import annotations

import logging
import re
import sys
import tempfile
import threading
import traceback
from pathlib import Path
from types import TracebackType

_CACHE_DIRNAME = "computer_use_test"
_CACHE_FILENAME = "turn_runner_latest.log"
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _sanitize_terminal_text(text: str) -> str:
    """Normalize terminal control output before mirroring it to the log file."""
    return _ANSI_ESCAPE_RE.sub("", text).replace("\r", "\n")


class _TeeBinaryStream:
    """Mirror binary writes to the original stream and the run-log file."""

    def __init__(self, primary, mirror, lock: threading.RLock):
        self._primary = primary
        self._mirror = mirror
        self._lock = lock

    def write(self, data: bytes) -> int:
        with self._lock:
            written = self._primary.write(data)
            self._mirror.write(data)
            self._mirror.flush()
        return written if written is not None else len(data)

    def flush(self) -> None:
        with self._lock:
            self._primary.flush()
            self._mirror.flush()

    def close(self) -> None:
        self.flush()

    def __getattr__(self, name: str):
        return getattr(self._primary, name)


class _TeeTextStream:
    """Mirror text writes to the original stream and the run-log file."""

    def __init__(self, primary, mirror, lock: threading.RLock):
        self._primary = primary
        self._mirror = mirror
        self._lock = lock

        primary_buffer = getattr(primary, "buffer", None)
        mirror_buffer = getattr(mirror, "buffer", None)
        self.buffer = (
            _TeeBinaryStream(primary_buffer, mirror_buffer, lock)
            if primary_buffer is not None and mirror_buffer is not None
            else primary_buffer
        )

    def write(self, data: str) -> int:
        text = str(data)
        with self._lock:
            written = self._primary.write(text)
            mirrored = _sanitize_terminal_text(text)
            if mirrored:
                self._mirror.write(mirrored)
                self._mirror.flush()
        return written if written is not None else len(text)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        with self._lock:
            self._primary.flush()
            self._mirror.flush()

    def close(self) -> None:
        self.flush()

    def __getattr__(self, name: str):
        return getattr(self._primary, name)


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
        self._previous_stdout = sys.stdout
        self._previous_stderr = sys.stderr
        self._handler = logging.FileHandler(self.path, mode="w", encoding="utf-8")
        self._handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        self._handler.setLevel(logging.NOTSET)
        self._lock = threading.RLock()
        self._closed = False

        self._root_logger.addHandler(self._handler)
        stream = self._handler.stream
        if stream is None:
            raise RuntimeError("Run log file handler did not open a writable stream")
        sys.stdout = _TeeTextStream(self._previous_stdout, stream, self._lock)
        sys.stderr = _TeeTextStream(self._previous_stderr, stream, self._lock)
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

        sys.stdout = self._previous_stdout
        sys.stderr = self._previous_stderr
        sys.excepthook = self._previous_hook
        self._root_logger.removeHandler(self._handler)
        self._handler.close()
        self._closed = True


def start_run_log_session(base_dir: Path | str | None = None) -> RunLogSession:
    """Start capturing the latest raw turn-runner log in a temp cache file."""
    path = get_run_log_cache_path(base_dir=base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    return RunLogSession(path=path)
