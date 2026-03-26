"""Utilities for caching the latest raw turn-runner log inside the project."""

from __future__ import annotations

import logging
import re
import sys
import threading
import traceback
from pathlib import Path
from types import TracebackType

from civStation.utils.project_runtime import get_project_runtime_root

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
            written = self._write_primary(data)
            self._write_mirror(data)
            self._flush_mirror()
        return written if written is not None else len(data)

    def flush(self) -> None:
        with self._lock:
            self._flush_primary()
            self._flush_mirror()

    def close(self) -> None:
        self.flush()

    def _write_primary(self, data: bytes) -> int | None:
        if self._primary is None or getattr(self._primary, "closed", False):
            return None
        try:
            return self._primary.write(data)
        except (ValueError, OSError):
            return None

    def _flush_primary(self) -> None:
        if self._primary is None or getattr(self._primary, "closed", False):
            return
        try:
            self._primary.flush()
        except (ValueError, OSError):
            return

    def _write_mirror(self, data: bytes) -> None:
        if self._mirror is None or getattr(self._mirror, "closed", False):
            return
        try:
            self._mirror.write(data)
        except (ValueError, OSError):
            return

    def _flush_mirror(self) -> None:
        if self._mirror is None or getattr(self._mirror, "closed", False):
            return
        try:
            self._mirror.flush()
        except (ValueError, OSError):
            return

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
            written = self._write_primary(text)
            mirrored = _sanitize_terminal_text(text)
            if mirrored:
                self._write_mirror(mirrored)
                self._flush_mirror()
        return written if written is not None else len(text)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        with self._lock:
            self._flush_primary()
            self._flush_mirror()

    def close(self) -> None:
        self.flush()

    def _write_primary(self, text: str) -> int | None:
        if self._primary is None or getattr(self._primary, "closed", False):
            return None
        try:
            return self._primary.write(text)
        except (ValueError, OSError):
            return None

    def _flush_primary(self) -> None:
        if self._primary is None or getattr(self._primary, "closed", False):
            return
        try:
            self._primary.flush()
        except (ValueError, OSError):
            return

    def _write_mirror(self, text: str) -> None:
        if self._mirror is None or getattr(self._mirror, "closed", False):
            return
        try:
            self._mirror.write(text)
        except (ValueError, OSError):
            return

    def _flush_mirror(self) -> None:
        if self._mirror is None or getattr(self._mirror, "closed", False):
            return
        try:
            self._mirror.flush()
        except (ValueError, OSError):
            return

    def __getattr__(self, name: str):
        return getattr(self._primary, name)


def get_run_log_cache_path(base_dir: Path | str | None = None) -> Path:
    """Return the deterministic cache path for the latest turn-runner log."""
    root = get_project_runtime_root(base_dir=base_dir)
    return root / _CACHE_FILENAME


class RunLogSession:
    """Owns one root file handler and exception hook for the current run."""

    def __init__(self, path: Path):
        self.path = path
        self._root_logger = logging.getLogger()
        self._previous_hook = sys.excepthook
        self._previous_stdout = sys.stdout
        self._previous_stderr = sys.stderr
        self.path.write_text("", encoding="utf-8")
        self._handler = logging.FileHandler(self.path, mode="a", encoding="utf-8")
        self._handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        self._handler.setLevel(logging.NOTSET)
        self._lock = threading.RLock()
        self._closed = False
        self._mirror_stream = self.path.open("a", encoding="utf-8")

        self._root_logger.addHandler(self._handler)
        sys.stdout = _TeeTextStream(self._previous_stdout, self._mirror_stream, self._lock)
        sys.stderr = _TeeTextStream(self._previous_stderr, self._mirror_stream, self._lock)
        sys.excepthook = self._handle_uncaught_exception

    def _handle_uncaught_exception(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType | None,
    ) -> None:
        stream = self._mirror_stream
        if not getattr(stream, "closed", False):
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=stream)
            stream.flush()

        previous_hook = self._previous_hook
        if previous_hook is not None:
            previous_hook(exc_type, exc_value, exc_traceback)

    def close(self) -> None:
        """Detach the file handler and restore the prior exception hook."""
        with self._lock:
            if self._closed:
                return

            sys.stdout = self._previous_stdout
            sys.stderr = self._previous_stderr
            sys.excepthook = self._previous_hook
            self._root_logger.removeHandler(self._handler)
            self._handler.close()
            self._mirror_stream.close()
            self._closed = True


def start_run_log_session(base_dir: Path | str | None = None) -> RunLogSession:
    """Start capturing the latest raw turn-runner log in the project runtime cache."""
    path = get_run_log_cache_path(base_dir=base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    return RunLogSession(path=path)
