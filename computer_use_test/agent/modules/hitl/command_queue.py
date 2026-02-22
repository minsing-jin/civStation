"""
Command Queue — Async directive buffer for HITL commands.

Allows user input (strategy changes, stop, pause, resume) to be
queued asynchronously and drained at primitive-level checkpoints.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class DirectiveType(str, Enum):
    """Types of HITL directives."""

    CHANGE_STRATEGY = "change_strategy"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    CUSTOM = "custom"
    PRIMITIVE_OVERRIDE = "primitive_override"


@dataclass
class Directive:
    """A single queued HITL directive."""

    directive_type: DirectiveType
    payload: str = ""
    source: str = "unknown"  # "stdin", "voice", "chatapp", "web_ui"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


class CommandQueue:
    """
    Thread-safe directive queue for async HITL input.

    Writers: QueueListener (daemon thread), FastAPI endpoint
    Readers: Main agent loop (drain at checkpoints), StateBridge (peek for UI)
    """

    def __init__(self, maxlen: int = 50) -> None:
        self._queue: deque[Directive] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._event = threading.Event()

    def push(self, directive: Directive) -> None:
        """Push a directive onto the queue (called from input threads)."""
        with self._lock:
            self._queue.append(directive)
            self._event.set()
        logger.info(f"Directive queued: {directive.directive_type.value} from {directive.source}")

    def drain(self) -> list[Directive]:
        """Pop all pending directives (called at main loop checkpoints)."""
        with self._lock:
            items = list(self._queue)
            self._queue.clear()
            self._event.clear()
        return items

    def peek(self) -> list[Directive]:
        """Read all pending directives without removing them (for UI display)."""
        with self._lock:
            return list(self._queue)

    def has_pending(self) -> bool:
        """Check if there are pending directives."""
        return self._event.is_set()

    @property
    def size(self) -> int:
        """Current number of queued directives."""
        with self._lock:
            return len(self._queue)

    def clear(self) -> None:
        """Remove all pending directives."""
        with self._lock:
            self._queue.clear()
            self._event.clear()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until a directive is available or timeout expires."""
        return self._event.wait(timeout=timeout)
