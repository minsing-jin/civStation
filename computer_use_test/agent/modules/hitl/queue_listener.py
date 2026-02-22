"""
Queue Listener — Daemon thread that monitors input sources and pushes directives.

Reuses HITLInputManager (text/voice/chatapp providers) to listen for user input
and converts it into Directive objects pushed into a CommandQueue.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from computer_use_test.agent.modules.hitl.command_queue import CommandQueue, Directive, DirectiveType

if TYPE_CHECKING:
    from computer_use_test.agent.modules.hitl.input_manager import HITLInputManager

logger = logging.getLogger(__name__)

# Keywords that map to directive types
_STOP_KEYWORDS = {"stop", "quit", "exit", "중지", "종료"}
_PAUSE_KEYWORDS = {"pause", "일시정지", "멈춰"}
_RESUME_KEYWORDS = {"resume", "start", "start agent", "계속", "재개", "시작", "에이전트 시작"}


class QueueListener:
    """
    Daemon thread that listens for user input and pushes directives to a CommandQueue.

    Uses HITLInputManager to support text, voice, and chatapp input sources.
    """

    def __init__(
        self,
        command_queue: CommandQueue,
        input_manager: HITLInputManager,
        prompt: str = "[HITL] 명령 입력 (전략 변경 / stop / pause): ",
    ) -> None:
        self._queue = command_queue
        self._input_manager = input_manager
        self._prompt = prompt
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        """Start the listener daemon thread."""
        if self._thread and self._thread.is_alive():
            return
        if not self._input_manager.is_available():
            logger.info("QueueListener not started: no active input provider")
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True, name="QueueListener")
        self._thread.start()
        logger.info(f"QueueListener started (mode={self._input_manager.get_effective_mode()})")

    def stop(self) -> None:
        """Signal the listener to stop."""
        self._running = False
        logger.info("QueueListener stop requested")

    def _listen_loop(self) -> None:
        """Main loop: block on input, parse, push directive."""
        while self._running:
            try:
                raw = self._input_manager.get_input(self._prompt)
                if not raw or not raw.strip():
                    continue
                directive = self._parse_input_to_directive(raw.strip())
                self._queue.push(directive)
            except EOFError:
                logger.info("QueueListener: stdin closed, stopping")
                break
            except Exception:
                logger.exception("QueueListener: error reading input")

    def _parse_input_to_directive(self, text: str) -> Directive:
        """Parse raw user input into a Directive."""
        lower = text.lower()
        source = self._input_manager.get_effective_mode()

        if lower in _STOP_KEYWORDS:
            return Directive(directive_type=DirectiveType.STOP, payload=text, source=source)
        if lower in _PAUSE_KEYWORDS:
            return Directive(directive_type=DirectiveType.PAUSE, payload=text, source=source)
        if lower in _RESUME_KEYWORDS:
            return Directive(directive_type=DirectiveType.RESUME, payload=text, source=source)

        # Default: treat as strategy change
        return Directive(directive_type=DirectiveType.CHANGE_STRATEGY, payload=text, source=source)
