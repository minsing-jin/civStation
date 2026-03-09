"""
Strategy Updater — Background worker that keeps the strategy up-to-date.

Runs in a daemon thread, following the same pattern as ``ContextUpdater``.
The main pipeline never blocks on strategy VLM calls; instead it reads
the latest strategy from ``ContextManager.get_strategy_string()`` which
returns immediately.

Triggers:
    INITIAL        — First turn, generate initial strategy.
    NEW_GAME_TURN  — Router detected an in-game turn increase.
    HITL_CHANGE    — User requested a strategy change (highest priority).
    PERIODIC       — Every N game-turns, re-evaluate strategy.

Priority rule:
    A pending HITL_CHANGE request is never replaced by a non-HITL request.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from computer_use_test.agent.modules.context.context_manager import ContextManager
    from computer_use_test.agent.modules.strategy.strategy_planner import StrategyPlanner

logger = logging.getLogger(__name__)


class StrategyTrigger(Enum):
    INITIAL = auto()
    NEW_GAME_TURN = auto()
    HITL_CHANGE = auto()
    PERIODIC = auto()


@dataclass
class StrategyRequest:
    trigger: StrategyTrigger
    human_input: str | None = None


class StrategyUpdater:
    """Background worker that asynchronously updates the game strategy.

    Usage::

        updater = StrategyUpdater(ctx, planner)
        updater.start()

        # Submit initial strategy (non-blocking)
        updater.submit(StrategyRequest(StrategyTrigger.INITIAL, human_input="과학 승리"))

        # In the main loop, after routing:
        if router_result.is_new_turn:
            updater.submit(StrategyRequest(StrategyTrigger.NEW_GAME_TURN))
            updater.submit_if_periodic_due()

        # On shutdown:
        updater.stop()
    """

    def __init__(
        self,
        context_manager: ContextManager,
        planner: StrategyPlanner,
        periodic_interval: int = 5,
    ) -> None:
        self._ctx = context_manager
        self._planner = planner
        self._periodic_interval = periodic_interval

        # Single-slot mailbox (latest-wins, with HITL priority protection)
        self._pending: StrategyRequest | None = None
        self._cond = threading.Condition()
        self._stopped = False
        self._thread: threading.Thread | None = None

        # Tracks how many NEW_GAME_TURN triggers have been seen
        self._game_turn_counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background strategy thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name="strategy-updater", daemon=True)
        self._thread.start()
        logger.info("StrategyUpdater background worker started")

    def stop(self) -> None:
        """Signal the worker to stop and wait for it to finish."""
        with self._cond:
            self._stopped = True
            self._cond.notify()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        logger.info("StrategyUpdater background worker stopped")

    def submit(self, request: StrategyRequest) -> None:
        """Submit a strategy request (non-blocking).

        If a previous request has not been processed yet, the new request
        replaces it **unless** the pending request is HITL_CHANGE and the
        new one is not — HITL requests have priority protection.
        """
        with self._cond:
            if (
                self._pending is not None
                and self._pending.trigger == StrategyTrigger.HITL_CHANGE
                and request.trigger != StrategyTrigger.HITL_CHANGE
            ):
                logger.debug("Skipping non-HITL request — pending HITL_CHANGE has priority")
                return
            self._pending = request
            self._cond.notify()

    def submit_if_periodic_due(self) -> None:
        """Increment the game-turn counter and submit a PERIODIC request if due."""
        self._game_turn_counter += 1
        if self._game_turn_counter % self._periodic_interval == 0:
            self.submit(StrategyRequest(StrategyTrigger.PERIODIC))

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Background loop: wait for a request, process it, update context."""
        while True:
            with self._cond:
                while self._pending is None and not self._stopped:
                    self._cond.wait()
                if self._stopped:
                    return
                request = self._pending
                self._pending = None

            try:
                self._process(request)
            except Exception as e:
                logger.warning(f"StrategyUpdater processing failed: {e}")

    def _process(self, request: StrategyRequest) -> None:
        """Dispatch the request to the appropriate planner method."""
        import time as _time

        trigger = request.trigger
        t0 = _time.monotonic()

        if trigger == StrategyTrigger.INITIAL:
            strategy = self._planner.generate_strategy(
                context=self._ctx,
                human_input=request.human_input,
            )
        elif trigger == StrategyTrigger.HITL_CHANGE:
            if request.human_input:
                strategy = self._planner.refine_strategy(request.human_input, self._ctx)
            else:
                strategy = self._planner.generate_strategy(context=self._ctx)
        elif trigger in (StrategyTrigger.NEW_GAME_TURN, StrategyTrigger.PERIODIC):
            current = self._ctx.high_level_context.current_strategy
            if current is None:
                strategy = self._planner.generate_strategy(context=self._ctx)
            else:
                reason = "새 게임턴 시작" if trigger == StrategyTrigger.NEW_GAME_TURN else "정기 재평가"
                strategy = self._planner.update_strategy(current, self._ctx, reason=reason)
        else:
            logger.warning(f"Unknown strategy trigger: {trigger}")
            return

        elapsed = _time.monotonic() - t0

        # Atomic reference swap — no lock needed for Python attribute assignment
        self._ctx.set_strategy(strategy)
        logger.info(f"Strategy updated via {trigger.name}: {strategy.victory_goal.value} victory")

        # Update Rich dashboard with full strategy details
        from computer_use_test.utils.rich_logger import RichLogger

        RichLogger.get().update_strategy(
            goal=strategy.victory_goal.value,
            trigger=trigger.name,
            phase=strategy.current_phase,
            text=strategy.text,
            directives=strategy.primitive_directives,
            duration=elapsed,
        )
