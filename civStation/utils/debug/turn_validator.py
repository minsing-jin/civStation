"""
Turn Validator — Checks consistency of router-observed turn numbers.

Validates that:
- Turn numbers increase monotonically (no backward jumps).
- Turn advances are sequential (no unexpected jumps of > 1).
- Macro-turn boundaries align with observed in-game turns.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SEP = "─" * 60


class TurnValidator:
    """
    Validates router-observed in-game turn numbers and emits diagnostics.

    Usage::

        validator = TurnValidator()

        # Called once per micro-turn (primitive action):
        validator.validate(
            observed_turn=router_result.observed_turn,
            is_new_turn=router_result.is_new_turn,
            micro_turn=turn_number,
            macro_turn=macro_turn_manager.macro_turn_number,
        )
    """

    def __init__(self) -> None:
        self._first_observed: int | None = None
        self._last_observed: int | None = None
        self._undetected_streak: int = 0  # consecutive None detections

    def validate(
        self,
        observed_turn: int | None,
        is_new_turn: bool,
        micro_turn: int,
        macro_turn: int,
    ) -> None:
        """
        Validate the router-observed turn number and emit diagnostic logs.

        Args:
            observed_turn: Turn number read from the screenshot (None if undetected).
            is_new_turn: True when the router detected an in-game turn advance.
            micro_turn: Current agent micro-turn index (action counter).
            macro_turn: Current macro-turn counter from MacroTurnManager.
        """
        logger.info(_SEP)
        screen = observed_turn if observed_turn is not None else "N/A"
        logger.info(
            f"[TurnValidator] micro={micro_turn} | macro_turn=#{macro_turn}"
            f" | screen_turn={screen} | new_turn_flag={is_new_turn}"
        )

        # --- Case: turn number not detected ---
        if observed_turn is None:
            self._undetected_streak += 1
            if self._undetected_streak >= 3:
                streak = self._undetected_streak
                logger.warning(
                    f"[TurnValidator] ⚠ Turn number undetected for {streak} consecutive"
                    " micro-turns. Router may be struggling to read the top-right HUD."
                )
            logger.info(_SEP)
            return

        self._undetected_streak = 0

        # --- First observation ---
        if self._first_observed is None:
            self._first_observed = observed_turn
            logger.info(f"[TurnValidator] ✓ First game turn observed: {observed_turn}")
            self._last_observed = observed_turn
            logger.info(_SEP)
            return

        # --- Consistency check ---
        diff = observed_turn - self._last_observed  # type: ignore[operator]

        if diff == 0:
            logger.debug(f"[TurnValidator] Same game turn ({observed_turn}), no advance.")

        elif diff == 1:
            logger.info(f"[TurnValidator] ✓ Turn advanced normally: {self._last_observed} → {observed_turn}")

        elif diff > 1:
            logger.warning(
                f"[TurnValidator] ⚠ Turn JUMP: {self._last_observed} → {observed_turn}"
                f" (Δ={diff}). Possible missed 'Next Turn' detection or save-loaded."
            )

        else:  # diff < 0
            logger.warning(
                f"[TurnValidator] ⚠ Turn BACKWARDS: {self._last_observed} → {observed_turn}."
                " Possible OCR mis-read or game reloaded."
            )

        # --- Macro-turn alignment ---
        if is_new_turn:
            logger.info(
                f"[TurnValidator] 🔔 Macro-turn boundary: game turn {self._last_observed}"
                f" → {observed_turn}. Now on macro-turn #{macro_turn}."
            )

        self._last_observed = observed_turn
        logger.info(_SEP)

    @property
    def last_observed_turn(self) -> int | None:
        return self._last_observed

    @property
    def first_observed_turn(self) -> int | None:
        return self._first_observed
