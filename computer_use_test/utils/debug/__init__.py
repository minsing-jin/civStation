"""
Debug utilities for the Civilization VI agent.

All debug/logging helpers live here so they stay isolated from
production code.  Enable individual features via DebugOptions:

    opts = DebugOptions(log_context=True, validate_turns=True)
    run_multi_turn(..., debug_options=opts)

Or from the CLI:

    python -m ... --debug context,turns
    python -m ... --debug all
"""

from __future__ import annotations

from computer_use_test.utils.debug.context_logger import log_context
from computer_use_test.utils.debug.debug_options import DebugOptions
from computer_use_test.utils.debug.turn_validator import TurnValidator

__all__ = ["DebugOptions", "TurnValidator", "log_context"]
