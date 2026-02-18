"""
Context Logger — Logs the full context injected into primitive prompts.

Enabled via DebugOptions.log_context=True.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_THICK_SEP = "━" * 60


def log_context(
    primitive_name: str,
    strategy_string: str | None,
    context_string: str,
) -> None:
    """
    Log the full context that is about to be injected into a primitive prompt.

    Args:
        primitive_name: Name of the primitive being executed.
        strategy_string: High-level strategy text (may be None).
        context_string: Full context string from ContextManager.
    """
    logger.info(_THICK_SEP)
    logger.info(f"[DEBUG CONTEXT] primitive = {primitive_name}")
    logger.info("[DEBUG CONTEXT] ── strategy ──")
    for line in (strategy_string or "(전략 없음)").splitlines():
        logger.info(f"  {line}")
    logger.info("[DEBUG CONTEXT] ── context_string ──")
    for line in context_string.splitlines():
        logger.info(f"  {line}")
    logger.info(_THICK_SEP)
