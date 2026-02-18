"""
DebugOptions — Flags that control which debug logging is enabled.

Pass a single DebugOptions instance to run_one_turn / run_multi_turn
instead of many individual bool parameters.

Usage::

    # Programmatic
    from computer_use_test.utils.debug import DebugOptions

    opts = DebugOptions(log_context=True, validate_turns=True)
    run_multi_turn(..., debug_options=opts)

    # From CLI string (e.g. --debug context,turns or --debug all)
    opts = DebugOptions.from_str("context,turns")
    opts = DebugOptions.from_str("all")
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class DebugOptions:
    """Flags that toggle individual debug/logging features.

    Attributes:
        log_context: Log the full context string injected into each
                     primitive prompt (strategy + game state).
        validate_turns: Run TurnValidator checks on every micro-turn
                        to verify turn-number consistency from the Router.
    """

    log_context: bool = False
    validate_turns: bool = False

    # ---------------------------------------------------------------------------
    # Factory helpers
    # ---------------------------------------------------------------------------

    @classmethod
    def from_str(cls, spec: str) -> DebugOptions:
        """
        Build a DebugOptions from a comma-separated string of feature names.

        Special value ``"all"`` enables every flag.

        Examples::

            DebugOptions.from_str("context,turns")  # enable two features
            DebugOptions.from_str("all")             # enable everything
            DebugOptions.from_str("")                # nothing enabled
        """
        if not spec:
            return cls()

        tokens = {t.strip().lower() for t in spec.split(",") if t.strip()}

        if "all" in tokens:
            return cls(**{f.name: True for f in fields(cls)})

        # Map short aliases → field names
        _ALIASES: dict[str, str] = {
            "context": "log_context",
            "turns": "validate_turns",
            "turn": "validate_turns",
        }

        kwargs: dict[str, bool] = {}
        valid_names = {f.name for f in fields(cls)}
        for token in tokens:
            field_name = _ALIASES.get(token, token)
            if field_name in valid_names:
                kwargs[field_name] = True

        return cls(**kwargs)

    @classmethod
    def none(cls) -> DebugOptions:
        """Return an instance with all features disabled (default)."""
        return cls()

    @classmethod
    def all(cls) -> DebugOptions:
        """Return an instance with all features enabled."""
        return cls(**{f.name: True for f in fields(cls)})

    def any_enabled(self) -> bool:
        """Return True if at least one debug feature is enabled."""
        return any(getattr(self, f.name) for f in fields(self))

    def __str__(self) -> str:
        enabled = [f.name for f in fields(self) if getattr(self, f.name)]
        return f"DebugOptions({', '.join(enabled) or 'none'})"
