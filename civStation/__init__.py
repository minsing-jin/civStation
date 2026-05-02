"""
civStation -- Computer-use action evaluation framework for game environments.
"""

__version__ = "0.0.1"

from typing import Any


def create_provider(*args: Any, **kwargs: Any) -> Any:
    """Lazily proxy provider creation to avoid import-time provider dependencies."""
    from civStation.utils.llm_provider import create_provider as _create_provider

    return _create_provider(*args, **kwargs)


def get_available_providers() -> dict[str, str]:
    """Lazily proxy provider discovery to avoid import-time provider dependencies."""
    from civStation.utils.llm_provider import get_available_providers as _get_available_providers

    return _get_available_providers()


__all__ = ["__version__", "create_provider", "get_available_providers"]
