"""
civStation -- Computer-use action evaluation framework for game environments.
"""

__version__ = "0.0.1"

from civStation.utils.llm_provider import create_provider, get_available_providers

__all__ = ["__version__", "create_provider", "get_available_providers"]
