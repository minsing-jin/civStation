"""
VLM Provider package for multi-provider support.

This package provides a unified interface for different Vision-Language Model providers:
- Claude (Anthropic)
- Gemini (Google)
- GPT (OpenAI)
- Mock (for testing)

Example usage:
    >>> from computer_use_test.utils.provider import create_provider
    >>> provider = create_provider("claude", api_key="your-api-key")
    >>> response = provider.call_vlm("Analyze this image", image_path="screenshot.png")
    >>> plan = provider.parse_to_agent_plan(response, "unit_ops_primitive")
"""

from dotenv import load_dotenv

from computer_use_test.utils.llm_provider.anthropic_computer import AnthropicComputerVLMProvider
from computer_use_test.utils.llm_provider.base import (
    BaseVLMProvider,
    MockVLMProvider,
    VLMResponse,
)
from computer_use_test.utils.llm_provider.claude import ClaudeVLMProvider
from computer_use_test.utils.llm_provider.gemini import GeminiVLMProvider
from computer_use_test.utils.llm_provider.gpt import GPTVLMProvider
from computer_use_test.utils.llm_provider.openai_computer import OpenAIComputerVLMProvider
from computer_use_test.utils.llm_provider.parser import (
    AgentAction,
    parse_action_json,
    parse_to_agent_plan,
    strip_markdown,
    validate_action,
)
from computer_use_test.utils.utils import load_env_variable

__all__ = [
    "AgentAction",
    "BaseVLMProvider",
    "VLMResponse",
    "ClaudeVLMProvider",
    "GeminiVLMProvider",
    "GPTVLMProvider",
    "OpenAIComputerVLMProvider",
    "AnthropicComputerVLMProvider",
    "MockVLMProvider",
    "create_provider",
    "get_available_providers",
    "parse_action_json",
    "parse_to_agent_plan",
    "strip_markdown",
    "validate_action",
]

load_dotenv()


def create_provider(
    provider_name: str,
    api_key: str = None,
    model: str = None,
) -> BaseVLMProvider:
    """
    Factory function to create a VLM provider by name.

    Args:
        provider_name: Name of provider ("claude", "gemini", "gpt", "mock")
        api_key: API key for the provider (optional, will try environment variables)
        model: Model identifier (optional, will use provider default)

    Returns:
        Initialized VLM provider

    Raises:
        ValueError: If provider name is not recognized

    Examples:
        >>> provider = create_provider("claude")
        >>> provider = create_provider("gpt", model="gpt-4o-mini")
        >>> provider = create_provider("mock")  # For testing
    """
    provider_name = provider_name.lower()

    providers = {
        "claude": ClaudeVLMProvider,
        "gemini": GeminiVLMProvider,
        "gpt": GPTVLMProvider,
        "openai": GPTVLMProvider,  # Alias
        "openai-computer": OpenAIComputerVLMProvider,
        "gpt-computer": OpenAIComputerVLMProvider,
        "anthropic-computer": AnthropicComputerVLMProvider,
        "claude-computer": AnthropicComputerVLMProvider,
        "mock": MockVLMProvider,
    }

    if provider_name not in providers:
        available = ", ".join(providers.keys())
        raise ValueError(f"Unknown provider: {provider_name}. Available providers: {available}")

    provider_class = providers[provider_name]
    api_key = api_key or load_env_variable(provider_name)

    # MockVLMProvider doesn't need api_key
    if provider_name == "mock":
        return provider_class(model=model)

    return provider_class(api_key=api_key, model=model)


def get_available_providers() -> dict[str, str]:
    """
    Get list of available providers and their default models.

    Returns:
        Dictionary mapping provider names to default model identifiers

    Example:
        >>> providers = get_available_providers()
        >>> print(providers)
        {'claude': 'claude-4-5-sonnet-20241022', 'gemini': 'gemini-3.0-flash-preview', ...}
    """
    return {
        "claude": ClaudeVLMProvider.DEFAULT_MODEL,
        "gemini": GeminiVLMProvider.DEFAULT_MODEL,
        "gpt": GPTVLMProvider.DEFAULT_MODEL,
        "openai": GPTVLMProvider.DEFAULT_MODEL,
        "openai-computer": OpenAIComputerVLMProvider.DEFAULT_MODEL,
        "anthropic-computer": AnthropicComputerVLMProvider.DEFAULT_MODEL,
        "mock": "mock-vlm",
    }
