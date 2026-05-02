"""
VLM Provider package for multi-provider support.

This package provides a unified interface for different Vision-Language Model providers:
- Claude (Anthropic)
- Gemini (Google)
- GPT (OpenAI)
- Mock (for testing)

Example usage:
    >>> from civStation.utils.provider import create_provider
    >>> provider = create_provider("claude", api_key="your-api-key")
    >>> response = provider.call_vlm("Analyze this image", image_path="screenshot.png")
    >>> plan = provider.parse_to_agent_plan(response, "unit_ops_primitive")
"""

from civStation.utils.llm_provider.base import (
    BaseVLMProvider,
    MockVLMProvider,
    VLMResponse,
)
from civStation.utils.llm_provider.parser import (
    AgentAction,
    parse_action_json,
    parse_to_agent_plan,
    strip_markdown,
    validate_action,
)

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

_PROVIDER_CLASS_PATHS = {
    "ClaudeVLMProvider": "civStation.utils.llm_provider.claude",
    "GeminiVLMProvider": "civStation.utils.llm_provider.gemini",
    "GPTVLMProvider": "civStation.utils.llm_provider.gpt",
    "OpenAIComputerVLMProvider": "civStation.utils.llm_provider.openai_computer",
    "AnthropicComputerVLMProvider": "civStation.utils.llm_provider.anthropic_computer",
}

_PROVIDER_DEFAULT_MODELS = {
    "ClaudeVLMProvider": "claude-4-5-sonnet-20241022",
    "GeminiVLMProvider": "gemini-3-flash-preview",
    "GPTVLMProvider": "gpt-4o",
    "OpenAIComputerVLMProvider": "gpt-5.4",
    "AnthropicComputerVLMProvider": "claude-4-5-sonnet-20241022",
}

_PROVIDER_NAMES = {
    "claude": "ClaudeVLMProvider",
    "gemini": "GeminiVLMProvider",
    "gpt": "GPTVLMProvider",
    "openai": "GPTVLMProvider",
    "openai-computer": "OpenAIComputerVLMProvider",
    "gpt-computer": "OpenAIComputerVLMProvider",
    "anthropic-computer": "AnthropicComputerVLMProvider",
    "claude-computer": "AnthropicComputerVLMProvider",
}


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _load_provider_class(class_name: str):
    import importlib

    module_name = _PROVIDER_CLASS_PATHS[class_name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _get_provider_default_model(class_name: str) -> str:
    try:
        return _load_provider_class(class_name).DEFAULT_MODEL
    except ImportError:
        return _PROVIDER_DEFAULT_MODELS[class_name]


def __getattr__(name: str):
    if name in _PROVIDER_CLASS_PATHS:
        provider_class = _load_provider_class(name)
        globals()[name] = provider_class
        return provider_class
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_load_dotenv_if_available()


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

    if provider_name not in _PROVIDER_NAMES and provider_name != "mock":
        available = ", ".join([*_PROVIDER_NAMES, "mock"])
        raise ValueError(f"Unknown provider: {provider_name}. Available providers: {available}")

    # MockVLMProvider doesn't need api_key
    if provider_name == "mock":
        return MockVLMProvider(model=model)

    from civStation.utils.utils import load_env_variable

    api_key = api_key or load_env_variable(provider_name)
    provider_class = _load_provider_class(_PROVIDER_NAMES[provider_name])
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
        "claude": _get_provider_default_model("ClaudeVLMProvider"),
        "gemini": _get_provider_default_model("GeminiVLMProvider"),
        "gpt": _get_provider_default_model("GPTVLMProvider"),
        "openai": _get_provider_default_model("GPTVLMProvider"),
        "openai-computer": _get_provider_default_model("OpenAIComputerVLMProvider"),
        "anthropic-computer": _get_provider_default_model("AnthropicComputerVLMProvider"),
        "mock": "mock-vlm",
    }
