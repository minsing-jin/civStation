import os

# Provider name → environment variable mapping
_ENV_KEY_MAPPING: dict[str, str] = {
    "claude": "ANTHROPIC_API_KEY",
    "gemini": "GENAI_API_KEY",
    "gpt": "OPENAI_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def load_env_variable(provider_name: str, default: str | None = None) -> str:
    """Load the API key for a VLM provider from environment variables.

    Args:
        provider_name: One of "claude", "gemini", "gpt", "openai".
        default: Fallback value when the env var is unset. If *None*
                 (the default) and the key is missing, a ``ValueError``
                 is raised.

    Returns:
        The API key string.

    Raises:
        ValueError: If *provider_name* is unknown or the env var is
                    not set and no *default* was given.
    """
    env_var_name = _ENV_KEY_MAPPING.get(provider_name)
    if env_var_name is None:
        available = ", ".join(sorted(_ENV_KEY_MAPPING))
        raise ValueError(f"Unknown provider '{provider_name}'. Available providers: {available}")

    api_key = os.getenv(env_var_name, default)

    if not api_key:
        raise ValueError(
            f"API key not found for provider '{provider_name}'. "
            f"Please set the environment variable '{env_var_name}' in your .env file or system environment."
        )
    return api_key
