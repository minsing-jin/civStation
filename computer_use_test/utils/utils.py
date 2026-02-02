import os


# Retrieve API key from environment variables
def load_env_variable(provider_name, default=None):
    env_key_mapping = {
        "claude": "ANTHROPIC_API_KEY",
        "gemini": "GENAI_API_KEY",
        "gpt": "OPENAI_API_KEY",
        "openai": "OPENAI_API_KEY",
    }

    # Retrieve API key from environment variables
    env_var_name = env_key_mapping.get(provider_name)
    api_key = os.getenv(env_var_name)

    if not api_key:
        raise ValueError(
            f"API key not found for provider '{provider_name}'. "
            f"Please set the environment variable '{env_var_name}' in your .env file or system environment."
        )
    return api_key
