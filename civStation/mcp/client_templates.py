from __future__ import annotations

from pathlib import Path

_CLIENT_TEMPLATE_ROOT = Path(__file__).resolve().parents[2] / "templates" / "clients"

_CLIENT_TEMPLATES = {
    "codex": _CLIENT_TEMPLATE_ROOT / "codex" / "config.toml",
    "claude-code": _CLIENT_TEMPLATE_ROOT / "claude-code" / "project.mcp.json",
}

_DEFAULT_OUTPUTS = {
    "codex": ".codex/config.toml",
    "claude-code": ".mcp.json",
}


def list_supported_clients() -> list[str]:
    return sorted(_CLIENT_TEMPLATES)


def get_client_template_path(client: str) -> Path:
    normalized = client.strip().lower()
    if normalized not in _CLIENT_TEMPLATES:
        supported = ", ".join(list_supported_clients())
        raise ValueError(f"Unsupported client: {client!r}. Supported clients: {supported}")
    return _CLIENT_TEMPLATES[normalized]


def render_client_template(client: str) -> str:
    return get_client_template_path(client).read_text(encoding="utf-8")


def default_output_path(client: str) -> str:
    normalized = client.strip().lower()
    if normalized not in _DEFAULT_OUTPUTS:
        supported = ", ".join(list_supported_clients())
        raise ValueError(f"Unsupported client: {client!r}. Supported clients: {supported}")
    return _DEFAULT_OUTPUTS[normalized]
