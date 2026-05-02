"""Shared private payload helpers for civ6-mcp backend normalization.

Equivalent duplicate candidates from ``state_parser`` and
``observation_schema``:
- MCP wrapper field access has identical mapping/attribute behavior and returns
  one field value or ``None``; keep that behavior in ``payload_value``.
- MCP ``content`` text-block extraction has identical block filtering behavior
  and returns ``list[str]``; keep that behavior in ``extract_text_blocks``.
- MCP wrapper body selection has identical precedence and returns
  ``PayloadBodySelection(value, source)``; keep that behavior in
  ``select_payload_body``.
- Parser text rendering should use ``render_payload_body`` when callers need
  both wrapper precedence and canonical state-parser text output.
- JSON text payload loading has identical string trimming and decode behavior;
  keep strict planner parsing and lenient overview probing on
  ``_load_json_payload`` with different container-prefix requirements.
- Planner tool-call entry validation shares envelope unwrapping, object-entry
  checks, and configurable tool-name alias checks.

Other similar parser/validator helpers are intentionally excluded here when
their output shapes differ, such as rendered text, booleans, normalized
observations, or typed planner action objects.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_VLM_ACTION_PLAN_ERROR = (
    "VLM/computer-use action-plan payload cannot run on the civ6-mcp backend; expected 'tool_calls'."
)


@dataclass(frozen=True)
class PayloadBodySelection:
    """Record the MCP payload body and wrapper field selected for rendering."""

    value: Any
    source: str | None = None


@dataclass(frozen=True)
class RenderedPayloadBody:
    """Record a selected MCP payload body together with its rendered text."""

    value: Any
    text: str
    source: str | None = None


def dump_model(value: Any, *, json_safe: bool = False) -> Any:
    """Convert model-like values into payload data, optionally JSON-safe."""
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif hasattr(value, "dict"):
        value = value.dict()

    if not json_safe or isinstance(value, dict | list | str | int | float | bool) or value is None:
        return value

    try:
        return json.loads(json.dumps(value))
    except TypeError:
        return str(value)


def _load_json_payload(payload: Any, *, require_json_container: bool = False) -> Any | None:
    """Load a JSON string or mapping payload for backend-local parsers.

    When ``require_json_container`` is true, non-container strings are treated
    as heuristic prose and return ``None`` without attempting fallback parsing.
    Malformed JSON that passes the prefix check still raises
    ``json.JSONDecodeError`` so callers can preserve their existing strict or
    logged lenient behavior.
    """
    if isinstance(payload, dict):
        return payload
    if not isinstance(payload, str):
        return None

    stripped = payload.strip()
    if require_json_container and (not stripped or stripped[0] not in "{["):
        return None
    return json.loads(payload)


def extract_text_blocks(content: list[Any]) -> list[str]:
    """Extract renderable text from MCP content block objects or mappings."""
    text_parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            text_parts.append(text)
            continue
        if isinstance(block, Mapping):
            dict_text = block.get("text")
            if isinstance(dict_text, str):
                text_parts.append(dict_text)
    return text_parts


def payload_value(payload: Any, key: str) -> Any | None:
    """Read a field from mapping-shaped or attribute-shaped payload values."""
    if isinstance(payload, Mapping):
        return payload.get(key)
    return getattr(payload, key, None)


def select_payload_body(payload: Any) -> PayloadBodySelection:
    """Select the parser-preferred MCP payload body by wrapper precedence."""
    content = payload_value(payload, "content")
    if isinstance(content, list):
        text_blocks = extract_text_blocks(content)
        if text_blocks:
            return PayloadBodySelection("\n".join(text_blocks), "content")

    content_blocks = payload_value(payload, "content_blocks")
    if isinstance(content_blocks, list | tuple):
        text_blocks = [str(block) for block in content_blocks if str(block).strip()]
        if text_blocks:
            return PayloadBodySelection("\n".join(text_blocks), "content_blocks")

    text = payload_value(payload, "text")
    if isinstance(text, str) and text.strip():
        return PayloadBodySelection(text, "text")

    for key in ("structuredContent", "structured_content"):
        value = payload_value(payload, key)
        if value is not None:
            return PayloadBodySelection(dump_model(value), key)

    return PayloadBodySelection(payload)


def _render_payload_text(payload: Any) -> str:
    """Render an already-selected MCP payload body into stable parser text.

    This intentionally does not unwrap result objects or choose between
    ``content``/``text``/structured fields; callers must use
    ``select_payload_body`` first when they need wrapper precedence.
    """
    if isinstance(payload, str):
        return payload
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")
    if payload is None:
        return ""
    if isinstance(payload, list | tuple) and all(isinstance(item, str | bytes) for item in payload):
        return "\n".join(_render_payload_text(item) for item in payload)
    payload = dump_model(payload)
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(payload)


def render_payload_body(payload: Any) -> RenderedPayloadBody:
    """Select and render the parser-preferred body from an MCP-style payload."""
    selected = select_payload_body(payload)
    return RenderedPayloadBody(
        value=selected.value,
        text=_render_payload_text(selected.value),
        source=selected.source,
    )


def planner_tool_call_items(payload: Any, *, reject_vlm_actions: bool = False) -> list[Any]:
    """Extract raw civ6-mcp tool-call items from a planner envelope or list."""
    if isinstance(payload, dict) and "tool_calls" in payload:
        items = payload["tool_calls"]
    elif reject_vlm_actions and isinstance(payload, dict) and "actions" in payload:
        raise ValueError(_VLM_ACTION_PLAN_ERROR)
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError(f"Unexpected planner payload shape: {type(payload).__name__}")

    if not isinstance(items, list):
        raise ValueError("Planner payload 'tool_calls' must be a list.")
    return list(items)


def planner_tool_call_dicts(
    payload: Any,
    *,
    reject_vlm_actions: bool = False,
    tool_name_keys: tuple[str, ...] = ("tool", "name"),
) -> list[dict[str, Any]]:
    """Extract planner tool-call dictionaries with caller-specific name aliases."""
    items = planner_tool_call_items(payload, reject_vlm_actions=reject_vlm_actions)
    parsed: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Tool call entry must be an object, got {type(item).__name__}")
        if tool_name_keys and not any(item.get(key) for key in tool_name_keys):
            raise ValueError(f"Tool call missing 'tool' name: {item!r}")
        parsed.append(item)
    return parsed


def payload_has_body(payload: Any) -> bool:
    """Return whether a raw MCP-style payload contains an observable body."""
    if payload is None:
        return False
    if isinstance(payload, str | bytes):
        return bool(payload.strip())

    selected = select_payload_body(payload)
    if selected.source in {"content_blocks", "text", "structuredContent", "structured_content"}:
        return True
    if selected.source == "content" and not isinstance(payload, Mapping):
        return payload_has_body(selected.value)

    looks_like_result = False
    text = payload_value(payload, "text")
    looks_like_result = looks_like_result or hasattr(payload, "text")
    if isinstance(text, str) and text.strip():
        return True

    content_blocks = payload_value(payload, "content_blocks")
    looks_like_result = looks_like_result or hasattr(payload, "content_blocks")
    if isinstance(content_blocks, list | tuple) and any(str(block).strip() for block in content_blocks):
        return True

    content = payload_value(payload, "content")
    looks_like_result = looks_like_result or hasattr(payload, "content")
    if isinstance(content, list | tuple) and any(block.strip() for block in extract_text_blocks(list(content))):
        return True

    for structured_name in ("structured_content", "structuredContent"):
        structured = payload_value(payload, structured_name)
        looks_like_result = looks_like_result or hasattr(payload, structured_name)
        if structured is not None:
            return True

    if isinstance(payload, Mapping):
        return any(payload_has_body(value) for value in payload.values())
    if looks_like_result:
        return False
    return True
