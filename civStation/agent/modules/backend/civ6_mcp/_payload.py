"""Shared private payload helpers for civ6-mcp backend normalization.

Inventory comparison note:
- ``state_parser`` unwraps MCP SDK-ish payloads into text/structured values.
- ``observation_schema`` first checks whether those same payload shapes contain
  an observation body before delegating to ``state_parser``.

Keep these near-duplicate parsing behaviors aligned here so public parser and
normalizer APIs can stay unchanged.
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
    """Represent a body value selected from an MCP-style payload wrapper."""

    value: Any
    source: str | None = None


def dump_model(value: Any, *, json_safe: bool = False) -> Any:
    """Convert model-like values into plain payload data."""
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


def extract_text_blocks(content: list[Any]) -> list[str]:
    """Extract text strings from MCP block objects or dict-shaped blocks."""
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
    """Select the first parser-preferred body from a raw MCP-style payload.

    The source order mirrors the historical state parser behavior: text
    ``content`` blocks first, then ``content_blocks``, direct ``text``, and
    finally structured payload fields. If no wrapper body is found, the original
    payload is returned with ``source`` set to ``None``.
    """
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


def planner_tool_call_items(payload: Any, *, reject_vlm_actions: bool = False) -> list[Any]:
    """Extract planner tool-call items from dict or bare-list payloads.

    Item-level validation stays with callers because executor and operation
    coercion intentionally accept slightly different tool-call aliases.
    """
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
