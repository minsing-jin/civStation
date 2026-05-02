"""Tests for shared private civ6-mcp payload parsing helpers."""

from __future__ import annotations

from civStation.agent.modules.backend.civ6_mcp._payload import select_payload_body


def test_select_payload_body_preserves_mcp_result_body_precedence() -> None:
    payload = {
        "content": [{"type": "text", "text": "Units:\n- Builder at (3, 4)"}],
        "structured_content": {"units": [{"type": "Builder"}]},
        "text": "fallback text",
    }

    selected = select_payload_body(payload)

    assert selected.value == "Units:\n- Builder at (3, 4)"
    assert selected.source == "content"


def test_select_payload_body_keeps_empty_content_text_from_falling_through() -> None:
    payload = {
        "content": [{"type": "text", "text": ""}],
        "structured_content": {"turn": 120},
    }

    selected = select_payload_body(payload)

    assert selected.value == ""
    assert selected.source == "content"


def test_select_payload_body_renders_structured_payloads_and_marks_passthrough() -> None:
    structured = {"turn_number": "118", "era": "Future Era"}

    assert select_payload_body({"structuredContent": structured}).value == structured

    selected = select_payload_body("raw text")
    assert selected.value == "raw text"
    assert selected.source is None
