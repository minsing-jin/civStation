"""Tests for shared private civ6-mcp payload parsing helpers."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any

from civStation.agent.modules.backend.civ6_mcp._payload import (
    _load_json_payload,
    _render_payload_text,
    extract_text_blocks,
    payload_has_body,
    payload_value,
    planner_tool_call_dicts,
    render_payload_body,
    select_payload_body,
)
from civStation.agent.modules.backend.civ6_mcp.executor import coerce_tool_calls
from civStation.agent.modules.backend.civ6_mcp.observation_schema import parse_observation_tool_response
from civStation.agent.modules.backend.civ6_mcp.operations import coerce_civ6_mcp_requests
from civStation.agent.modules.backend.civ6_mcp.response import normalize_mcp_tool_result
from civStation.agent.modules.backend.civ6_mcp.state_parser import parse_game_overview


class FakePayload:
    def __init__(
        self,
        *,
        content: list[object] | None = None,
        structured_content: object | None = None,
        text: str = "",
    ) -> None:
        self.content = content or []
        self.structured_content = structured_content
        self.text = text


class FakeTextBlock:
    def __init__(self, text: object) -> None:
        self.text = text


class FakeModel:
    def model_dump(self) -> dict[str, object]:
        return {"turn": 118, "unit": object()}


def test_payload_value_preserves_mapping_and_attribute_field_access() -> None:
    attr_payload = FakePayload(text="attribute text")

    assert payload_value({"text": "mapping text"}, "text") == "mapping text"
    assert payload_value(attr_payload, "text") == "attribute text"
    assert payload_value(attr_payload, "missing") is None


def test_extract_text_blocks_preserves_shared_mcp_content_filtering() -> None:
    content = [
        FakeTextBlock("attribute block"),
        {"text": "mapping block"},
        FakeTextBlock(123),
        {"text": 456},
        object(),
    ]

    assert extract_text_blocks(content) == ["attribute block", "mapping block"]


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


def test_render_payload_text_keeps_state_parser_body_rendering_contract() -> None:
    assert _render_payload_text("raw text") == "raw text"
    assert _render_payload_text(b"raw \xff bytes") == "raw \ufffd bytes"
    assert _render_payload_text(None) == ""
    assert _render_payload_text(["Units:", b"- Scout"]) == "Units:\n- Scout"
    assert _render_payload_text({"b": 2, "a": 1}) == '{"a": 1, "b": 2}'
    assert _render_payload_text(FakeModel()).startswith('{"turn": 118, "unit": "')


def test_render_payload_body_preserves_selection_and_rendering_contract() -> None:
    payload = {
        "content": [{"type": "text", "text": "Units:\n- Builder"}],
        "structured_content": {"units": [{"type": "Builder"}]},
    }

    rendered = render_payload_body(payload)

    assert rendered.value == "Units:\n- Builder"
    assert rendered.text == "Units:\n- Builder"
    assert rendered.source == "content"


def test_load_json_payload_supports_strict_and_container_only_parser_modes() -> None:
    payload = {"tool_calls": []}

    assert _load_json_payload(payload) is payload
    assert _load_json_payload(json.dumps(payload)) == payload
    assert _load_json_payload("Here is the plan: {}", require_json_container=True) is None

    try:
        _load_json_payload("Here is the plan: {}")
    except json.JSONDecodeError as exc:
        assert exc.msg == "Expecting value"
    else:  # pragma: no cover - defensive assertion for parser strictness
        raise AssertionError("strict JSON payload loading must reject prose-wrapped JSON")


def test_planner_tool_call_dicts_preserves_shared_entry_validation_errors() -> None:
    try:
        planner_tool_call_dicts({"tool_calls": ["get_game_overview"]})
    except ValueError as exc:
        assert str(exc) == "Tool call entry must be an object, got str"
    else:  # pragma: no cover - defensive assertion for parser strictness
        raise AssertionError("planner tool-call parser must reject non-object entries")

    try:
        planner_tool_call_dicts({"tool_calls": [{"arguments": {}}]})
    except ValueError as exc:
        assert str(exc) == "Tool call missing 'tool' name: {'arguments': {}}"
    else:  # pragma: no cover - defensive assertion for parser strictness
        raise AssertionError("planner tool-call parser must require a tool-name alias")


def test_planner_tool_call_dicts_preserves_caller_specific_tool_aliases() -> None:
    payload = {"tool_calls": [{"type": "choose_research", "tech": "WRITING"}]}

    try:
        planner_tool_call_dicts(payload)
    except ValueError as exc:
        assert "missing 'tool' name" in str(exc)
    else:  # pragma: no cover - defensive assertion for operations semantics
        raise AssertionError("default tool-call parser must reject action-type aliases")

    assert planner_tool_call_dicts(
        payload,
        tool_name_keys=("tool", "name", "type", "action_type", "action"),
    ) == [{"type": "choose_research", "tech": "WRITING"}]


def test_tool_call_envelope_paths_match_executor_and_operations_parsers() -> None:
    tool_calls = [
        {"tool": "get_units", "arguments": {}, "reasoning": "Refresh tactical unit state."},
        {
            "name": "set_research",
            "arguments": {"tech_or_civic": "WRITING"},
            "reasoning": "Unlock campuses faster.",
        },
    ]

    executor_from_envelope = _tool_call_snapshot(coerce_tool_calls({"tool_calls": tool_calls}))
    executor_from_list = _tool_call_snapshot(coerce_tool_calls(tool_calls))
    operations_from_envelope = _request_snapshot(coerce_civ6_mcp_requests({"tool_calls": tool_calls}))
    operations_from_list = _request_snapshot(coerce_civ6_mcp_requests(tool_calls))

    assert executor_from_envelope == executor_from_list
    assert operations_from_envelope == operations_from_list
    assert executor_from_envelope == operations_from_envelope


def test_tool_call_envelope_errors_match_executor_and_operations_parsers() -> None:
    payload = {"tool_calls": "get_units"}

    assert _value_error_message(coerce_tool_calls, payload) == "Planner payload 'tool_calls' must be a list."
    assert _value_error_message(coerce_civ6_mcp_requests, payload) == "Planner payload 'tool_calls' must be a list."


def test_shared_payload_selection_matches_state_observation_and_response_parsers() -> None:
    payload = FakePayload(
        content=[FakeTextBlock("Turn: 12\nEra: Ancient Era")],
        structured_content={"turn": 99, "era": "Future Era"},
        text="Turn: 44",
    )

    selected = select_payload_body(payload)
    overview = parse_game_overview(payload)
    observation = parse_observation_tool_response("get_game_overview", payload)
    response = normalize_mcp_tool_result("get_game_overview", {}, payload)

    assert selected.value == "Turn: 12\nEra: Ancient Era"
    assert selected.source == "content"
    assert payload_has_body(payload) is True
    assert overview.current_turn == 12
    assert overview.game_era == "Ancient"
    assert observation.bundle.overview.current_turn == 12
    assert response.text == "Turn: 12\nEra: Ancient Era"


def _tool_call_snapshot(calls: Iterable[Any]) -> list[tuple[str, dict[str, Any], str]]:
    return [(call.tool, call.arguments, call.reasoning) for call in calls]


def _request_snapshot(requests: Iterable[Any]) -> list[tuple[str, dict[str, Any], str]]:
    return [(request.tool, request.arguments, request.reasoning) for request in requests]


def _value_error_message(func: Callable[[object], object], payload: object) -> str:
    try:
        func(payload)
    except ValueError as exc:
        return str(exc)
    raise AssertionError("expected parser to reject invalid payload")
