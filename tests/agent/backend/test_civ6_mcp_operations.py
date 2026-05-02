"""Tests for civ6-mcp backend request construction and dispatch."""

from __future__ import annotations

from typing import Any

import pytest

from civStation.agent.modules.backend.civ6_mcp.client import Civ6McpError
from civStation.agent.modules.backend.civ6_mcp.operations import (
    _DOCUMENTED_PREFIX_CLASSIFICATIONS as OPERATIONS_DOCUMENTED_PREFIX_CLASSIFICATIONS,
)
from civStation.agent.modules.backend.civ6_mcp.operations import (
    _UNCOVERED_DOCUMENTED_PREFIX_CLASSIFICATIONS as OPERATIONS_UNCOVERED_DOCUMENTED_PREFIX_CLASSIFICATIONS,
)
from civStation.agent.modules.backend.civ6_mcp.operations import (
    Civ6McpOperationDispatcher,
    Civ6McpRequestBuilder,
    SupportedCiv6McpOperation,
    _documented_prefix_classification_gaps,
    classify_civ6_mcp_text,
    coerce_civ6_mcp_requests,
)
from civStation.agent.modules.backend.civ6_mcp.response import (
    _UPSTREAM_TEXT_PREFIX_CHECKLIST,
    _UPSTREAM_TEXT_PREFIX_CLASSIFICATIONS,
)
from civStation.agent.modules.backend.civ6_mcp.response import classify_civ6_mcp_text as classify_response_text


class FakeCiv6McpClient:
    def __init__(self, responses: dict[str, str | Exception], known: set[str] | None = None) -> None:
        self._responses = responses
        self._known = known if known is not None else set(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def has_tool(self, name: str) -> bool:
        return name in self._known

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        self.calls.append((name, dict(arguments or {})))
        response = self._responses.get(name, "")
        if isinstance(response, Exception):
            raise response
        return response


def test_request_builder_constructs_observation_request() -> None:
    request = Civ6McpRequestBuilder.observation("get_game_overview")
    assert request.operation == SupportedCiv6McpOperation.OBSERVE
    assert request.tool == "get_game_overview"
    assert request.arguments == {}


def test_request_builder_rejects_unsupported_tool() -> None:
    with pytest.raises(ValueError, match="Unsupported civ6-mcp tool"):
        Civ6McpRequestBuilder.build("not_real")


def test_request_builder_constructs_end_turn_request_with_reflections() -> None:
    request = Civ6McpRequestBuilder.end_turn(
        tactical="Queued research.",
        strategic="Pursue science.",
        tooling="Tools succeeded.",
        planning="Check cities next.",
        hypothesis="Writing unlocks libraries.",
    )
    assert request.operation == SupportedCiv6McpOperation.END_TURN
    assert request.tool == "end_turn"
    assert request.arguments["strategic"] == "Pursue science."


def test_request_builder_rejects_blank_end_turn_reflection() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Civ6McpRequestBuilder.end_turn(
            tactical="Queued research.",
            strategic="",
            tooling="Tools succeeded.",
            planning="Check cities next.",
            hypothesis="Writing unlocks libraries.",
        )


def test_coerce_requests_accepts_planner_payload() -> None:
    requests = coerce_civ6_mcp_requests(
        {
            "tool_calls": [
                {"tool": "get_units"},
                {"tool": "set_research", "arguments": {"tech_or_civic": "WRITING"}},
            ]
        }
    )
    assert [request.tool for request in requests] == ["get_units", "set_research"]
    assert requests[1].operation == SupportedCiv6McpOperation.ACT


def test_dispatcher_calls_client_with_constructed_request() -> None:
    client = FakeCiv6McpClient({"set_research": "Research set to WRITING."})
    dispatcher = Civ6McpOperationDispatcher(client)  # type: ignore[arg-type]
    request = Civ6McpRequestBuilder.action("set_research", {"tech_or_civic": "WRITING"})
    result = dispatcher.dispatch(request)
    assert result.success is True
    assert result.classification == "ok"
    assert client.calls == [("set_research", {"tech_or_civic": "WRITING"})]


def test_dispatcher_blocks_unknown_server_tool_without_calling_client() -> None:
    client = FakeCiv6McpClient({"get_units": "ok"}, known={"get_units"})
    dispatcher = Civ6McpOperationDispatcher(client)  # type: ignore[arg-type]
    request = Civ6McpRequestBuilder.action("set_research", {"tech_or_civic": "WRITING"})
    result = dispatcher.dispatch(request)
    assert result.success is False
    assert result.classification == "error"
    assert client.calls == []


def test_dispatcher_classifies_terminal_text_and_stops_sequence() -> None:
    client = FakeCiv6McpClient(
        {
            "get_units": "ok",
            "end_turn": "*** GAME OVER - VICTORY ***",
            "set_research": "should not run",
        }
    )
    dispatcher = Civ6McpOperationDispatcher(client)  # type: ignore[arg-type]
    results = dispatcher.dispatch_many(
        [
            Civ6McpRequestBuilder.observation("get_units"),
            Civ6McpRequestBuilder.end_turn(
                tactical="a",
                strategic="b",
                tooling="c",
                planning="d",
                hypothesis="e",
            ),
            Civ6McpRequestBuilder.action("set_research", {"tech_or_civic": "WRITING"}),
        ]
    )
    assert [name for name, _args in client.calls] == ["get_units", "end_turn"]
    assert results[-1].classification == "game_over"


@pytest.mark.parametrize(("text", "expected"), OPERATIONS_DOCUMENTED_PREFIX_CLASSIFICATIONS)
def test_classify_text_covers_documented_upstream_prefixes(text: str, expected: str) -> None:
    assert classify_civ6_mcp_text(text) == expected


@pytest.mark.parametrize(("text", "expected"), OPERATIONS_DOCUMENTED_PREFIX_CLASSIFICATIONS)
def test_documented_prefix_inventory_matches_response_classifier(text: str, expected: str) -> None:
    response_classification = classify_response_text(text).value

    assert response_classification == expected
    assert classify_civ6_mcp_text(text) == response_classification


def test_documented_cannot_end_turn_colon_prefix_sample_is_blocked() -> None:
    cannot_end_turn_samples = tuple(
        (text, expected)
        for text, expected in OPERATIONS_DOCUMENTED_PREFIX_CLASSIFICATIONS
        if text.startswith("Cannot end turn:")
    )

    assert cannot_end_turn_samples == (("Cannot end turn: choose production first.", "blocked"),)


@pytest.mark.parametrize(
    "text",
    [
        "Cannot end turn: choose production first.",
        "Cannot end turn: incoming trade deal pending.",
        "Cannot end turn: pending choice timed out.",
    ],
)
def test_cannot_end_turn_colon_prefix_classifies_as_blocked(text: str) -> None:
    assert text.startswith("Cannot end turn:")
    assert classify_response_text(text).value == "blocked"
    assert classify_civ6_mcp_text(text) == "blocked"


def test_documented_prefix_inventory_covers_shared_upstream_prefixes() -> None:
    covered = {
        (prefix, expected)
        for text, expected in OPERATIONS_DOCUMENTED_PREFIX_CLASSIFICATIONS
        for prefix, classification in _UPSTREAM_TEXT_PREFIX_CLASSIFICATIONS
        if expected == classification and text.startswith(prefix)
    }

    assert covered == set(_UPSTREAM_TEXT_PREFIX_CLASSIFICATIONS)


def test_documented_prefix_inventory_includes_literal_upstream_prefix_samples() -> None:
    documented_prefixes = {
        (text.split(maxsplit=1)[0], expected)
        for text, expected in OPERATIONS_DOCUMENTED_PREFIX_CLASSIFICATIONS
        if text.split(maxsplit=1)[0] in {prefix for prefix, _expected in _UPSTREAM_TEXT_PREFIX_CLASSIFICATIONS}
    }

    assert documented_prefixes == set(_UPSTREAM_TEXT_PREFIX_CLASSIFICATIONS)


def test_operations_documented_prefix_audit_records_no_uncovered_prefixes() -> None:
    uncovered = tuple(
        (item.prefix, item.legacy_classification)
        for item in _UPSTREAM_TEXT_PREFIX_CHECKLIST
        if not any(
            expected == item.legacy_classification and text.startswith(item.prefix)
            for text, expected in OPERATIONS_DOCUMENTED_PREFIX_CLASSIFICATIONS
        )
    )

    assert OPERATIONS_UNCOVERED_DOCUMENTED_PREFIX_CLASSIFICATIONS == uncovered
    assert OPERATIONS_UNCOVERED_DOCUMENTED_PREFIX_CLASSIFICATIONS == ()


def test_documented_prefix_gap_helper_names_each_incomplete_table() -> None:
    gaps = _documented_prefix_classification_gaps(
        {
            "missing_ok": (("ERR:UNIT_NOT_FOUND", "error"),),
            "missing_err": (("OK: action accepted by civ6-mcp.", "ok"),),
        }
    )

    assert gaps == (
        ("missing_ok", "OK:", "ok"),
        ("missing_err", "ERR:", "error"),
    )


def test_dispatcher_rechecks_stop_predicate_between_requests() -> None:
    client = FakeCiv6McpClient(
        {
            "get_units": "ok",
            "set_research": "should not run",
        }
    )
    dispatcher = Civ6McpOperationDispatcher(client)  # type: ignore[arg-type]

    results = dispatcher.dispatch_many(
        [
            Civ6McpRequestBuilder.observation("get_units"),
            Civ6McpRequestBuilder.action("set_research", {"tech_or_civic": "WRITING"}),
        ],
        stop_requested=lambda: len(client.calls) >= 1,
    )

    assert [request.request.tool for request in results] == ["get_units"]
    assert client.calls == [("get_units", {})]


def test_dispatcher_surfaces_client_errors() -> None:
    client = FakeCiv6McpClient({"set_research": Civ6McpError("boom")})
    dispatcher = Civ6McpOperationDispatcher(client)  # type: ignore[arg-type]
    result = dispatcher.dispatch(Civ6McpRequestBuilder.action("set_research", {"tech_or_civic": "WRITING"}))
    assert result.success is False
    assert result.classification == "error"
    assert "boom" in result.error


def test_dispatcher_classifies_non_civ6_mcp_exceptions_without_leaking() -> None:
    client = FakeCiv6McpClient({"end_turn": TimeoutError("tool call exceeded 30s")})
    dispatcher = Civ6McpOperationDispatcher(client)  # type: ignore[arg-type]

    result = dispatcher.dispatch(
        Civ6McpRequestBuilder.end_turn(
            tactical="a",
            strategic="b",
            tooling="c",
            planning="d",
            hypothesis="e",
        )
    )

    assert result.success is False
    assert result.classification == "timeout"
    assert result.status == "retryable"
    assert "tool call exceeded 30s" in result.error
