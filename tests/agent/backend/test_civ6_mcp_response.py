"""Response parsing and normalization tests for civ6-mcp MCP results."""

from __future__ import annotations

from typing import Any

from civStation.agent.modules.backend.civ6_mcp.operations import (
    Civ6McpOperationDispatcher,
    Civ6McpRequestBuilder,
)
from civStation.agent.modules.backend.civ6_mcp.response import (
    Civ6McpResponseClassification,
    normalize_mcp_response_error,
    normalize_mcp_response_timeout,
    normalize_mcp_tool_result,
)


class FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeResult:
    def __init__(
        self,
        content: list[Any],
        *,
        is_error: bool = False,
        structured_content: dict[str, Any] | None = None,
    ) -> None:
        self.content = content
        self.isError = is_error
        self.structuredContent = structured_content


class FakeTypedClient:
    def __init__(self, result) -> None:  # noqa: ANN001
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def has_tool(self, name: str) -> bool:
        return name == "set_research"

    def call_tool_result(self, name: str, arguments: dict[str, Any] | None = None):  # noqa: ANN201
        self.calls.append((name, dict(arguments or {})))
        return self.result


def test_normalize_mcp_tool_result_extracts_text_and_structured_content() -> None:
    raw = FakeResult(
        [
            FakeTextBlock("Research set."),
            {"type": "text", "text": "Queued Writing."},
            {"type": "image", "data": "..."},
        ],
        structured_content={"ok": True, "id": 42},
    )

    result = normalize_mcp_tool_result("set_research", {"tech_or_civic": "WRITING"}, raw)

    assert result.success is True
    assert result.text == "Research set.\nQueued Writing."
    assert result.structured_content == {"ok": True, "id": 42}
    assert result.content_blocks == ("Research set.", "Queued Writing.")
    assert result.classification == Civ6McpResponseClassification.OK


def test_normalize_mcp_tool_result_preserves_json_rpc_error_without_raising() -> None:
    raw = FakeResult([FakeTextBlock("must specify known tech")], is_error=True)

    result = normalize_mcp_tool_result("set_research", {"tech_or_civic": "X"}, raw)

    assert result.success is False
    assert result.is_error is True
    assert result.error == "must specify known tech"
    assert result.classification == Civ6McpResponseClassification.ERROR


def test_normalize_timeout_result_is_typed_and_unsuccessful() -> None:
    result = normalize_mcp_response_timeout("end_turn", {}, timeout_seconds=1.5)

    assert result.success is False
    assert result.timed_out is True
    assert result.classification == Civ6McpResponseClassification.TIMEOUT
    assert "timed out after 1.5s" in result.error


def test_normalize_error_result_preserves_legacy_timeout_message() -> None:
    result = normalize_mcp_response_error("end_turn", {}, "civ6-mcp tool 'end_turn' timed out after 120s")

    assert result.success is False
    assert result.timed_out is True
    assert result.classification == Civ6McpResponseClassification.TIMEOUT


def test_normalize_error_result_classifies_terminal_text_when_possible() -> None:
    result = normalize_mcp_response_error("end_turn", {}, "*** GAME OVER - VICTORY ***")

    assert result.success is False
    assert result.classification == Civ6McpResponseClassification.GAME_OVER


def test_dispatcher_uses_typed_client_results_without_losing_error_metadata() -> None:
    typed = normalize_mcp_response_error(
        "set_research",
        {"tech_or_civic": "X"},
        "Error: must specify a known tech",
        raw={"jsonrpc": "2.0"},
    )
    client = FakeTypedClient(typed)
    dispatcher = Civ6McpOperationDispatcher(client)  # type: ignore[arg-type]

    result = dispatcher.dispatch(Civ6McpRequestBuilder.action("set_research", {"tech_or_civic": "X"}))

    assert result.success is False
    assert result.classification == "error"
    assert result.error == "Error: must specify a known tech"
    assert result.response is typed
    assert result.response.raw == {"jsonrpc": "2.0"}
    assert client.calls == [("set_research", {"tech_or_civic": "X"})]
