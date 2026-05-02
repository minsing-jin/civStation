"""Response parsing and normalization tests for civ6-mcp MCP results."""

from __future__ import annotations

from typing import Any

import pytest

from civStation.agent.modules.backend.civ6_mcp.operations import (
    Civ6McpOperationDispatcher,
    Civ6McpRequestBuilder,
)
from civStation.agent.modules.backend.civ6_mcp.response import (
    CIV6_MCP_CLASSIFICATION_PRECEDENCE,
    CIV6_MCP_EXCEPTION_CLASSIFICATION_PRECEDENCE,
    Civ6McpClassificationStatus,
    Civ6McpExceptionClassificationRule,
    Civ6McpResponseClassification,
    classify_civ6_mcp_exception,
    classify_civ6_mcp_exception_status,
    classify_civ6_mcp_status,
    classify_civ6_mcp_text,
    normalize_mcp_response_error,
    normalize_mcp_response_exception,
    normalize_mcp_response_text,
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


class TimeoutException(Exception):
    """Predicate-only timeout exception used by upstream MCP transports."""


class ClosedResourceError(Exception):
    """Predicate-only closed-resource exception used by anyio transports."""


class UnknownTransportException(Exception):
    """Unknown exception name that should fall through to fatal."""


def test_civ6_mcp_status_taxonomy_contains_only_canonical_ac_statuses() -> None:
    assert {status.value for status in Civ6McpClassificationStatus} == {
        "success",
        "blocked",
        "retryable",
        "fatal",
        "aborted",
        "hang",
        "game_over",
    }


def test_civ6_mcp_classification_precedence_is_explicit_and_terminal_first() -> None:
    assert [rule.status for rule in CIV6_MCP_CLASSIFICATION_PRECEDENCE] == [
        Civ6McpClassificationStatus.GAME_OVER,
        Civ6McpClassificationStatus.ABORTED,
        Civ6McpClassificationStatus.HANG,
        Civ6McpClassificationStatus.FATAL,
        Civ6McpClassificationStatus.BLOCKED,
        Civ6McpClassificationStatus.RETRYABLE,
        Civ6McpClassificationStatus.SUCCESS,
    ]


def test_civ6_mcp_exception_precedence_uses_canonical_taxonomy() -> None:
    assert all(
        isinstance(rule, Civ6McpExceptionClassificationRule) for rule in CIV6_MCP_EXCEPTION_CLASSIFICATION_PRECEDENCE
    )
    assert [rule.status for rule in CIV6_MCP_EXCEPTION_CLASSIFICATION_PRECEDENCE] == [
        Civ6McpClassificationStatus.RETRYABLE,
        Civ6McpClassificationStatus.ABORTED,
        Civ6McpClassificationStatus.FATAL,
        Civ6McpClassificationStatus.FATAL,
    ]
    assert {rule.legacy_classification for rule in CIV6_MCP_EXCEPTION_CLASSIFICATION_PRECEDENCE}.issubset(
        set(Civ6McpResponseClassification)
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Research set.", Civ6McpClassificationStatus.SUCCESS),
        ("Cannot end turn: incoming trade deal pending.", Civ6McpClassificationStatus.BLOCKED),
        ("End turn requested but units still need orders.", Civ6McpClassificationStatus.RETRYABLE),
        ("civ6-mcp tool 'end_turn' timed out after 120s", Civ6McpClassificationStatus.RETRYABLE),
        ("Error: must specify a known tech", Civ6McpClassificationStatus.FATAL),
        ("RUN ABORTED: upstream civ6-mcp server exited.", Civ6McpClassificationStatus.ABORTED),
        ("HANG RECOVERY FAILED after repeated no-op turns.", Civ6McpClassificationStatus.HANG),
        ("*** GAME OVER - VICTORY ***", Civ6McpClassificationStatus.GAME_OVER),
    ],
)
def test_classify_civ6_mcp_status_maps_expected_text_patterns(
    text: str,
    expected: Civ6McpClassificationStatus,
) -> None:
    assert classify_civ6_mcp_status(text) == expected


def test_classify_civ6_mcp_status_applies_precedence_before_success_default() -> None:
    text = "*** GAME OVER - VICTORY ***\nError: Cannot end turn: units still need orders."

    assert classify_civ6_mcp_status(text) == Civ6McpClassificationStatus.GAME_OVER


@pytest.mark.parametrize(
    ("exc", "expected_status", "expected_legacy"),
    [
        (TimeoutError("tool call exceeded 30s"), Civ6McpClassificationStatus.RETRYABLE, "timeout"),
        (ConnectionResetError("stdio stream closed"), Civ6McpClassificationStatus.ABORTED, "aborted"),
        (ConnectionError("ConnectionError: stdio stream closed"), Civ6McpClassificationStatus.ABORTED, "aborted"),
        (EOFError("server pipe closed"), Civ6McpClassificationStatus.ABORTED, "aborted"),
        (ValueError("unsupported civ6-mcp tool"), Civ6McpClassificationStatus.FATAL, "error"),
        (RuntimeError("RUN ABORTED: upstream server exited"), Civ6McpClassificationStatus.ABORTED, "aborted"),
        (RuntimeError("*** GAME OVER - VICTORY ***"), Civ6McpClassificationStatus.GAME_OVER, "game_over"),
    ],
)
def test_classify_civ6_mcp_exception_uses_taxonomy_and_message_precedence(
    exc: BaseException,
    expected_status: Civ6McpClassificationStatus,
    expected_legacy: str,
) -> None:
    assert classify_civ6_mcp_exception_status(exc) == expected_status
    assert classify_civ6_mcp_exception(exc).value == expected_legacy


@pytest.mark.parametrize(
    ("text", "expected_status", "expected_legacy"),
    [
        ("\n  *** GAME OVER - DEFEAT ***", Civ6McpClassificationStatus.GAME_OVER, "game_over"),
        ("stderr:\nRUN ABORTED: operator stopped automation.", Civ6McpClassificationStatus.ABORTED, "aborted"),
        ("tool output\nHANG RECOVERY FAILED after repeated no-op turns.", Civ6McpClassificationStatus.HANG, "hang"),
        ("Tool failed:\nError: must specify a known tech", Civ6McpClassificationStatus.FATAL, "error"),
        ("Cannot end turn - choose production first.", Civ6McpClassificationStatus.BLOCKED, "blocked"),
        ("End turn requested, but units still need orders.", Civ6McpClassificationStatus.RETRYABLE, "soft_block"),
        ("TimeoutError: civ6-mcp request exceeded 30s", Civ6McpClassificationStatus.RETRYABLE, "timeout"),
        (
            "asyncio.exceptions.TimeoutError: civ6-mcp request exceeded 30s",
            Civ6McpClassificationStatus.RETRYABLE,
            "timeout",
        ),
    ],
)
def test_raw_civ6_mcp_text_rules_handle_common_wrapping_and_variants(
    text: str,
    expected_status: Civ6McpClassificationStatus,
    expected_legacy: str,
) -> None:
    assert classify_civ6_mcp_status(text) == expected_status
    assert classify_civ6_mcp_text(text).value == expected_legacy


@pytest.mark.parametrize(
    ("raw", "expected_status", "expected_legacy", "expected_success", "expected_error"),
    [
        pytest.param(
            {"content": [{"type": "text", "text": "Research set."}]},
            Civ6McpClassificationStatus.SUCCESS,
            Civ6McpResponseClassification.OK,
            True,
            "",
            id="success",
        ),
        pytest.param(
            {"content": [{"type": "text", "text": "Cannot end turn: choose production first."}]},
            Civ6McpClassificationStatus.BLOCKED,
            Civ6McpResponseClassification.BLOCKED,
            False,
            "Cannot end turn: choose production first.",
            id="blocked",
        ),
        pytest.param(
            {"content": [{"type": "text", "text": "End turn requested, but units still need orders."}]},
            Civ6McpClassificationStatus.RETRYABLE,
            Civ6McpResponseClassification.SOFT_BLOCK,
            True,
            "",
            id="soft-block",
        ),
        pytest.param(
            {"content": [{"type": "text", "text": "TimeoutError: civ6-mcp request exceeded 30s"}]},
            Civ6McpClassificationStatus.RETRYABLE,
            Civ6McpResponseClassification.TIMEOUT,
            False,
            "TimeoutError: civ6-mcp request exceeded 30s",
            id="timeout",
        ),
        pytest.param(
            {"content": [{"type": "text", "text": "Error: unsupported policy card"}], "isError": True},
            Civ6McpClassificationStatus.FATAL,
            Civ6McpResponseClassification.ERROR,
            False,
            "Error: unsupported policy card",
            id="fatal",
        ),
        pytest.param(
            {"content": [{"type": "text", "text": "RUN ABORTED: operator stopped automation."}]},
            Civ6McpClassificationStatus.ABORTED,
            Civ6McpResponseClassification.ABORTED,
            False,
            "RUN ABORTED: operator stopped automation.",
            id="aborted",
        ),
        pytest.param(
            {"content": [{"type": "text", "text": "HANG RECOVERY FAILED after repeated no-op turns."}]},
            Civ6McpClassificationStatus.HANG,
            Civ6McpResponseClassification.HANG,
            False,
            "HANG RECOVERY FAILED after repeated no-op turns.",
            id="hang",
        ),
        pytest.param(
            {"content": [{"type": "text", "text": "*** GAME OVER - DEFEAT ***"}]},
            Civ6McpClassificationStatus.GAME_OVER,
            Civ6McpResponseClassification.GAME_OVER,
            False,
            "*** GAME OVER - DEFEAT ***",
            id="game-over",
        ),
    ],
)
def test_normalize_raw_mcp_tool_result_covers_all_classification_outcomes(
    raw: dict[str, Any],
    expected_status: Civ6McpClassificationStatus,
    expected_legacy: Civ6McpResponseClassification,
    expected_success: bool,
    expected_error: str,
) -> None:
    result = normalize_mcp_tool_result("end_turn", {}, raw)

    assert result.status == expected_status
    assert result.classification == expected_legacy
    assert result.success is expected_success
    assert result.error == expected_error


def test_normalize_raw_mcp_tool_result_extracts_error_message_without_content() -> None:
    raw = {"content": [], "is_error": True, "error": {"message": "JSON-RPC server returned -32603"}}

    result = normalize_mcp_tool_result("set_research", {"tech_or_civic": "X"}, raw)

    assert result.status == Civ6McpClassificationStatus.FATAL
    assert result.classification == Civ6McpResponseClassification.ERROR
    assert result.success is False
    assert result.error == "JSON-RPC server returned -32603"
    assert result.text == "JSON-RPC server returned -32603"


@pytest.mark.parametrize(
    ("text", "expected_status", "expected_legacy"),
    [
        pytest.param(
            "*** GAME OVER - VICTORY ***\nRUN ABORTED: operator stop\nHANG RECOVERY FAILED\nError: failed\n"
            "Cannot end turn: pending choice\nTimeoutError: request exceeded 30s",
            Civ6McpClassificationStatus.GAME_OVER,
            Civ6McpResponseClassification.GAME_OVER,
            id="game-over-first",
        ),
        pytest.param(
            "RUN ABORTED: operator stop\nHANG RECOVERY FAILED\nError: failed\nCannot end turn: pending choice\n"
            "TimeoutError: request exceeded 30s",
            Civ6McpClassificationStatus.ABORTED,
            Civ6McpResponseClassification.ABORTED,
            id="aborted-before-hang",
        ),
        pytest.param(
            "HANG RECOVERY FAILED\nError: failed\nCannot end turn: pending choice\nTimeoutError: request exceeded 30s",
            Civ6McpClassificationStatus.HANG,
            Civ6McpResponseClassification.HANG,
            id="hang-before-error",
        ),
        pytest.param(
            "Error: Cannot end turn because request timed out",
            Civ6McpClassificationStatus.FATAL,
            Civ6McpResponseClassification.ERROR,
            id="fatal-before-blocked-or-retryable",
        ),
        pytest.param(
            "Cannot end turn: pending choice timed out",
            Civ6McpClassificationStatus.BLOCKED,
            Civ6McpResponseClassification.BLOCKED,
            id="blocked-before-retryable",
        ),
        pytest.param(
            "TimeoutError: Cannot end turn after 30s",
            Civ6McpClassificationStatus.RETRYABLE,
            Civ6McpResponseClassification.TIMEOUT,
            id="timeout-error-not-generic-error",
        ),
        pytest.param(
            "",
            Civ6McpClassificationStatus.SUCCESS,
            Civ6McpResponseClassification.OK,
            id="empty-success-default",
        ),
    ],
)
def test_classification_precedence_handles_collisions_and_edge_cases(
    text: str,
    expected_status: Civ6McpClassificationStatus,
    expected_legacy: Civ6McpResponseClassification,
) -> None:
    assert classify_civ6_mcp_status(text) == expected_status
    assert classify_civ6_mcp_text(text) == expected_legacy


@pytest.mark.parametrize(
    ("exc", "expected_status", "expected_legacy", "expected_timed_out"),
    [
        pytest.param(
            TimeoutException("server did not respond"),
            Civ6McpClassificationStatus.RETRYABLE,
            Civ6McpResponseClassification.TIMEOUT,
            True,
            id="predicate-timeout",
        ),
        pytest.param(
            ClosedResourceError("stdio receive stream closed"),
            Civ6McpClassificationStatus.ABORTED,
            Civ6McpResponseClassification.ABORTED,
            False,
            id="predicate-aborted",
        ),
        pytest.param(
            RuntimeError("End turn requested, but units still need orders."),
            Civ6McpClassificationStatus.RETRYABLE,
            Civ6McpResponseClassification.SOFT_BLOCK,
            False,
            id="message-soft-block-before-runtime-fatal",
        ),
        pytest.param(
            TimeoutError("Error: upstream replied with an error envelope"),
            Civ6McpClassificationStatus.RETRYABLE,
            Civ6McpResponseClassification.TIMEOUT,
            True,
            id="timeout-type-before-generic-error-message",
        ),
        pytest.param(
            UnknownTransportException("unmapped transport failure"),
            Civ6McpClassificationStatus.FATAL,
            Civ6McpResponseClassification.ERROR,
            False,
            id="unknown-exception-defaults-fatal",
        ),
    ],
)
def test_exception_classification_covers_predicates_and_precedence_edges(
    exc: BaseException,
    expected_status: Civ6McpClassificationStatus,
    expected_legacy: Civ6McpResponseClassification,
    expected_timed_out: bool,
) -> None:
    normalized = normalize_mcp_response_exception("end_turn", {}, exc)

    assert classify_civ6_mcp_exception_status(exc) == expected_status
    assert classify_civ6_mcp_exception(exc) == expected_legacy
    assert normalized.status == expected_status
    assert normalized.classification == expected_legacy
    assert normalized.timed_out is expected_timed_out


def test_normalize_text_response_uses_raw_classification_taxonomy() -> None:
    result = normalize_mcp_response_text(
        "end_turn",
        {},
        "\nEnd turn requested, but units still need orders.\n",
    )

    assert result.success is True
    assert result.classification == Civ6McpResponseClassification.SOFT_BLOCK
    assert result.status == Civ6McpClassificationStatus.RETRYABLE
    assert result.error == ""


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
    assert result.status == Civ6McpClassificationStatus.SUCCESS


def test_normalize_mcp_tool_result_preserves_json_rpc_error_without_raising() -> None:
    raw = FakeResult([FakeTextBlock("must specify known tech")], is_error=True)

    result = normalize_mcp_tool_result("set_research", {"tech_or_civic": "X"}, raw)

    assert result.success is False
    assert result.is_error is True
    assert result.error == "must specify known tech"
    assert result.classification == Civ6McpResponseClassification.ERROR
    assert result.status == Civ6McpClassificationStatus.FATAL


def test_normalize_timeout_result_is_typed_and_unsuccessful() -> None:
    result = normalize_mcp_response_timeout("end_turn", {}, timeout_seconds=1.5)

    assert result.success is False
    assert result.timed_out is True
    assert result.classification == Civ6McpResponseClassification.TIMEOUT
    assert result.status == Civ6McpClassificationStatus.RETRYABLE
    assert "timed out after 1.5s" in result.error


def test_normalize_error_result_preserves_legacy_timeout_message() -> None:
    result = normalize_mcp_response_error("end_turn", {}, "civ6-mcp tool 'end_turn' timed out after 120s")

    assert result.success is False
    assert result.timed_out is True
    assert result.classification == Civ6McpResponseClassification.TIMEOUT
    assert result.status == Civ6McpClassificationStatus.RETRYABLE


def test_normalize_error_result_classifies_terminal_text_when_possible() -> None:
    result = normalize_mcp_response_error("end_turn", {}, "*** GAME OVER - VICTORY ***")

    assert result.success is False
    assert result.classification == Civ6McpResponseClassification.GAME_OVER
    assert result.status == Civ6McpClassificationStatus.GAME_OVER


def test_normalize_exception_result_preserves_exception_taxonomy() -> None:
    exc = TimeoutError("tool call exceeded 30s")

    result = normalize_mcp_response_exception("end_turn", {}, exc)

    assert result.success is False
    assert result.is_error is True
    assert result.timed_out is True
    assert result.error == "tool call exceeded 30s"
    assert result.classification == Civ6McpResponseClassification.TIMEOUT
    assert result.status == Civ6McpClassificationStatus.RETRYABLE
    assert result.raw is exc


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
