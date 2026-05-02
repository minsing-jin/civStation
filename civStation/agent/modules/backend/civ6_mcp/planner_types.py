"""Planner-facing types for the civ6-mcp backend.

These types model civ6-mcp tool calls, not VLM/computer-use screen actions.
Keeping them separate prevents the MCP backend from depending on pixel action
schemas used by the default VLM pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TypeAlias, runtime_checkable

from civStation.agent.modules.backend.civ6_mcp.executor import ToolCall
from civStation.agent.modules.backend.civ6_mcp.operations import (
    ACTION_TOOLS,
    END_TURN_TOOL,
    OBSERVATION_TOOLS,
)

Civ6McpToolArguments: TypeAlias = dict[str, Any]
Civ6McpToolSchema: TypeAlias = Mapping[str, Any]
Civ6McpToolCatalog: TypeAlias = Mapping[str, Civ6McpToolSchema]
Civ6McpPlannerPayload: TypeAlias = dict[str, Any] | Sequence[Mapping[str, Any]]


class Civ6McpIntentType(str, Enum):
    """Planner intent categories understood by the civ6-mcp backend."""

    OBSERVE = "observe"
    ACT = "act"
    END_TURN = "end_turn"


class Civ6McpActionType(str, Enum):
    """Executable action categories for the civ6-mcp backend."""

    TOOL_CALL = "tool_call"


@dataclass(frozen=True)
class Civ6McpPlannerIntent:
    """A planner intent before execution.

    The intent is still backend-specific: it identifies an upstream civ6-mcp
    tool and its arguments rather than a VLM screen action.
    """

    tool: str
    arguments: Civ6McpToolArguments = field(default_factory=dict)
    reasoning: str = ""
    intent_type: Civ6McpIntentType = Civ6McpIntentType.ACT

    @classmethod
    def from_tool(
        cls,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        reasoning: str = "",
    ) -> Civ6McpPlannerIntent:
        """Build an intent and infer its category from the civ6-mcp tool name."""
        return cls(
            tool=tool,
            arguments=dict(arguments or {}),
            reasoning=reasoning,
            intent_type=infer_civ6_mcp_intent_type(tool),
        )

    @property
    def type(self) -> Civ6McpIntentType:
        """Alias for consumers that use action-schema style ``type`` fields."""
        return self.intent_type

    def to_action(self) -> Civ6McpPlannerAction:
        """Convert this intent into the executable backend action shape."""
        return Civ6McpPlannerAction(
            tool=self.tool,
            arguments=dict(self.arguments),
            reasoning=self.reasoning,
        )

    def to_tool_call(self) -> ToolCall:
        """Convert this intent directly into the executor's ToolCall."""
        return self.to_action().to_tool_call()


@dataclass(frozen=True)
class Civ6McpPlannerAction:
    """Executable planner action for civ6-mcp.

    This is the MCP backend counterpart to VLM ``AgentAction`` but represents a
    JSON-RPC tool invocation, not coordinates or keyboard/mouse operations.
    """

    tool: str
    arguments: Civ6McpToolArguments = field(default_factory=dict)
    reasoning: str = ""
    action_type: Civ6McpActionType = Civ6McpActionType.TOOL_CALL

    @classmethod
    def from_tool_call(cls, call: ToolCall) -> Civ6McpPlannerAction:
        """Build a planner action from an executor ToolCall."""
        return cls(
            tool=call.tool,
            arguments=dict(call.arguments),
            reasoning=call.reasoning,
        )

    @property
    def type(self) -> Civ6McpActionType:
        """Alias for consumers that use action-schema style ``type`` fields."""
        return self.action_type

    def to_tool_call(self) -> ToolCall:
        """Convert this action into the executor's ToolCall."""
        return ToolCall(
            tool=self.tool,
            arguments=dict(self.arguments),
            reasoning=self.reasoning,
        )


@dataclass
class PlannerResult:
    """Parsed output from a civ6-mcp planner invocation."""

    tool_calls: list[ToolCall]
    raw_response: str = ""
    parsed_payload: dict[str, Any] | None = None

    @property
    def actions(self) -> list[ToolCall]:
        """Alias for planner results consumed as backend actions."""
        return self.tool_calls


@runtime_checkable
class Civ6McpPlannerResponse(Protocol):
    """Response object returned by civStation text-capable LLM providers."""

    content: str


@runtime_checkable
class Civ6McpPlannerProvider(Protocol):
    """Provider surface required by Civ6McpToolPlanner."""

    def _build_text_content(self, text: str) -> object:
        """Build a text-only content part for the provider."""
        ...

    def _send_to_api(self, content_parts: Sequence[object], **kwargs: Any) -> Civ6McpPlannerResponse:
        """Send content parts to the provider and return a response with text content."""
        ...


@runtime_checkable
class Civ6McpPlanner(Protocol):
    """Protocol implemented by civ6-mcp planners."""

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        """Tool names the planner may emit."""
        ...

    def render_tool_catalog(self) -> str:
        """Render the tool catalog included in planner prompts."""
        ...

    def plan(
        self,
        *,
        strategy: str,
        state_context: str,
        recent_calls: str,
        hitl_directive: str = "",
    ) -> PlannerResult:
        """Produce a parsed civ6-mcp tool-call plan."""
        ...


def infer_civ6_mcp_intent_type(tool: str) -> Civ6McpIntentType:
    """Infer a planner intent category from a supported civ6-mcp tool name."""
    if tool == END_TURN_TOOL:
        return Civ6McpIntentType.END_TURN
    if tool in OBSERVATION_TOOLS:
        return Civ6McpIntentType.OBSERVE
    if tool in ACTION_TOOLS:
        return Civ6McpIntentType.ACT
    raise ValueError(f"Unsupported civ6-mcp tool: {tool!r}")


Civ6McpIntent = Civ6McpPlannerIntent
Civ6McpAction = Civ6McpPlannerAction
Civ6McpIntentKind = Civ6McpIntentType
Civ6McpBackendIntent = Civ6McpPlannerIntent
Civ6McpBackendAction = Civ6McpPlannerAction


__all__ = [
    "Civ6McpAction",
    "Civ6McpActionType",
    "Civ6McpBackendAction",
    "Civ6McpBackendIntent",
    "Civ6McpIntent",
    "Civ6McpIntentKind",
    "Civ6McpIntentType",
    "Civ6McpPlanner",
    "Civ6McpPlannerAction",
    "Civ6McpPlannerIntent",
    "Civ6McpPlannerPayload",
    "Civ6McpPlannerProvider",
    "Civ6McpPlannerResponse",
    "Civ6McpToolArguments",
    "Civ6McpToolCatalog",
    "Civ6McpToolSchema",
    "PlannerResult",
    "infer_civ6_mcp_intent_type",
]
