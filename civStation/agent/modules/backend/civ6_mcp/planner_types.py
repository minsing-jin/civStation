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
from civStation.agent.modules.backend.civ6_mcp.observation_schema import Civ6McpNormalizedObservation
from civStation.agent.modules.backend.civ6_mcp.operations import (
    _ACTION_TOOL_ORDER,
    _OBSERVATION_TOOL_ORDER,
    ACTION_TOOLS,
    END_TURN_TOOL,
    OBSERVATION_TOOLS,
)

Civ6McpToolArguments: TypeAlias = dict[str, Any]
Civ6McpToolSchema: TypeAlias = Mapping[str, Any]
Civ6McpToolCatalog: TypeAlias = Mapping[str, Civ6McpToolSchema]
Civ6McpPlannerPayload: TypeAlias = dict[str, Any] | Sequence[Mapping[str, Any]]

DEFAULT_PLANNER_TOOL_ALLOWLIST: tuple[str, ...] = (
    *_OBSERVATION_TOOL_ORDER,
    *_ACTION_TOOL_ORDER,
    END_TURN_TOOL,
)
"""Canonical ordered civ6-mcp tool allow-list for planner prompts and defaults."""


class Civ6McpIntentType(str, Enum):
    """Planner intent categories for supported civ6-mcp tools."""

    OBSERVE = "observe"
    ACT = "act"
    END_TURN = "end_turn"


class Civ6McpActionType(str, Enum):
    """Executable action categories emitted for civ6-mcp tool calls."""

    TOOL_CALL = "tool_call"


@dataclass(frozen=True)
class Civ6McpPlannerIntent:
    """Planner intent describing one civ6-mcp tool before execution."""

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
        """Build an intent and classify the supported civ6-mcp tool name."""
        return cls(
            tool=tool,
            arguments=dict(arguments or {}),
            reasoning=reasoning,
            intent_type=infer_civ6_mcp_intent_type(tool),
        )

    @property
    def type(self) -> Civ6McpIntentType:
        """Expose the intent category as an action-schema-style ``type`` field."""
        return self.intent_type

    def to_action(self) -> Civ6McpPlannerAction:
        """Return this intent as an executable civ6-mcp planner action."""
        return Civ6McpPlannerAction(
            tool=self.tool,
            arguments=dict(self.arguments),
            reasoning=self.reasoning,
        )

    def to_tool_call(self) -> ToolCall:
        """Return this intent as the executor ``ToolCall`` shape."""
        return self.to_action().to_tool_call()


@dataclass(frozen=True)
class Civ6McpPlannerAction:
    """Executable planner action for one civ6-mcp JSON-RPC tool call."""

    tool: str
    arguments: Civ6McpToolArguments = field(default_factory=dict)
    reasoning: str = ""
    action_type: Civ6McpActionType = Civ6McpActionType.TOOL_CALL

    @classmethod
    def from_tool_call(cls, call: ToolCall) -> Civ6McpPlannerAction:
        """Build a planner action from an executor ``ToolCall``."""
        return cls(
            tool=call.tool,
            arguments=dict(call.arguments),
            reasoning=call.reasoning,
        )

    @property
    def type(self) -> Civ6McpActionType:
        """Expose the action category as an action-schema-style ``type`` field."""
        return self.action_type

    def to_tool_call(self) -> ToolCall:
        """Return this action as the executor ``ToolCall`` shape."""
        return ToolCall(
            tool=self.tool,
            arguments=dict(self.arguments),
            reasoning=self.reasoning,
        )


@dataclass
class PlannerResult:
    """Parsed civ6-mcp planner output with executable tool calls."""

    tool_calls: list[ToolCall]
    raw_response: str = ""
    parsed_payload: dict[str, Any] | None = None

    @property
    def actions(self) -> list[ToolCall]:
        """Expose planner tool calls under the backend-action alias."""
        return self.tool_calls


@runtime_checkable
class Civ6McpPlannerResponse(Protocol):
    """Minimal text response returned by civStation planner providers."""

    content: str


@runtime_checkable
class Civ6McpPlannerProvider(Protocol):
    """Text-provider surface required to request a civ6-mcp plan."""

    def _build_text_content(self, text: str) -> object:
        """Build a text-only content part for the provider."""
        ...

    def _send_to_api(self, content_parts: Sequence[object], **kwargs: Any) -> Civ6McpPlannerResponse:
        """Send content parts to the provider and return a response with text content."""
        ...


@runtime_checkable
class Civ6McpPlanner(Protocol):
    """Planner protocol that converts civ6-mcp observations into tool calls."""

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        """Return tool names the planner may emit."""
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
        """Return a parsed civ6-mcp tool-call plan from rendered state context."""
        ...

    def plan_from_observation(
        self,
        *,
        observation: Civ6McpNormalizedObservation | object,
        strategy: str,
        recent_calls: str,
        hitl_directive: str = "",
    ) -> PlannerResult:
        """Return a parsed civ6-mcp tool-call plan from an observation payload."""
        ...


def infer_civ6_mcp_intent_type(tool: str) -> Civ6McpIntentType:
    """Classify a supported civ6-mcp tool name as observe, act, or end-turn."""
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
    "DEFAULT_PLANNER_TOOL_ALLOWLIST",
    "PlannerResult",
    "infer_civ6_mcp_intent_type",
]
