from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from PIL import Image

from civStation.mcp.civ6_mcp_adapters import (
    CIV6_MCP_ADAPTER_NAME,
    register_civ6_mcp_adapters,
)
from civStation.mcp.runtime import LayerAdapterRegistry
from civStation.mcp.server import LayeredComputerUseMCP


class FakeCiv6McpClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._responses = {
            "get_game_overview": (
                "Turn: 12\n"
                "Era: Classical Era\n"
                "Science: +14.0/turn\n"
                "Culture: +6.0/turn\n"
                "Researching: WRITING\n"
                "Civic Researching: POLITICAL_PHILOSOPHY\n"
            ),
            "set_research": "Research set to MATHEMATICS.",
        }

    def has_tool(self, name: str) -> bool:
        return name in self._responses

    def tool_schemas(self) -> dict[str, dict[str, Any]]:
        return {
            "get_game_overview": {"description": "current overview", "input_schema": {"properties": {}}},
            "set_research": {
                "description": "queue research",
                "input_schema": {
                    "properties": {"tech_or_civic": {"type": "string"}},
                    "required": ["tech_or_civic"],
                },
            },
        }

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        self.calls.append((name, dict(arguments or {})))
        return self._responses[name]


@dataclass
class FakeResponse:
    content: str


class FakeProvider:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def _build_text_content(self, text: str) -> dict[str, str]:
        return {"type": "text", "text": text}

    def _send_to_api(self, content_parts, **kwargs):  # noqa: ANN001, ARG002
        prompt = "".join(part.get("text", "") for part in content_parts if isinstance(part, dict))
        self.prompts.append(prompt)
        return FakeResponse(
            content=json.dumps(
                {
                    "tool_calls": [
                        {
                            "tool": "set_research",
                            "arguments": {"tech_or_civic": "MATHEMATICS"},
                            "reasoning": "Unlock stronger campuses.",
                        }
                    ]
                }
            )
        )


def _call_tool(app: LayeredComputerUseMCP, name: str, arguments: dict[str, object]):
    result = asyncio.run(app.server.call_tool(name, arguments))
    if isinstance(result, tuple) and len(result) == 2:
        return result[1]
    return result


def test_register_civ6_mcp_adapters_adds_named_slots() -> None:
    client = FakeCiv6McpClient()
    registry = LayerAdapterRegistry(include_builtins=False)

    register_civ6_mcp_adapters(registry, client)  # type: ignore[arg-type]

    available = registry.list_available()
    assert CIV6_MCP_ADAPTER_NAME in available["action_router"]
    assert CIV6_MCP_ADAPTER_NAME in available["action_planner"]
    assert CIV6_MCP_ADAPTER_NAME in available["context_observer"]
    assert CIV6_MCP_ADAPTER_NAME in available["action_executor"]


def test_enable_civ6_mcp_registers_adapter_set_without_replacing_existing_registrations(monkeypatch) -> None:
    from civStation.agent.modules.backend.civ6_mcp import turn_loop

    client = FakeCiv6McpClient()
    build_calls: list[dict[str, Any]] = []

    def custom_router(session, pil_image):  # noqa: ANN001
        return {"primitive": "custom", "reasoning": "preserved"}

    def custom_planner(session, pil_image, primitive_name, **kwargs):  # noqa: ANN001, ARG001
        return None

    def custom_observer(session, pil_image):  # noqa: ANN001, ARG001
        return {"situation_summary": "preserved"}

    def custom_refiner(session, raw_input):  # noqa: ANN001, ARG001
        return raw_input

    def custom_executor(session, action, capture):  # noqa: ANN001, ARG001
        return {"executed": False}

    def fake_build_civ6_mcp_client(**kwargs):  # noqa: ANN003
        build_calls.append(dict(kwargs))
        return client

    monkeypatch.setattr(turn_loop, "build_civ6_mcp_client", fake_build_civ6_mcp_client)

    registry = LayerAdapterRegistry(
        action_routers={"custom": custom_router},
        action_planners={"custom": custom_planner},
        context_observers={"custom": custom_observer},
        strategy_refiners={"custom": custom_refiner},
        action_executors={"custom": custom_executor},
    )
    existing = {
        "action_router": dict(registry.action_routers),
        "action_planner": dict(registry.action_planners),
        "context_observer": dict(registry.context_observers),
        "strategy_refiner": dict(registry.strategy_refiners),
        "action_executor": dict(registry.action_executors),
    }

    registry.enable_civ6_mcp(
        install_path="/tmp/civ6-mcp",
        launcher="python",
        env_overrides={"CIV6_MCP_TEST": "1"},
    )

    assert build_calls == [
        {
            "install_path": "/tmp/civ6-mcp",
            "launcher": "python",
            "env_overrides": {"CIV6_MCP_TEST": "1"},
        }
    ]
    available = registry.list_available()
    assert CIV6_MCP_ADAPTER_NAME in available["action_router"]
    assert CIV6_MCP_ADAPTER_NAME in available["action_planner"]
    assert CIV6_MCP_ADAPTER_NAME in available["context_observer"]
    assert CIV6_MCP_ADAPTER_NAME in available["action_executor"]
    assert CIV6_MCP_ADAPTER_NAME not in available["strategy_refiner"]

    for slot_name, adapters in existing.items():
        current = getattr(registry, f"{slot_name}s")
        for adapter_name, adapter in adapters.items():
            assert current[adapter_name] is adapter


def test_civ6_mcp_adapters_drive_outward_workflow_without_duplicate_observation_notes() -> None:
    client = FakeCiv6McpClient()
    provider = FakeProvider()
    registry = LayerAdapterRegistry(
        include_builtins=False,
        provider_factory=lambda provider_name, model: provider,  # noqa: ARG005
        screen_capture=lambda: (Image.new("RGB", (8, 8), "black"), 8, 8, 0, 0),
    )
    register_civ6_mcp_adapters(registry, client)  # type: ignore[arg-type]
    app = LayeredComputerUseMCP(adapter_registry=registry)

    create_result = _call_tool(
        app,
        "session_create",
        {
            "name": "civ6-mcp",
            "runtime": {"execution_mode": "live", "require_execute_confirmation": False},
            "adapter_overrides": {
                "action_router": CIV6_MCP_ADAPTER_NAME,
                "action_planner": CIV6_MCP_ADAPTER_NAME,
                "context_observer": CIV6_MCP_ADAPTER_NAME,
                "action_executor": CIV6_MCP_ADAPTER_NAME,
            },
        },
    )
    session_id = create_result["session_id"]

    observe_result = _call_tool(app, "workflow_observe", {"session_id": session_id, "use_last_capture": False})
    context = observe_result["context"]
    assert context["global_context"]["current_turn"] == 12
    assert context["global_context"]["current_research"] == "WRITING"
    notes = context["high_level_context"]["notes"]
    assert len([note for note in notes if "Turn 12" in note]) == 1

    route_result = _call_tool(app, "action_route", {"session_id": session_id})
    assert route_result["primitive"] == "civ6_mcp"

    plan_result = _call_tool(
        app,
        "action_plan",
        {
            "session_id": session_id,
            "primitive_name": "civ6_mcp",
            "recent_actions_override": "previous call: get_game_overview",
        },
    )
    action = plan_result["action"]
    assert action["action"] == "civ6_mcp_tool_plan"
    assert "previous call: get_game_overview" in provider.prompts[-1]
    payload = json.loads(action["text"])
    assert payload["tool_calls"][0]["tool"] == "set_research"

    execute_result = _call_tool(app, "action_execute", {"session_id": session_id, "action": action})
    assert execute_result["executed"] is True
    assert execute_result["results"][0]["success"] is True
    assert ("set_research", {"tech_or_civic": "MATHEMATICS"}) in client.calls
