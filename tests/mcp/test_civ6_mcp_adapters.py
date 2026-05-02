from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, fields
from typing import Any

from PIL import Image

from civStation.agent.modules.backend import BackendKind
from civStation.agent.modules.backend.civ6_mcp.planner_types import PlannerResult
from civStation.agent.modules.backend.civ6_mcp.results import ToolCall
from civStation.mcp import server as server_module
from civStation.mcp.civ6_mcp_adapters import (
    CIV6_MCP_ADAPTER_NAME,
    CIV6_MCP_TOOL_CALL_ACTION,
    CIV6_MCP_TOOL_PLAN_ACTION,
    encode_civ6_mcp_planner_result,
    make_civ6_mcp_observer_adapter,
    make_civ6_mcp_planner_adapter,
    make_civ6_mcp_router_adapter,
    register_civ6_mcp_adapters,
)
from civStation.mcp.runtime import LayerAdapterRegistry, adapter_overrides_for_backend
from civStation.mcp.server import LayeredComputerUseMCP
from civStation.mcp.session import LayeredSession
from civStation.utils.llm_provider.parser import AgentAction

_EXPECTED_ACTION_PLAN_RESULT_FIELDS: tuple[str, ...] = ("session_id", "capture", "primitive", "action")
_EXPECTED_SERIALIZED_PLAN_FIELDS: tuple[str, ...] = ("primitive", "action")
_EXPECTED_AGENT_ACTION_FIELDS: tuple[str, ...] = (
    "action",
    "coord_space",
    "x",
    "y",
    "end_x",
    "end_y",
    "scroll_amount",
    "button",
    "key",
    "text",
    "reasoning",
    "task_status",
    "policy_card_name",
    "policy_source_tab",
    "policy_target_slot_id",
    "policy_reasoning",
)
_EXPECTED_CIV6_MCP_PLANNER_PAYLOAD_FIELDS: tuple[str, ...] = ("tool_calls",)
_EXPECTED_CIV6_MCP_TOOL_CALL_FIELDS: tuple[str, ...] = ("tool", "arguments", "reasoning")
_EXPECTED_CIV6_MCP_OBSERVER_RESULT_FIELDS: tuple[str, ...] = ("situation_summary", "threats", "opportunities")


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
            "end_turn": "Turn ended.",
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
            "end_turn": {
                "description": "finish the current turn",
                "input_schema": {
                    "properties": {
                        "tactical": {"type": "string"},
                        "strategic": {"type": "string"},
                        "tooling": {"type": "string"},
                        "planning": {"type": "string"},
                        "hypothesis": {"type": "string"},
                    },
                    "required": ["tactical", "strategic", "tooling", "planning", "hypothesis"],
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
                        },
                        {
                            "tool": "end_turn",
                            "arguments": {
                                "tactical": "Queued research before ending the turn.",
                                "strategic": "Mathematics supports the science plan.",
                                "tooling": "Used civ6-mcp tool calls only.",
                                "planning": "No further blockers were present.",
                                "hypothesis": "Next turn should continue toward campuses.",
                            },
                            "reasoning": "Finish the turn after setting research.",
                        },
                    ]
                }
            )
        )


def _call_tool(app: LayeredComputerUseMCP, name: str, arguments: dict[str, object]):
    result = asyncio.run(app.server.call_tool(name, arguments))
    if isinstance(result, tuple) and len(result) == 2:
        return result[1]
    return result


def _validate_civ6_mcp_outward_plan_schema(plan_result: dict[str, Any]) -> dict[str, Any]:
    """Assert the civ6-mcp planner adapter preserves the public action_plan schema."""
    assert tuple(plan_result) == _EXPECTED_ACTION_PLAN_RESULT_FIELDS
    assert isinstance(plan_result["session_id"], str)
    assert isinstance(plan_result["capture"], dict)
    return _validate_civ6_mcp_serialized_plan_schema(
        {"primitive": plan_result["primitive"], "action": plan_result["action"]}
    )


def _validate_civ6_mcp_serialized_plan_schema(plan: dict[str, Any]) -> dict[str, Any]:
    """Assert a serialized civ6-mcp plan uses the existing AgentAction envelope."""
    assert tuple(plan) == _EXPECTED_SERIALIZED_PLAN_FIELDS
    assert plan["primitive"] == CIV6_MCP_ADAPTER_NAME

    action = plan["action"]
    assert isinstance(action, dict)
    assert tuple(action) == _EXPECTED_AGENT_ACTION_FIELDS
    assert tuple(field.name for field in fields(AgentAction)) == _EXPECTED_AGENT_ACTION_FIELDS
    assert action["action"] == CIV6_MCP_TOOL_PLAN_ACTION
    assert action["coord_space"] == "normalized"
    assert action["text"]
    assert "tool_calls" not in plan
    assert "actions" not in action

    payload = json.loads(action["text"])
    assert tuple(payload) == _EXPECTED_CIV6_MCP_PLANNER_PAYLOAD_FIELDS
    assert isinstance(payload["tool_calls"], list)
    for call in payload["tool_calls"]:
        assert tuple(call) == _EXPECTED_CIV6_MCP_TOOL_CALL_FIELDS
        assert isinstance(call["tool"], str)
        assert isinstance(call["arguments"], dict)
        assert isinstance(call["reasoning"], str)
    return payload


def _validate_civ6_mcp_outward_observation_schema(observation: dict[str, Any]) -> None:
    """Assert the civ6-mcp observer adapter preserves the public context_observer schema."""
    assert tuple(observation) == _EXPECTED_CIV6_MCP_OBSERVER_RESULT_FIELDS
    assert isinstance(observation["situation_summary"], str)
    assert isinstance(observation["threats"], list)
    assert isinstance(observation["opportunities"], list)
    assert "raw_state" not in observation
    assert "tool_results" not in observation
    assert "planner_context" not in observation


def test_register_civ6_mcp_adapters_adds_named_slots() -> None:
    client = FakeCiv6McpClient()
    registry = LayerAdapterRegistry(include_builtins=False)

    register_civ6_mcp_adapters(registry, client)  # type: ignore[arg-type]

    available = registry.list_available()
    assert CIV6_MCP_ADAPTER_NAME in available["action_router"]
    assert CIV6_MCP_ADAPTER_NAME in available["action_planner"]
    assert CIV6_MCP_ADAPTER_NAME in available["context_observer"]
    assert CIV6_MCP_ADAPTER_NAME in available["action_executor"]


def test_registered_civ6_mcp_observer_adapter_returns_schema_compatible_observation() -> None:
    client = FakeCiv6McpClient()
    registry = LayerAdapterRegistry(include_builtins=False)
    register_civ6_mcp_adapters(registry, client)  # type: ignore[arg-type]
    adapter = registry.context_observers[CIV6_MCP_ADAPTER_NAME]
    session = LayeredSession(session_id="registered_observer", name="registered-observer")

    observation = adapter(session, Image.new("RGB", (1, 1), "black"))

    _validate_civ6_mcp_outward_observation_schema(observation)
    assert observation == {
        "situation_summary": "Turn 12 | Era Classical | Sci +14.0/t | Cul +6.0/t | Research WRITING | "
        "Civic POLITICAL_PHILOSOPHY",
        "threats": [],
        "opportunities": [],
    }
    assert client.calls == [("get_game_overview", {})]
    assert session.global_context.current_turn == 12
    assert session.global_context.current_research == "WRITING"
    assert session.high_level_context.latest_game_observation == {
        "current_turn": 12,
        "game_era": "Classical",
        "science_per_turn": 14.0,
        "culture_per_turn": 6.0,
        "current_research": "WRITING",
        "current_civic": "POLITICAL_PHILOSOPHY",
    }
    assert session.high_level_context.notes == []


def test_backend_adapter_selection_includes_civ6_mcp_observer_only_for_civ6_mcp() -> None:
    civ6_mcp_overrides = adapter_overrides_for_backend("civ6-mcp")

    assert civ6_mcp_overrides == {
        "action_router": CIV6_MCP_ADAPTER_NAME,
        "action_planner": CIV6_MCP_ADAPTER_NAME,
        "context_observer": CIV6_MCP_ADAPTER_NAME,
        "action_executor": CIV6_MCP_ADAPTER_NAME,
    }
    assert adapter_overrides_for_backend("vlm") == {}
    assert adapter_overrides_for_backend(None) == {}


def test_layered_mcp_civ6_mcp_backend_defaults_session_to_civ6_mcp_observer() -> None:
    client = FakeCiv6McpClient()
    registry = LayerAdapterRegistry(
        include_builtins=False,
        screen_capture=lambda: (Image.new("RGB", (8, 8), "black"), 8, 8, 0, 0),
    )
    register_civ6_mcp_adapters(registry, client)  # type: ignore[arg-type]
    app = LayeredComputerUseMCP(adapter_registry=registry, default_backend="civ6-mcp")

    create_result = _call_tool(app, "session_create", {"name": "default-civ6-mcp"})

    assert create_result["adapter_overrides"]["context_observer"] == CIV6_MCP_ADAPTER_NAME
    observe_result = _call_tool(app, "workflow_observe", {"session_id": create_result["session_id"]})
    assert observe_result["context"]["global_context"]["current_turn"] == 12
    assert ("get_game_overview", {}) in client.calls


def test_layered_mcp_vlm_backend_does_not_default_to_civ6_mcp_observer() -> None:
    registry = LayerAdapterRegistry(include_builtins=False)
    app = LayeredComputerUseMCP(adapter_registry=registry, default_backend="vlm")

    create_result = _call_tool(app, "session_create", {"name": "default-vlm"})

    assert create_result["adapter_overrides"] == {}
    assert CIV6_MCP_ADAPTER_NAME not in registry.list_available()["context_observer"]


def test_mcp_server_backend_flag_enables_civ6_mcp_adapters_only_for_civ6_mcp(
    monkeypatch,
) -> None:
    registries: list[Any] = []
    app_kwargs: list[dict[str, Any]] = []
    transports: list[str] = []

    class FakeRegistry:
        def __init__(self) -> None:
            self.enable_calls: list[dict[str, Any]] = []
            registries.append(self)

        def enable_civ6_mcp(self, **kwargs: Any) -> None:
            self.enable_calls.append(dict(kwargs))

    class FakeApp:
        def __init__(self, **kwargs: Any) -> None:
            self.server = self
            app_kwargs.append(dict(kwargs))

        def run(self, *, transport: str) -> None:
            transports.append(transport)

    monkeypatch.setattr(server_module, "LayerAdapterRegistry", FakeRegistry)
    monkeypatch.setattr(server_module, "LayeredComputerUseMCP", FakeApp)

    server_module.main(
        [
            "--backend",
            "civ6-mcp",
            "--civ6-mcp-path",
            "/tmp/civ6-mcp",
            "--civ6-mcp-launcher",
            "python",
        ]
    )
    server_module.main(["--backend", "vlm"])

    assert registries[0].enable_calls == [
        {"install_path": "/tmp/civ6-mcp", "launcher": "python"},
    ]
    assert app_kwargs[0]["adapter_registry"] is registries[0]
    assert app_kwargs[0]["default_backend"] is BackendKind.CIV6_MCP
    assert registries[1].enable_calls == []
    assert app_kwargs[1]["adapter_registry"] is registries[1]
    assert app_kwargs[1]["default_backend"] is BackendKind.VLM
    assert transports == ["stdio", "stdio"]


def test_civ6_mcp_planner_adapter_is_exported_from_mcp_registration_surface() -> None:
    import civStation.mcp as mcp

    assert "CIV6_MCP_ADAPTER_NAME" in mcp.__all__
    assert "make_civ6_mcp_planner_adapter" in mcp.__all__
    assert "register_civ6_mcp_adapters" in mcp.__all__
    assert mcp.CIV6_MCP_ADAPTER_NAME == CIV6_MCP_ADAPTER_NAME
    assert mcp.make_civ6_mcp_planner_adapter is make_civ6_mcp_planner_adapter
    assert mcp.register_civ6_mcp_adapters is register_civ6_mcp_adapters


def test_civ6_mcp_router_adapter_preserves_outward_route_schema() -> None:
    client = FakeCiv6McpClient()
    router = make_civ6_mcp_router_adapter(client)  # type: ignore[arg-type]

    route = router(object(), Image.new("RGB", (1, 1), "black"))

    assert set(route) == {"primitive", "reasoning"}
    assert route["primitive"] == "civ6_mcp"
    assert route["reasoning"]
    assert client.calls == []


def test_civ6_mcp_planner_result_encoding_preserves_agent_action_envelope() -> None:
    result = PlannerResult(
        tool_calls=[
            ToolCall(
                tool="set_research",
                arguments={"tech_or_civic": "MATHEMATICS"},
                reasoning="Unlock stronger campuses.",
            )
        ],
        raw_response='{"tool_calls":[]}',
        parsed_payload={"tool_calls": []},
    )

    payload = encode_civ6_mcp_planner_result(result)

    assert set(payload) == {"tool_calls"}
    assert payload["tool_calls"] == [
        {
            "tool": "set_research",
            "arguments": {"tech_or_civic": "MATHEMATICS"},
            "reasoning": "Unlock stronger campuses.",
        }
    ]


def test_civ6_mcp_planner_adapter_refuses_vlm_primitive_before_provider_creation() -> None:
    client = FakeCiv6McpClient()
    provider_created = False

    def provider_factory(provider_name: str, model: str | None) -> FakeProvider:  # noqa: ARG001
        nonlocal provider_created
        provider_created = True
        return FakeProvider()

    adapter = make_civ6_mcp_planner_adapter(client, provider_factory=provider_factory)  # type: ignore[arg-type]
    session = object()

    try:
        adapter(session, Image.new("RGB", (1, 1), "black"), "click")
    except ValueError as exc:
        assert "cannot plan VLM/computer-use primitives" in str(exc)
    else:
        raise AssertionError("expected civ6-mcp planner adapter to reject VLM primitive")

    assert provider_created is False
    assert client.calls == []


def test_civ6_mcp_observer_adapter_preserves_outward_observation_schema_and_syncs_fields() -> None:
    client = FakeCiv6McpClient()
    adapter = make_civ6_mcp_observer_adapter(client)  # type: ignore[arg-type]
    session = LayeredSession(session_id="session_schema", name="schema")

    observation = adapter(session, Image.new("RGB", (1, 1), "black"))

    _validate_civ6_mcp_outward_observation_schema(observation)
    assert observation == {
        "situation_summary": "Turn 12 | Era Classical | Sci +14.0/t | Cul +6.0/t | Research WRITING | "
        "Civic POLITICAL_PHILOSOPHY",
        "threats": [],
        "opportunities": [],
    }
    assert session.global_context.current_turn == 12
    assert session.global_context.game_era == "Classical"
    assert session.global_context.current_research == "WRITING"
    assert session.high_level_context.latest_game_observation == {
        "current_turn": 12,
        "game_era": "Classical",
        "science_per_turn": 14.0,
        "culture_per_turn": 6.0,
        "current_research": "WRITING",
        "current_civic": "POLITICAL_PHILOSOPHY",
    }
    assert session.high_level_context.notes == []


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
    payload = _validate_civ6_mcp_outward_plan_schema(plan_result)
    assert "tool_calls" not in plan_result
    assert set(plan_result) == {"session_id", "capture", "primitive", "action"}
    action = plan_result["action"]
    assert action["action"] == CIV6_MCP_TOOL_PLAN_ACTION
    assert "text" in action
    assert "previous call: get_game_overview" in provider.prompts[-1]
    assert payload["tool_calls"][0]["tool"] == "set_research"

    execute_result = _call_tool(app, "action_execute", {"session_id": session_id, "action": action})
    assert execute_result["executed"] is True
    assert execute_result["results"][0]["success"] is True
    assert ("set_research", {"tech_or_civic": "MATHEMATICS"}) in client.calls


def test_civ6_mcp_planner_adapter_preserves_workflow_decide_plan_schema() -> None:
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
            "name": "civ6-mcp-workflow-schema",
            "adapter_overrides": {
                "action_router": CIV6_MCP_ADAPTER_NAME,
                "action_planner": CIV6_MCP_ADAPTER_NAME,
            },
        },
    )

    decide_result = _call_tool(
        app,
        "workflow_decide",
        {
            "session_id": create_result["session_id"],
            "recent_actions_override": "schema check",
        },
    )

    assert tuple(decide_result) == ("session_id", "capture", "route", "plan")
    assert tuple(decide_result["route"]) == ("primitive", "reasoning")
    assert decide_result["route"]["primitive"] == CIV6_MCP_ADAPTER_NAME
    payload = _validate_civ6_mcp_serialized_plan_schema(decide_result["plan"])
    assert payload == {
        "tool_calls": [
            {
                "tool": "set_research",
                "arguments": {"tech_or_civic": "MATHEMATICS"},
                "reasoning": "Unlock stronger campuses.",
            },
            {
                "tool": "end_turn",
                "arguments": {
                    "tactical": "Queued research before ending the turn.",
                    "strategic": "Mathematics supports the science plan.",
                    "tooling": "Used civ6-mcp tool calls only.",
                    "planning": "No further blockers were present.",
                    "hypothesis": "Next turn should continue toward campuses.",
                },
                "reasoning": "Finish the turn after setting research.",
            },
        ]
    }


def test_civ6_mcp_executor_adapter_resolves_free_form_action_type_without_vlm_dispatch(
    monkeypatch,
) -> None:
    def forbidden_execute_action(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("civ6-mcp adapter must not dispatch through the VLM/computer-use pipeline")

    monkeypatch.setattr("civStation.utils.screen.execute_action", forbidden_execute_action, raising=False)

    client = FakeCiv6McpClient()
    registry = LayerAdapterRegistry(
        include_builtins=False,
        screen_capture=lambda: (Image.new("RGB", (8, 8), "black"), 8, 8, 0, 0),
    )
    register_civ6_mcp_adapters(registry, client)  # type: ignore[arg-type]
    app = LayeredComputerUseMCP(adapter_registry=registry)

    create_result = _call_tool(
        app,
        "session_create",
        {
            "name": "civ6-mcp-free-form",
            "runtime": {"execution_mode": "live", "require_execute_confirmation": False},
            "adapter_overrides": {
                "action_executor": CIV6_MCP_ADAPTER_NAME,
            },
        },
    )
    action = {
        "action": CIV6_MCP_TOOL_CALL_ACTION,
        "text": json.dumps(
            {
                "type": "choose_research",
                "tech": "MATHEMATICS",
                "reasoning": "Unlock better campuses.",
            }
        ),
        "reasoning": "free-form direct dispatch",
    }

    execute_result = _call_tool(
        app,
        "action_execute",
        {"session_id": create_result["session_id"], "action": action},
    )

    assert execute_result["executed"] is True
    assert execute_result["results"][0]["tool"] == "set_research"
    assert ("set_research", {"tech_or_civic": "MATHEMATICS"}) in client.calls


def test_civ6_mcp_executor_uses_existing_action_execute_schema_without_vlm_components() -> None:
    def forbidden_provider_factory(provider_name: str, model: str | None) -> object:  # noqa: ARG001
        raise AssertionError("civ6-mcp execution must not create VLM providers")

    def forbidden_screen_capture() -> tuple[Image.Image, int, int, int, int]:
        raise AssertionError("civ6-mcp execution must not require screen capture")

    def forbidden_low_level_executor(
        action: AgentAction,
        screen_w: int,
        screen_h: int,
        normalizing_range: int,
        x_offset: int,
        y_offset: int,
    ) -> None:
        raise AssertionError("civ6-mcp execution must not dispatch through VLM/computer-use execution")

    baseline_app = LayeredComputerUseMCP(adapter_registry=LayerAdapterRegistry(include_builtins=False))
    baseline_execute_schema = dict(baseline_app.server._tool_manager._tools["action_execute"].parameters)

    client = FakeCiv6McpClient()
    registry = LayerAdapterRegistry(
        include_builtins=False,
        provider_factory=forbidden_provider_factory,
        screen_capture=forbidden_screen_capture,
        low_level_executor=forbidden_low_level_executor,
    )
    register_civ6_mcp_adapters(registry, client)  # type: ignore[arg-type]
    app = LayeredComputerUseMCP(adapter_registry=registry, default_backend="civ6-mcp")

    execute_schema = app.server._tool_manager._tools["action_execute"].parameters
    assert execute_schema == baseline_execute_schema
    assert tuple(execute_schema["properties"]) == ("session_id", "action", "capture", "confirm_execute")
    assert execute_schema["properties"]["action"] == {
        "additionalProperties": True,
        "title": "Action",
        "type": "object",
    }
    assert "civ6_mcp" not in execute_schema["properties"]
    assert "primitive_name" not in execute_schema["properties"]

    create_result = _call_tool(
        app,
        "session_create",
        {
            "name": "civ6-mcp-execute-schema",
            "runtime": {"execution_mode": "live", "require_execute_confirmation": False},
        },
    )
    action = {
        "action": CIV6_MCP_TOOL_CALL_ACTION,
        "text": json.dumps(
            {
                "tool": "set_research",
                "arguments": {"tech_or_civic": "MATHEMATICS"},
                "reasoning": "Use the unchanged AgentAction transport envelope.",
            }
        ),
        "reasoning": "schema-preserving civ6-mcp execution",
    }

    execute_result = _call_tool(
        app,
        "action_execute",
        {
            "session_id": create_result["session_id"],
            "action": action,
            "capture": {"screen_w": 0, "screen_h": 0, "x_offset": 0, "y_offset": 0},
        },
    )

    assert execute_result["executed"] is True
    assert execute_result["results"] == [
        {
            "tool": "set_research",
            "success": True,
            "classification": "ok",
            "text": "Research set to MATHEMATICS.",
            "error": "",
        }
    ]
    assert client.calls == [("set_research", {"tech_or_civic": "MATHEMATICS"})]
    assert registry.list_available() == {
        "action_router": [CIV6_MCP_ADAPTER_NAME],
        "action_planner": [CIV6_MCP_ADAPTER_NAME],
        "context_observer": [CIV6_MCP_ADAPTER_NAME],
        "strategy_refiner": [],
        "action_executor": [CIV6_MCP_ADAPTER_NAME],
    }


def test_civ6_mcp_executor_adapter_rechecks_agent_gate_between_tool_calls() -> None:
    class StopAfterFirstToolClient(FakeCiv6McpClient):
        def __init__(self) -> None:
            super().__init__()
            self.gate: Any | None = None

        def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
            text = super().call_tool(name, arguments)
            if len(self.calls) == 1 and self.gate is not None:
                self.gate.stop()
            return text

    client = StopAfterFirstToolClient()
    registry = LayerAdapterRegistry(
        include_builtins=False,
        screen_capture=lambda: (Image.new("RGB", (8, 8), "black"), 8, 8, 0, 0),
    )
    register_civ6_mcp_adapters(registry, client)  # type: ignore[arg-type]
    app = LayeredComputerUseMCP(adapter_registry=registry, default_backend="civ6-mcp")
    create_result = _call_tool(
        app,
        "session_create",
        {
            "name": "civ6-mcp-stop-between-calls",
            "runtime": {"execution_mode": "live", "require_execute_confirmation": False},
        },
    )
    session = app.sessions.get(create_result["session_id"])
    client.gate = session.agent_gate
    action = {
        "action": CIV6_MCP_TOOL_PLAN_ACTION,
        "text": json.dumps(
            {
                "tool_calls": [
                    {"tool": "get_game_overview", "arguments": {}, "reasoning": "Observe current game state."},
                    {
                        "tool": "set_research",
                        "arguments": {"tech_or_civic": "MATHEMATICS"},
                        "reasoning": "Must not dispatch after stop.",
                    },
                ]
            }
        ),
        "reasoning": "multi-tool sequence",
    }

    execute_result = _call_tool(
        app,
        "action_execute",
        {"session_id": create_result["session_id"], "action": action},
    )

    assert execute_result["executed"] is True
    assert execute_result["tool_call_count"] == 1
    assert execute_result["results"][0]["tool"] == "get_game_overview"
    assert client.calls == [("get_game_overview", {})]


def test_civ6_mcp_executor_adapter_rejects_vlm_action_envelope_even_with_tool_payload() -> None:
    client = FakeCiv6McpClient()
    registry = LayerAdapterRegistry(
        include_builtins=False,
        screen_capture=lambda: (Image.new("RGB", (8, 8), "black"), 8, 8, 0, 0),
    )
    register_civ6_mcp_adapters(registry, client)  # type: ignore[arg-type]
    app = LayeredComputerUseMCP(adapter_registry=registry)
    create_result = _call_tool(
        app,
        "session_create",
        {
            "name": "civ6-mcp-reject-vlm-action",
            "runtime": {"execution_mode": "live", "require_execute_confirmation": False},
            "adapter_overrides": {"action_executor": CIV6_MCP_ADAPTER_NAME},
        },
    )
    action = {
        "action": "click",
        "x": 100,
        "y": 200,
        "text": json.dumps(
            {
                "tool_calls": [
                    {
                        "tool": "set_research",
                        "arguments": {"tech_or_civic": "MATHEMATICS"},
                        "reasoning": "This must not bypass the backend boundary.",
                    }
                ]
            }
        ),
    }

    execute_result = _call_tool(
        app,
        "action_execute",
        {"session_id": create_result["session_id"], "action": action},
    )

    assert execute_result["executed"] is False
    assert execute_result["blocked"] is True
    assert "only accepts civ6-mcp AgentAction envelopes" in execute_result["reason"]
    assert ("set_research", {"tech_or_civic": "MATHEMATICS"}) not in client.calls
