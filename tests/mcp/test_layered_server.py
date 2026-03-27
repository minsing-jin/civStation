from __future__ import annotations

import asyncio
import json
from pathlib import Path

from PIL import Image

from civStation.agent.modules.strategy.strategy_schemas import StructuredStrategy, VictoryType
from civStation.utils.llm_provider.parser import AgentAction


def _build_test_image(path: Path) -> Path:
    image = Image.new("RGB", (32, 32), color=(12, 34, 56))
    image.save(path)
    return path


def _build_fake_registry():
    from civStation.mcp.runtime import LayerAdapterRegistry

    execution_log: list[dict[str, object]] = []

    def fake_router(session, pil_image):
        assert pil_image.size == (32, 32)
        return {
            "primitive": "policy_primitive",
            "reasoning": f"fake route for {session.session_id}",
        }

    def fake_planner(session, pil_image, primitive_name, *, strategy_override=None, recent_actions_override=None):
        assert pil_image.size == (32, 32)
        assert primitive_name == "policy_primitive"
        return AgentAction(
            action="click",
            coord_space="normalized",
            x=111,
            y=222,
            reasoning=strategy_override or recent_actions_override or "fake planner",
            task_status="in_progress",
        )

    def fake_context_observer(session, pil_image):
        assert pil_image.size == (32, 32)
        return {
            "situation_summary": "mocked context summary",
            "threats": ["barbarian nearby"],
            "opportunities": ["free settler"],
        }

    def fake_strategy_refiner(session, raw_input):
        return StructuredStrategy(
            text=f"refined::{raw_input}",
            victory_goal=VictoryType.SCIENCE,
            current_phase="mid_development",
            primitive_directives={"policy_primitive": "slot science cards"},
        )

    def fake_executor(session, action, capture):
        execution_log.append(
            {
                "session_id": session.session_id,
                "action": action.action,
                "x": action.x,
                "y": action.y,
                "capture": capture,
            }
        )
        return {
            "executed": True,
            "action": action.action,
            "x": action.x,
            "y": action.y,
        }

    registry = LayerAdapterRegistry(
        action_routers={"fake": fake_router},
        action_planners={"fake": fake_planner},
        context_observers={"fake": fake_context_observer},
        strategy_refiners={"fake": fake_strategy_refiner},
        action_executors={"fake": fake_executor},
    )
    return registry, execution_log


def _build_app():
    from civStation.mcp.server import LayeredComputerUseMCP

    registry, execution_log = _build_fake_registry()
    app = LayeredComputerUseMCP(adapter_registry=registry)
    return app, execution_log


def _call_tool(app, name: str, arguments: dict[str, object]):
    result = asyncio.run(app.server.call_tool(name, arguments))
    if isinstance(result, tuple) and len(result) == 2:
        return result[1]
    return result


def _read_resource(app, uri: str):
    return list(asyncio.run(app.server.read_resource(uri)))


def _get_prompt(app, name: str, arguments: dict[str, object]):
    return asyncio.run(app.server.get_prompt(name, arguments))


def test_session_layer_state_roundtrip():
    app, _ = _build_app()

    create_result = _call_tool(
        app,
        "session_create",
        {
            "name": "demo",
            "adapter_overrides": {
                "action_router": "fake",
                "action_planner": "fake",
                "context_observer": "fake",
                "strategy_refiner": "fake",
                "action_executor": "fake",
            },
        },
    )
    session_id = create_result["session_id"]

    _call_tool(
        app,
        "context_update",
        {
            "session_id": session_id,
            "patch": {
                "global_context": {"current_turn": 42, "game_era": "Medieval"},
                "primitive_context": {"selected_unit_info": {"name": "warrior", "moves": 2}},
                "high_level_context": {"active_threats": ["enemy scout"]},
            },
        },
    )
    context_result = _call_tool(app, "context_get", {"session_id": session_id})
    assert context_result["global_context"]["current_turn"] == 42
    assert context_result["primitive_context"]["selected_unit_info"]["name"] == "warrior"
    assert context_result["high_level_context"]["active_threats"] == ["enemy scout"]

    _call_tool(
        app,
        "strategy_set",
        {
            "session_id": session_id,
            "strategy": {
                "text": "focus on science",
                "victory_goal": "science",
                "current_phase": "mid_development",
                "primitive_directives": {"policy_primitive": "science cards first"},
            },
        },
    )
    strategy_result = _call_tool(app, "strategy_get", {"session_id": session_id})
    assert strategy_result["strategy"]["text"] == "focus on science"
    assert strategy_result["strategy"]["primitive_directives"]["policy_primitive"] == "science cards first"

    _call_tool(
        app,
        "memory_start_task",
        {
            "session_id": session_id,
            "primitive_name": "policy_primitive",
            "max_steps": 7,
            "enable_policy_state": True,
        },
    )
    _call_tool(
        app,
        "memory_update",
        {
            "session_id": session_id,
            "patch": {
                "current_stage": "select_policy",
                "task_hitl_directive": "slot science card",
                "policy_state": {"selected_tab_name": "경제", "enabled": True},
            },
        },
    )
    memory_result = _call_tool(app, "memory_get", {"session_id": session_id})
    assert memory_result["memory"]["primitive_name"] == "policy_primitive"
    assert memory_result["memory"]["current_stage"] == "select_policy"
    assert memory_result["memory"]["policy_state"]["selected_tab_name"] == "경제"

    export_result = _call_tool(app, "session_export", {"session_id": session_id})
    clone_result = _call_tool(app, "session_import", {"state": export_result["state"], "name": "clone"})
    clone_context = _call_tool(app, "context_get", {"session_id": clone_result["session_id"]})
    clone_memory = _call_tool(app, "memory_get", {"session_id": clone_result["session_id"]})

    assert clone_context["global_context"]["current_turn"] == 42
    assert clone_context["primitive_context"]["selected_unit_info"]["name"] == "warrior"
    assert clone_memory["memory"]["current_stage"] == "select_policy"


def test_action_workflow_and_hitl_tools(tmp_path: Path):
    app, execution_log = _build_app()
    image_path = _build_test_image(tmp_path / "screen.png")

    create_result = _call_tool(
        app,
        "session_create",
        {
            "name": "workflow",
            "runtime": {
                "execution_mode": "live",
                "require_execute_confirmation": False,
            },
            "adapter_overrides": {
                "action_router": "fake",
                "action_planner": "fake",
                "context_observer": "fake",
                "strategy_refiner": "fake",
                "action_executor": "fake",
            },
        },
    )
    session_id = create_result["session_id"]

    observe_result = _call_tool(app, "workflow_observe", {"session_id": session_id, "image_path": str(image_path)})
    assert observe_result["context"]["high_level_context"]["notes"][-1] == "mocked context summary"

    route_result = _call_tool(app, "action_route", {"session_id": session_id, "image_path": str(image_path)})
    assert route_result["primitive"] == "policy_primitive"

    plan_result = _call_tool(
        app,
        "action_plan",
        {
            "session_id": session_id,
            "image_path": str(image_path),
            "primitive_name": "policy_primitive",
            "strategy_override": "prefer science cards",
        },
    )
    assert plan_result["action"]["action"] == "click"
    assert plan_result["action"]["x"] == 111

    decide_result = _call_tool(app, "workflow_decide", {"session_id": session_id, "image_path": str(image_path)})
    assert decide_result["route"]["primitive"] == "policy_primitive"
    assert decide_result["plan"]["action"]["y"] == 222

    execute_result = _call_tool(
        app,
        "action_execute",
        {
            "session_id": session_id,
            "action": plan_result["action"],
            "capture": {"screen_w": 1920, "screen_h": 1080, "x_offset": 0, "y_offset": 0},
        },
    )
    assert execute_result["executed"] is True
    assert execution_log[-1]["action"] == "click"

    step_result = _call_tool(
        app,
        "workflow_step",
        {
            "session_id": session_id,
            "image_path": str(image_path),
            "execute": True,
        },
    )
    assert step_result["route"]["primitive"] == "policy_primitive"
    assert step_result["execution"]["executed"] is True

    refine_result = _call_tool(
        app,
        "strategy_refine",
        {"session_id": session_id, "raw_input": "science and campuses first"},
    )
    assert refine_result["strategy"]["text"] == "refined::science and campuses first"

    _call_tool(
        app,
        "hitl_send",
        {
            "session_id": session_id,
            "directive_type": "change_strategy",
            "payload": "switch to defense",
            "source": "test",
        },
    )
    _call_tool(
        app,
        "hitl_send",
        {
            "session_id": session_id,
            "directive_type": "start",
            "payload": "",
            "source": "test",
        },
    )
    hitl_result = _call_tool(app, "hitl_status", {"session_id": session_id})
    assert hitl_result["agent_state"] == "running"
    assert hitl_result["queued_directives"][0]["payload"] == "switch to defense"


def test_execution_guard_transport_config_and_install_resources(tmp_path: Path):
    from civStation.mcp.install_client_assets import (
        create_argument_parser as create_install_argument_parser,
    )
    from civStation.mcp.install_client_assets import (
        render_client_template,
    )
    from civStation.mcp.server import LayeredComputerUseMCP, create_argument_parser

    registry, execution_log = _build_fake_registry()
    app = LayeredComputerUseMCP(
        adapter_registry=registry,
        host="0.0.0.0",
        port=9100,
        streamable_http_path="/api/mcp",
        json_response=True,
        stateless_http=True,
    )
    image_path = _build_test_image(tmp_path / "screen.png")

    assert app.server.settings.host == "0.0.0.0"
    assert app.server.settings.port == 9100
    assert app.server.settings.streamable_http_path == "/api/mcp"
    assert app.server.settings.json_response is True
    assert app.server.settings.stateless_http is True

    parser = create_argument_parser()
    args = parser.parse_args(
        [
            "--transport",
            "streamable-http",
            "--host",
            "0.0.0.0",
            "--port",
            "9100",
            "--streamable-http-path",
            "/api/mcp",
            "--json-response",
            "--stateless-http",
        ]
    )
    assert args.transport == "streamable-http"
    assert args.host == "0.0.0.0"
    assert args.port == 9100
    assert args.streamable_http_path == "/api/mcp"
    assert args.json_response is True
    assert args.stateless_http is True

    install_parser = create_install_argument_parser()
    install_args = install_parser.parse_args(["--client", "claude-code", "--output", "/tmp/test.mcp.json", "--write"])
    assert install_args.client == "claude-code"
    assert install_args.output == "/tmp/test.mcp.json"
    assert install_args.write is True

    create_result = _call_tool(
        app,
        "session_create",
        {
            "name": "guarded",
            "adapter_overrides": {
                "action_router": "fake",
                "action_planner": "fake",
                "context_observer": "fake",
                "strategy_refiner": "fake",
                "action_executor": "fake",
            },
        },
    )
    session_id = create_result["session_id"]
    plan_result = _call_tool(
        app,
        "action_plan",
        {
            "session_id": session_id,
            "image_path": str(image_path),
            "primitive_name": "policy_primitive",
        },
    )

    dry_run_result = _call_tool(
        app,
        "action_execute",
        {
            "session_id": session_id,
            "action": plan_result["action"],
        },
    )
    assert dry_run_result["executed"] is False
    assert dry_run_result["blocked"] is True
    assert dry_run_result["mode"] == "dry_run"
    assert execution_log == []

    _call_tool(
        app,
        "session_config_update",
        {
            "session_id": session_id,
            "runtime_patch": {
                "execution_mode": "live",
                "require_execute_confirmation": True,
            },
        },
    )

    confirm_blocked = _call_tool(
        app,
        "action_execute",
        {
            "session_id": session_id,
            "action": plan_result["action"],
        },
    )
    assert confirm_blocked["executed"] is False
    assert confirm_blocked["blocked"] is True
    assert confirm_blocked["requires_confirmation"] is True
    assert execution_log == []

    confirmed_result = _call_tool(
        app,
        "action_execute",
        {
            "session_id": session_id,
            "action": plan_result["action"],
            "confirm_execute": True,
        },
    )
    assert confirmed_result["executed"] is True
    assert execution_log[-1]["action"] == "click"

    codex_resource = _read_resource(app, "civ6://install/codex-config")
    assert "[mcp_servers.civstation-layered]" in codex_resource[0].content
    assert 'command = ".venv/bin/python"' in codex_resource[0].content
    assert '"civStation.mcp.server"' in codex_resource[0].content
    assert '"stdio"' in codex_resource[0].content

    claude_resource = _read_resource(app, "civ6://install/claude-code-project-mcp-json")
    assert '"mcpServers"' in claude_resource[0].content
    assert '"civstation-layered"' in claude_resource[0].content
    assert "civStation.mcp.server" in claude_resource[0].content

    http_resource = _read_resource(app, "civ6://install/http-client-example")
    assert "http://127.0.0.1:8000/mcp" in http_resource[0].content

    contracts_resource = _read_resource(app, "civ6://contracts/layers")
    assert '"strategy_context"' in contracts_resource[0].content
    assert '"background"' in contracts_resource[0].content
    assert '"primitive_action"' in contracts_resource[0].content
    assert '"main_thread"' in contracts_resource[0].content

    setup_prompt = _get_prompt(app, "client_setup_workflow", {"client": "claude-code"})
    setup_text = setup_prompt.messages[0].content.text
    assert "civ6://install/claude-code-project-mcp-json" in setup_text
    assert "session_create" in setup_text

    codex_template = render_client_template("codex")
    claude_template = render_client_template("claude-code")
    assert "[mcp_servers.civstation-layered]" in codex_template
    assert 'command = ".venv/bin/python"' in codex_template
    assert '"mcpServers"' in claude_template
    assert '"civstation-layered"' in claude_template


def test_resources_and_prompts_surface_session_information(tmp_path: Path):
    app, _ = _build_app()
    image_path = _build_test_image(tmp_path / "screen.png")

    create_result = _call_tool(
        app,
        "session_create",
        {
            "name": "resources",
            "adapter_overrides": {
                "action_router": "fake",
                "action_planner": "fake",
                "context_observer": "fake",
                "strategy_refiner": "fake",
                "action_executor": "fake",
            },
        },
    )
    session_id = create_result["session_id"]
    _call_tool(app, "workflow_observe", {"session_id": session_id, "image_path": str(image_path)})

    sessions_resource = _read_resource(app, "civ6://sessions")
    assert any(session_id in getattr(item, "content", "") for item in sessions_resource)

    context_resource = _read_resource(app, f"civ6://sessions/{session_id}/context")
    context_payload = json.loads(context_resource[0].content)
    assert context_payload["session_id"] == session_id
    assert context_payload["context"]["high_level_context"]["notes"][-1] == "mocked context summary"

    prompt_result = _get_prompt(app, "strategy_only_workflow", {"session_id": session_id})
    prompt_text = prompt_result.messages[0].content.text
    assert "strategy_refine" in prompt_text
    assert session_id in prompt_text
