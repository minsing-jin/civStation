from __future__ import annotations

import asyncio
import json
from pathlib import Path

from PIL import Image

from computer_use_test.agent.modules.strategy.strategy_schemas import StructuredStrategy, VictoryType
from computer_use_test.utils.llm_provider.parser import AgentAction


def _build_test_image(path: Path) -> Path:
    image = Image.new("RGB", (32, 32), color=(12, 34, 56))
    image.save(path)
    return path


def _build_fake_registry():
    from computer_use_test.mcp.runtime import LayerAdapterRegistry

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
    from computer_use_test.mcp.server import LayeredComputerUseMCP

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
