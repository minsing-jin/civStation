"""Isolation tests for the civ6-mcp observer backend."""

from __future__ import annotations

import builtins
import importlib
import sys
from typing import Any

import pytest

_FORBIDDEN_VLM_RUNTIME_MODULE_PREFIXES = (
    "civStation.agent.turn_executor",
    "civStation.agent.modules.context.context_updater",
    "civStation.agent.modules.context.macro_turn_manager",
    "civStation.agent.modules.context.turn_detector",
    "civStation.agent.modules.primitive.multi_step_process",
    "civStation.agent.modules.router.primitive_registry",
    "civStation.utils.image_pipeline",
    "civStation.utils.llm_provider",
    "civStation.utils.screen",
)

_CIV6_MCP_OBSERVER_MODULE_PREFIXES = (
    "civStation.agent.modules.backend.civ6_mcp.client",
    "civStation.agent.modules.backend.civ6_mcp.observation_schema",
    "civStation.agent.modules.backend.civ6_mcp.observer",
    "civStation.agent.modules.backend.civ6_mcp.operations",
    "civStation.agent.modules.backend.civ6_mcp.response",
    "civStation.agent.modules.backend.civ6_mcp.state_parser",
)


class _ObservationOnlyClient:
    def __init__(self, responses: dict[str, str]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def tool_names(self) -> set[str]:
        return set(self._responses)

    def has_tool(self, name: str) -> bool:
        return name in self._responses

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        request_arguments = dict(arguments or {})
        self.calls.append((name, request_arguments))
        return self._responses[name]


class _RecordingContextManager:
    def __init__(self) -> None:
        self.global_updates: list[dict[str, Any]] = []
        self.game_observation_updates: list[dict[str, Any]] = []

    def update_global_context(self, **kwargs: Any) -> None:
        self.global_updates.append(dict(kwargs))

    def update_game_observation(self, **kwargs: Any) -> None:
        self.game_observation_updates.append(dict(kwargs))


def _unload_observer_and_forbidden_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    prefixes = (*_CIV6_MCP_OBSERVER_MODULE_PREFIXES, *_FORBIDDEN_VLM_RUNTIME_MODULE_PREFIXES)
    for module_name in list(sys.modules):
        if module_name.startswith(prefixes):
            monkeypatch.delitem(sys.modules, module_name, raising=False)


def _loaded_forbidden_modules() -> set[str]:
    return {
        module_name for module_name in sys.modules if module_name.startswith(_FORBIDDEN_VLM_RUNTIME_MODULE_PREFIXES)
    }


def _is_forbidden_import(name: str) -> bool:
    return name.startswith(_FORBIDDEN_VLM_RUNTIME_MODULE_PREFIXES)


def test_observe_does_not_import_instantiate_or_call_vlm_computer_use_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _unload_observer_and_forbidden_modules(monkeypatch)
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, Any] | None = None,  # noqa: A002
        locals: dict[str, Any] | None = None,  # noqa: A002
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if level == 0 and _is_forbidden_import(name):
            raise AssertionError(f"Civ6McpObserver touched forbidden VLM/computer-use import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    observer_module = importlib.import_module("civStation.agent.modules.backend.civ6_mcp.observer")
    assert _loaded_forbidden_modules() == set()

    client = _ObservationOnlyClient(
        {
            "get_game_overview": "\n".join(
                [
                    "Turn: 23",
                    "Era: Renaissance Era",
                    "Science: +42.0/turn",
                    "Culture: +18.5/turn",
                    "Researching: PRINTING",
                ]
            )
        }
    )
    context_manager = _RecordingContextManager()
    observer = observer_module.Civ6McpObserver(
        client=client,
        context_manager=context_manager,
        observe_tools=("get_game_overview",),
    )

    bundle = observer.observe()

    assert client.calls == [("get_game_overview", {})]
    assert bundle.overview.current_turn == 23
    assert context_manager.global_updates == [
        {
            "current_turn": 23,
            "game_era": "Renaissance",
            "science_per_turn": 42.0,
            "culture_per_turn": 18.5,
            "current_research": "PRINTING",
        }
    ]
    assert context_manager.game_observation_updates == [
        {
            "situation_summary": "Turn 23 | Era Renaissance | Sci +42.0/t | Cul +18.5/t | Research PRINTING",
            "observation_fields": {
                "current_turn": 23,
                "game_era": "Renaissance",
                "science_per_turn": 42.0,
                "culture_per_turn": 18.5,
                "current_research": "PRINTING",
            },
        }
    ]
    assert _loaded_forbidden_modules() == set()
