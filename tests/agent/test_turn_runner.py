import io
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from civStation.agent import turn_runner
from civStation.agent.modules.backend import BackendKind, parse_backend_kind

_CONCRETE_VLM_PROVIDER_MODULES = {
    "civStation.utils.llm_provider.anthropic_computer",
    "civStation.utils.llm_provider.claude",
    "civStation.utils.llm_provider.gemini",
    "civStation.utils.llm_provider.gpt",
    "civStation.utils.llm_provider.openai_computer",
}


def _unload_concrete_vlm_provider_modules(monkeypatch):
    for module_name in list(sys.modules):
        if module_name in _CONCRETE_VLM_PROVIDER_MODULES:
            monkeypatch.delitem(sys.modules, module_name, raising=False)


def _loaded_concrete_vlm_provider_modules() -> set[str]:
    return _CONCRETE_VLM_PROVIDER_MODULES.intersection(sys.modules)


def test_turn_runner_help_does_not_import_concrete_vlm_provider_modules(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    _unload_concrete_vlm_provider_modules(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        turn_runner.parse_args(["--help"])

    assert exc_info.value.code == 0
    assert "Run Civilization VI AI Agent" in capsys.readouterr().out
    assert _loaded_concrete_vlm_provider_modules() == set()


def test_parse_args_defaults_to_vlm_backend(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(turn_runner, "get_available_providers", lambda: {"gemini": object()})

    args = turn_runner.parse_args([])

    assert args.backend == "vlm"
    assert args.civ6_mcp_path is None
    assert args.civ6_mcp_launcher is None


def test_parse_args_default_backend_selects_vlm(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(turn_runner, "get_available_providers", lambda: {"gemini": object()})

    args = turn_runner.parse_args([])

    assert parse_backend_kind(args.backend) is BackendKind.VLM


def test_parse_args_accepts_civ6_mcp_backend_options(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(turn_runner, "get_available_providers", lambda: {"gemini": object()})
    mcp_path = tmp_path / "civ6-mcp"

    args = turn_runner.parse_args(
        [
            "--provider",
            "gemini",
            "--backend",
            "civ6-mcp",
            "--civ6-mcp-path",
            str(mcp_path),
            "--civ6-mcp-launcher",
            "python",
        ]
    )

    assert args.backend == "civ6-mcp"
    assert args.civ6_mcp_path == str(mcp_path)
    assert args.civ6_mcp_launcher == "python"


def test_parse_args_civ6_mcp_options_do_not_import_backend_implementation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(turn_runner, "get_available_providers", lambda: {"gemini": object()})
    backend_module_prefix = "civStation.agent.modules.backend.civ6_mcp"
    for module_name in list(sys.modules):
        if module_name.startswith(backend_module_prefix):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    args = turn_runner.parse_args(
        [
            "--backend",
            "civ6-mcp",
            "--civ6-mcp-path",
            str(tmp_path / "civ6-mcp"),
            "--civ6-mcp-launcher",
            "uv",
        ]
    )

    assert args.backend == "civ6-mcp"
    assert not any(module_name.startswith(backend_module_prefix) for module_name in sys.modules)


def test_civ6_mcp_options_do_not_implicitly_select_civ6_mcp_backend(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(turn_runner, "get_available_providers", lambda: {"gemini": object()})
    mcp_path = tmp_path / "civ6-mcp"

    args = turn_runner.parse_args(
        [
            "--civ6-mcp-path",
            str(mcp_path),
            "--civ6-mcp-launcher",
            "python",
        ]
    )

    assert args.backend == "vlm"
    assert parse_backend_kind(args.backend) is BackendKind.VLM
    assert args.civ6_mcp_path == str(mcp_path)
    assert args.civ6_mcp_launcher == "python"


class _DummyRunLogSession:
    def __init__(self):
        self.path = Path(".tmp/civStation/turn_runner_latest.log")
        self.closed = False

    def close(self):
        self.closed = True


class _DummyScreenshotTrajectorySession:
    def __init__(self):
        self.path = Path(".tmp/civStation/screenshot_trajectories/test-run")
        self.closed = False

    def close(self):
        self.closed = True


class _DummyWorker:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class _DummyRichLoggerInstance:
    def __init__(self):
        self.started = 0
        self.stopped = 0

    def start_live(self):
        self.started += 1

    def stop_live(self):
        self.stopped += 1


class _DummyAgentState:
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"


class _DummyAgentGate:
    def __init__(self, command_queue):
        self.command_queue = command_queue
        self.is_stopped = False
        self.states = []

    def set_state(self, state):
        self.states.append(state)

    def wait_for_start(self):
        return None


_OMIT_BACKEND = object()


def _patch_image_pipeline_config(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "civStation.utils.image_pipeline",
        SimpleNamespace(config_from_args=lambda *args: None),
    )


def _main_args(*, backend="vlm", **overrides):
    values = {
        "provider": None,
        "model": None,
        "router_provider": None,
        "router_model": None,
        "planner_provider": None,
        "planner_model": None,
        "turn_detector_provider": None,
        "turn_detector_model": None,
        "civ6_mcp_path": None,
        "civ6_mcp_launcher": None,
        "hitl": False,
        "autonomous": False,
        "hitl_mode": None,
        "strategy": None,
        "knowledge_index": None,
        "enable_web_search": False,
        "status_ui": False,
        "control_api": False,
        "status_port": 8765,
        "wait_for_start": False,
        "relay_url": None,
        "relay_token": None,
        "debug": "",
        "turns": 1,
        "range": 1000,
        "delay_action": 0.5,
        "delay_turn": 1.0,
        "prompt_language": "eng",
    }
    if backend is not _OMIT_BACKEND:
        values["backend"] = backend
    values.update(overrides)
    return SimpleNamespace(**values)


def _fake_civ6_mcp_checkout(tmp_path: Path) -> Path:
    path = tmp_path / "civ6-mcp"
    path.mkdir()
    (path / "pyproject.toml").write_text("[project]\nname = 'civ6-mcp'\n", encoding="utf-8")
    return path


def _patch_main_backend_harness(monkeypatch, args):
    session = _DummyRunLogSession()
    trajectory_session = _DummyScreenshotTrajectorySession()
    context_updater = _DummyWorker()
    rich_logger = _DummyRichLoggerInstance()
    vlm_calls = []

    class _DummyRichLogger:
        @classmethod
        def get(cls):
            return rich_logger

    monkeypatch.setattr(turn_runner, "parse_args", lambda: args)
    monkeypatch.setattr(turn_runner, "start_run_log_session", lambda: session)
    monkeypatch.setattr(turn_runner, "start_screenshot_trajectory_session", lambda **kwargs: trajectory_session)
    monkeypatch.setattr(turn_runner, "setup_providers", lambda args: (object(), object()))
    monkeypatch.setattr(turn_runner.ContextManager, "get_instance", staticmethod(lambda: object()))
    monkeypatch.setattr(turn_runner, "setup_chat_app", lambda *args: (None, None, None))
    monkeypatch.setattr(turn_runner, "setup_knowledge", lambda *args: None)
    monkeypatch.setattr(turn_runner, "CommandQueue", lambda: object())
    monkeypatch.setattr(turn_runner, "run_one_turn", lambda **kwargs: vlm_calls.append(("one", kwargs)))
    monkeypatch.setattr(turn_runner, "run_multi_turn", lambda **kwargs: vlm_calls.append(("multi", kwargs)))
    _patch_image_pipeline_config(monkeypatch)
    monkeypatch.setitem(
        sys.modules,
        "civStation.agent.modules.context.macro_turn_manager",
        SimpleNamespace(MacroTurnManager=lambda *args, **kwargs: object()),
    )
    monkeypatch.setitem(
        sys.modules,
        "civStation.agent.modules.context.context_updater",
        SimpleNamespace(ContextUpdater=lambda *args, **kwargs: context_updater),
    )
    monkeypatch.setitem(
        sys.modules,
        "civStation.agent.modules.hitl.agent_gate",
        SimpleNamespace(AgentGate=_DummyAgentGate, AgentState=_DummyAgentState),
    )
    monkeypatch.setitem(
        sys.modules,
        "civStation.utils.rich_logger",
        SimpleNamespace(RichLogger=_DummyRichLogger),
    )

    return SimpleNamespace(
        session=session,
        trajectory_session=trajectory_session,
        context_updater=context_updater,
        rich_logger=rich_logger,
        vlm_calls=vlm_calls,
    )


def _guard_kwargs(**overrides):
    values = {
        "macro_turn_manager": None,
        "context_updater": None,
        "turn_detector": None,
        "router_img_config": None,
        "planner_img_config": None,
        "context_img_config": None,
        "turn_detector_img_config": None,
        "civ6_mcp_client": None,
    }
    values.update(overrides)
    return values


def test_runtime_guard_allows_vlm_only_runtime_configuration() -> None:
    turn_runner._guard_backend_runtime_state(
        BackendKind.VLM,
        **_guard_kwargs(
            macro_turn_manager=object(),
            context_updater=object(),
            turn_detector=object(),
            router_img_config=object(),
            planner_img_config=object(),
            context_img_config=object(),
            turn_detector_img_config=object(),
        ),
    )


def test_runtime_guard_allows_civ6_mcp_only_runtime_configuration() -> None:
    turn_runner._guard_backend_runtime_state(
        BackendKind.CIV6_MCP,
        **_guard_kwargs(civ6_mcp_client=object()),
    )


@pytest.mark.parametrize(
    "component_name",
    [
        "macro_turn_manager",
        "context_updater",
        "turn_detector",
        "router_img_config",
        "planner_img_config",
        "context_img_config",
        "turn_detector_img_config",
        "future_observer_img_config",
    ],
)
def test_runtime_guard_rejects_civ6_mcp_with_vlm_components(component_name):
    with pytest.raises(turn_runner.BackendRuntimeConflictError, match=component_name):
        turn_runner._guard_backend_runtime_state(
            BackendKind.CIV6_MCP,
            **_guard_kwargs(**{component_name: object()}),
        )


def test_runtime_guard_rejects_vlm_with_civ6_mcp_client():
    with pytest.raises(turn_runner.BackendRuntimeConflictError, match="VLM backend cannot run"):
        turn_runner._guard_backend_runtime_state(
            BackendKind.VLM,
            **_guard_kwargs(civ6_mcp_client=object()),
        )


def test_main_omitted_backend_runs_vlm_and_not_civ6_mcp(monkeypatch):
    args = _main_args(
        backend=_OMIT_BACKEND,
        civ6_mcp_path="/tmp/civ6-mcp",
        civ6_mcp_launcher="python",
    )
    harness = _patch_main_backend_harness(monkeypatch, args)
    mcp_calls = []

    monkeypatch.setitem(
        sys.modules,
        "civStation.agent.modules.backend.civ6_mcp.turn_loop",
        SimpleNamespace(
            Civ6McpUnavailableError=RuntimeError,
            build_civ6_mcp_client=lambda **kwargs: mcp_calls.append(("build", kwargs)),
            run_one_turn_civ6_mcp=lambda **kwargs: mcp_calls.append(("one", kwargs)),
            run_multi_turn_civ6_mcp=lambda **kwargs: mcp_calls.append(("multi", kwargs)),
        ),
    )

    turn_runner.main()

    assert [name for name, _kwargs in harness.vlm_calls] == ["one"]
    assert mcp_calls == []
    assert harness.context_updater.started is True
    assert harness.context_updater.stopped is True


def test_main_explicit_vlm_backend_runs_vlm_only(monkeypatch):
    args = _main_args(backend="vlm")
    harness = _patch_main_backend_harness(monkeypatch, args)
    mcp_calls = []

    monkeypatch.setitem(
        sys.modules,
        "civStation.agent.modules.backend.civ6_mcp.turn_loop",
        SimpleNamespace(
            Civ6McpUnavailableError=RuntimeError,
            build_civ6_mcp_client=lambda **kwargs: mcp_calls.append(("build", kwargs)),
            run_one_turn_civ6_mcp=lambda **kwargs: mcp_calls.append(("one", kwargs)),
            run_multi_turn_civ6_mcp=lambda **kwargs: mcp_calls.append(("multi", kwargs)),
        ),
    )

    turn_runner.main()

    assert [name for name, _kwargs in harness.vlm_calls] == ["one"]
    assert mcp_calls == []
    assert harness.context_updater.started is True
    assert harness.context_updater.stopped is True


def test_main_explicit_civ6_mcp_backend_runs_mcp_only(monkeypatch, tmp_path):
    mcp_path = _fake_civ6_mcp_checkout(tmp_path)
    args = _main_args(
        backend="civ6-mcp",
        civ6_mcp_path=str(mcp_path),
        civ6_mcp_launcher="python",
        strategy="science",
    )
    harness = _patch_main_backend_harness(monkeypatch, args)
    mcp_calls = []
    monkeypatch.setitem(
        sys.modules,
        "civStation.agent.modules.backend.civ6_mcp.turn_loop",
        SimpleNamespace(
            Civ6McpError=RuntimeError,
            Civ6McpUnavailableError=RuntimeError,
            run_civ6_mcp_turn_loop=lambda **kwargs: mcp_calls.append(kwargs),
        ),
    )

    turn_runner.main()

    assert harness.vlm_calls == []
    assert len(mcp_calls) == 1
    assert mcp_calls[0]["num_turns"] == 1
    assert mcp_calls[0]["install_path"] == str(mcp_path)
    assert mcp_calls[0]["launcher"] == "python"
    assert mcp_calls[0]["high_level_strategy"] == "science"
    assert harness.context_updater.started is False
    assert harness.context_updater.stopped is False


def test_main_civ6_mcp_unavailable_does_not_fallback_to_vlm(monkeypatch, tmp_path):
    class _Unavailable(RuntimeError):
        pass

    mcp_path = _fake_civ6_mcp_checkout(tmp_path)
    args = _main_args(
        backend="civ6-mcp",
        civ6_mcp_path=str(mcp_path),
        civ6_mcp_launcher="python",
    )
    harness = _patch_main_backend_harness(monkeypatch, args)
    mcp_calls = []

    def _raise_unavailable(**_kwargs):
        raise _Unavailable("missing civ6-mcp")

    monkeypatch.setitem(
        sys.modules,
        "civStation.agent.modules.backend.civ6_mcp.turn_loop",
        SimpleNamespace(
            Civ6McpError=RuntimeError,
            Civ6McpUnavailableError=_Unavailable,
            run_civ6_mcp_turn_loop=_raise_unavailable,
        ),
    )

    turn_runner.main()

    assert harness.vlm_calls == []
    assert mcp_calls == []
    assert harness.context_updater.started is False
    assert harness.context_updater.stopped is False
    assert harness.rich_logger.started == 1
    assert harness.rich_logger.stopped == 1
    assert harness.session.closed is True
    assert harness.trajectory_session.closed is True


def test_main_validates_civ6_mcp_install_path_before_startup(monkeypatch, tmp_path):
    missing_path = tmp_path / "missing-civ6-mcp"
    args = _main_args(
        backend="civ6-mcp",
        civ6_mcp_path=str(missing_path),
        civ6_mcp_launcher="python",
    )
    harness = _patch_main_backend_harness(monkeypatch, args)
    setup_calls = []

    monkeypatch.setattr(turn_runner, "setup_providers", lambda args: setup_calls.append(args))

    turn_runner.main()

    assert setup_calls == []
    assert harness.vlm_calls == []
    assert harness.rich_logger.started == 0
    assert harness.rich_logger.stopped == 0
    assert harness.session.closed is True
    assert harness.trajectory_session.closed is True


def test_main_validates_civ6_mcp_launcher_command_before_startup(monkeypatch, tmp_path):
    mcp_path = _fake_civ6_mcp_checkout(tmp_path)
    args = _main_args(
        backend="civ6-mcp",
        civ6_mcp_path=str(mcp_path),
        civ6_mcp_launcher="python",
    )
    harness = _patch_main_backend_harness(monkeypatch, args)
    setup_calls = []

    monkeypatch.setattr(turn_runner.shutil, "which", lambda _name: None)
    monkeypatch.setattr(turn_runner, "setup_providers", lambda args: setup_calls.append(args))

    turn_runner.main()

    assert setup_calls == []
    assert harness.vlm_calls == []
    assert harness.rich_logger.started == 0
    assert harness.rich_logger.stopped == 0
    assert harness.session.closed is True
    assert harness.trajectory_session.closed is True


@pytest.mark.parametrize("backend", ["vlm", "civ6-mcp"])
def test_parse_args_accepts_valid_backend_values(monkeypatch, tmp_path, backend):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(turn_runner, "get_available_providers", lambda: {"gemini": object()})

    args = turn_runner.parse_args(["--backend", backend])

    assert args.backend == backend
    assert parse_backend_kind(args.backend) is (BackendKind.CIV6_MCP if backend == "civ6-mcp" else BackendKind.VLM)


@pytest.mark.parametrize(
    "backend",
    [
        "anthropic",
        "civ6",
        "computer-use",
        "cu",
        "mcp",
        "civ-mcp",
        "vlm,civ6-mcp",
        "computer-use,civ6-mcp",
        "vlm+civ6-mcp",
    ],
)
def test_parse_args_rejects_invalid_backend_values(monkeypatch, tmp_path, capsys, backend):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(turn_runner, "get_available_providers", lambda: {"gemini": object()})

    with pytest.raises(SystemExit) as exc_info:
        turn_runner.parse_args(["--backend", backend])

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "--backend" in stderr
    assert "invalid choice" in stderr
    assert backend in stderr
    assert "vlm" in stderr
    assert "civ6-mcp" in stderr


def test_main_closes_run_log_session_when_configuration_fails(monkeypatch):
    session = _DummyRunLogSession()
    trajectory_session = _DummyScreenshotTrajectorySession()

    monkeypatch.setattr(turn_runner, "parse_args", lambda: SimpleNamespace())
    monkeypatch.setattr(turn_runner, "start_run_log_session", lambda: session)
    monkeypatch.setattr(turn_runner, "start_screenshot_trajectory_session", lambda **kwargs: trajectory_session)
    monkeypatch.setattr(
        turn_runner,
        "setup_providers",
        lambda args: (_ for _ in ()).throw(ValueError("bad config")),
    )

    turn_runner.main()

    assert session.closed is True
    assert trajectory_session.closed is True


def test_main_closes_run_log_session_after_one_turn(monkeypatch):
    session = _DummyRunLogSession()
    trajectory_session = _DummyScreenshotTrajectorySession()
    context_updater = _DummyWorker()
    rich_logger = _DummyRichLoggerInstance()
    run_calls = []

    class _DummyRichLogger:
        @classmethod
        def get(cls):
            return rich_logger

    args = SimpleNamespace(
        provider=None,
        model=None,
        router_provider=None,
        router_model=None,
        planner_provider=None,
        planner_model=None,
        turn_detector_provider=None,
        turn_detector_model=None,
        hitl=False,
        autonomous=False,
        hitl_mode=None,
        strategy=None,
        knowledge_index=None,
        enable_web_search=False,
        status_ui=False,
        control_api=False,
        status_port=8765,
        wait_for_start=False,
        relay_url=None,
        relay_token=None,
        debug="",
        turns=1,
        range=1000,
        delay_action=0.5,
        delay_turn=1.0,
    )

    monkeypatch.setattr(turn_runner, "parse_args", lambda: args)
    monkeypatch.setattr(turn_runner, "start_run_log_session", lambda: session)
    monkeypatch.setattr(turn_runner, "start_screenshot_trajectory_session", lambda **kwargs: trajectory_session)
    monkeypatch.setattr(turn_runner, "setup_providers", lambda args: (object(), object()))
    monkeypatch.setattr(turn_runner.ContextManager, "get_instance", staticmethod(lambda: object()))
    monkeypatch.setattr(turn_runner, "setup_chat_app", lambda *args: (None, None, None))
    monkeypatch.setattr(turn_runner, "setup_knowledge", lambda *args: None)
    monkeypatch.setattr(turn_runner, "CommandQueue", lambda: object())
    monkeypatch.setattr(turn_runner, "run_one_turn", lambda **kwargs: run_calls.append(kwargs))
    _patch_image_pipeline_config(monkeypatch)
    monkeypatch.setitem(
        sys.modules,
        "civStation.agent.modules.context.macro_turn_manager",
        SimpleNamespace(MacroTurnManager=lambda *args, **kwargs: object()),
    )
    monkeypatch.setitem(
        sys.modules,
        "civStation.agent.modules.context.context_updater",
        SimpleNamespace(ContextUpdater=lambda *args, **kwargs: context_updater),
    )
    monkeypatch.setitem(
        sys.modules,
        "civStation.agent.modules.hitl.agent_gate",
        SimpleNamespace(AgentGate=_DummyAgentGate, AgentState=_DummyAgentState),
    )
    monkeypatch.setitem(
        sys.modules,
        "civStation.utils.rich_logger",
        SimpleNamespace(RichLogger=_DummyRichLogger),
    )

    turn_runner.main()

    assert run_calls and run_calls[0]["delay_before_action"] == 0.5
    assert context_updater.started is True
    assert context_updater.stopped is True
    assert rich_logger.started == 1
    assert rich_logger.stopped == 1
    assert session.closed is True
    assert trajectory_session.closed is True


@pytest.mark.parametrize(("turns", "expected_runner"), [(1, "one"), (3, "multi")])
def test_main_routes_civ6_mcp_backend_to_mcp_turn_loop(monkeypatch, tmp_path, turns, expected_runner):
    session = _DummyRunLogSession()
    trajectory_session = _DummyScreenshotTrajectorySession()
    context_updater = _DummyWorker()
    rich_logger = _DummyRichLoggerInstance()
    mcp_calls = []
    mcp_path = _fake_civ6_mcp_checkout(tmp_path)

    class _DummyRichLogger:
        @classmethod
        def get(cls):
            return rich_logger

    args = SimpleNamespace(
        provider=None,
        model=None,
        router_provider=None,
        router_model=None,
        planner_provider=None,
        planner_model=None,
        turn_detector_provider=None,
        turn_detector_model=None,
        backend="civ6-mcp",
        civ6_mcp_path=str(mcp_path),
        civ6_mcp_launcher="python",
        hitl=False,
        autonomous=False,
        hitl_mode=None,
        strategy="science",
        knowledge_index=None,
        enable_web_search=False,
        status_ui=False,
        control_api=False,
        status_port=8765,
        wait_for_start=False,
        relay_url=None,
        relay_token=None,
        debug="",
        turns=turns,
        range=1000,
        delay_action=0.5,
        delay_turn=1.0,
    )

    def _fail_vlm_runner(**_kwargs):
        raise AssertionError("VLM turn runner must not be called for civ6-mcp backend")

    monkeypatch.setattr(turn_runner, "parse_args", lambda: args)
    monkeypatch.setattr(turn_runner, "start_run_log_session", lambda: session)
    monkeypatch.setattr(turn_runner, "start_screenshot_trajectory_session", lambda **kwargs: trajectory_session)
    monkeypatch.setattr(turn_runner, "setup_providers", lambda args: (object(), object()))
    monkeypatch.setattr(turn_runner.ContextManager, "get_instance", staticmethod(lambda: object()))
    monkeypatch.setattr(turn_runner, "setup_chat_app", lambda *args: (None, None, None))
    monkeypatch.setattr(turn_runner, "setup_knowledge", lambda *args: None)
    monkeypatch.setattr(turn_runner, "CommandQueue", lambda: object())
    monkeypatch.setattr(turn_runner, "run_one_turn", _fail_vlm_runner)
    monkeypatch.setattr(turn_runner, "run_multi_turn", _fail_vlm_runner)
    _patch_image_pipeline_config(monkeypatch)
    monkeypatch.setitem(
        sys.modules,
        "civStation.agent.modules.context.macro_turn_manager",
        SimpleNamespace(MacroTurnManager=lambda *args, **kwargs: object()),
    )
    monkeypatch.setitem(
        sys.modules,
        "civStation.agent.modules.context.context_updater",
        SimpleNamespace(ContextUpdater=lambda *args, **kwargs: context_updater),
    )
    monkeypatch.setitem(
        sys.modules,
        "civStation.agent.modules.hitl.agent_gate",
        SimpleNamespace(AgentGate=_DummyAgentGate, AgentState=_DummyAgentState),
    )
    monkeypatch.setitem(
        sys.modules,
        "civStation.utils.rich_logger",
        SimpleNamespace(RichLogger=_DummyRichLogger),
    )
    monkeypatch.setitem(
        sys.modules,
        "civStation.agent.modules.backend.civ6_mcp.turn_loop",
        SimpleNamespace(
            Civ6McpError=RuntimeError,
            Civ6McpUnavailableError=RuntimeError,
            run_civ6_mcp_turn_loop=lambda **kwargs: mcp_calls.append(kwargs),
        ),
    )

    turn_runner.main()

    assert len(mcp_calls) == 1
    runner_kwargs = mcp_calls[0]
    assert expected_runner in {"one", "multi"}
    assert runner_kwargs["num_turns"] == turns
    assert runner_kwargs["high_level_strategy"] == "science"
    assert runner_kwargs["install_path"] == str(mcp_path)
    assert runner_kwargs["launcher"] == "python"
    assert runner_kwargs["delay_between_turns"] == 1.0
    assert context_updater.started is False
    assert context_updater.stopped is False
    assert rich_logger.started == 1
    assert rich_logger.stopped == 1
    assert session.closed is True
    assert trajectory_session.closed is True


def test_console_log_silencer_hides_info_from_stream_handlers_only(tmp_path):
    logger = logging.getLogger("tests.turn_runner.console_silencer")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    stream = io.StringIO()
    stream_handler = logging.StreamHandler(stream)
    stream_handler.setLevel(logging.INFO)

    file_path = tmp_path / "run.log"
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)

    original_handlers = logger.handlers[:]
    logger.handlers = [stream_handler, file_handler]

    try:
        silencer = turn_runner._ConsoleLogSilencer(logger=logger)
        silencer.enable()

        logger.info("hidden info")
        logger.warning("visible warning")

        silencer.disable()
    finally:
        for handler in logger.handlers:
            handler.flush()
            handler.close()
        logger.handlers = original_handlers

    stream_output = stream.getvalue()
    file_output = file_path.read_text(encoding="utf-8")

    assert "hidden info" not in stream_output
    assert "visible warning" in stream_output
    assert "hidden info" in file_output
    assert "visible warning" in file_output
