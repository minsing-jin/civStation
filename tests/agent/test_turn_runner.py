import io
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

from civStation.agent import turn_runner


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
    monkeypatch.setattr("civStation.utils.image_pipeline.config_from_args", lambda *args: None)
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
