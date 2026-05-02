"""Lifecycle tests for the civ6-mcp stdio MCP client.

These tests use fake MCP SDK objects so they verify client lifecycle behavior
without spawning civ6-mcp, Civ6, FireTuner, or uv.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import types
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest

from civStation.agent.modules.backend.civ6_mcp.client import (
    LOOP_THREAD_JOIN_TIMEOUT_SECONDS,
    Civ6McpClient,
    Civ6McpConfig,
    Civ6McpError,
    Civ6McpUnavailableError,
)


class FakeTool:
    def __init__(self, name: str, description: str = "", input_schema: dict[str, Any] | None = None) -> None:
        self.name = name
        self.description = description
        self.inputSchema = input_schema or {}


class FakeToolsResponse:
    def __init__(self, tools: list[FakeTool]) -> None:
        self.tools = tools


class FakeResultBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeCallResult:
    isError = False

    def __init__(self, text: str) -> None:
        self.content = [FakeResultBlock(text)]


class FakeStdioServerParameters:
    def __init__(self, *, command: str, args: list[str], env: dict[str, str]) -> None:
        self.command = command
        self.args = args
        self.env = env


class FakeProcess:
    def __init__(self) -> None:
        self.running = True

    def terminate(self) -> None:
        self.running = False


class FakeImplementation:
    def __init__(self, *, name: str, version: str) -> None:
        self.name = name
        self.version = version


class FakeStdioContext:
    events: list[str] = []
    entered_params = None
    processes: list[FakeProcess] = []

    async def __aenter__(self):
        self.process = FakeProcess()
        self.processes.append(self.process)
        self.events.append("stdio_enter")
        return object(), object()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        process = getattr(self, "process", None)
        if process is not None:
            process.terminate()
        self.events.append("stdio_exit")


class FakeClientSession:
    events: list[str] = []
    tools = [
        FakeTool("get_game_overview", "overview", {"type": "object"}),
        FakeTool("end_turn", "end the turn", {"type": "object"}),
    ]
    ping_count = 0
    open_session_count = 0
    initialize_loop_id: int | None = None
    initialize_loop: asyncio.AbstractEventLoop | None = None
    initialize_thread_name: str | None = None
    initialize_thread: threading.Thread | None = None
    spawn_background_task_on_initialize = False
    background_tasks: list[asyncio.Task[None]] = []

    def __init__(self, read_stream, write_stream, **kwargs) -> None:  # noqa: ANN001
        self.read_stream = read_stream
        self.write_stream = write_stream
        self.kwargs = kwargs

    async def __aenter__(self):
        self.events.append("session_enter")
        self.__class__.open_session_count += 1
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.events.append("session_exit")
        self.__class__.open_session_count -= 1

    async def initialize(self) -> None:
        loop = asyncio.get_running_loop()
        self.__class__.initialize_loop_id = id(loop)
        self.__class__.initialize_loop = loop
        self.__class__.initialize_thread_name = threading.current_thread().name
        self.__class__.initialize_thread = threading.current_thread()
        self.events.append("initialize")
        if self.__class__.spawn_background_task_on_initialize:
            task = asyncio.create_task(self._background_task())
            self.__class__.background_tasks.append(task)

    async def list_tools(self) -> FakeToolsResponse:
        self.events.append("list_tools")
        return FakeToolsResponse(self.tools)

    async def ping(self) -> None:
        self.events.append("ping")
        self.ping_count += 1

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> FakeCallResult:
        self.events.append(f"call_tool:{name}")
        return FakeCallResult(f"{name} ok {arguments}")

    async def _background_task(self) -> None:
        await asyncio.Event().wait()


@pytest.fixture
def fake_mcp_sdk(monkeypatch):
    mcp = types.ModuleType("mcp")
    client = types.ModuleType("mcp.client")
    stdio = types.ModuleType("mcp.client.stdio")
    mcp_types = types.ModuleType("mcp.types")

    monkeypatch.setitem(sys.modules, "mcp", mcp)
    monkeypatch.setitem(sys.modules, "mcp.client", client)
    monkeypatch.setitem(sys.modules, "mcp.client.stdio", stdio)
    monkeypatch.setitem(sys.modules, "mcp.types", mcp_types)
    monkeypatch.setattr(mcp, "client", client, raising=False)
    monkeypatch.setattr(client, "stdio", stdio, raising=False)
    monkeypatch.setattr(mcp, "StdioServerParameters", FakeStdioServerParameters, raising=False)
    monkeypatch.setattr(mcp_types, "Implementation", FakeImplementation, raising=False)

    FakeStdioContext.events = []
    FakeStdioContext.entered_params = None
    FakeStdioContext.processes = []
    FakeClientSession.events = []
    FakeClientSession.ping_count = 0
    FakeClientSession.open_session_count = 0
    FakeClientSession.initialize_loop_id = None
    FakeClientSession.initialize_loop = None
    FakeClientSession.initialize_thread_name = None
    FakeClientSession.initialize_thread = None
    FakeClientSession.spawn_background_task_on_initialize = False
    FakeClientSession.background_tasks = []
    FakeClientSession.tools = [
        FakeTool("get_game_overview", "overview", {"type": "object"}),
        FakeTool("end_turn", "end the turn", {"type": "object"}),
    ]

    def fake_stdio_client(params):
        FakeStdioContext.entered_params = params
        return FakeStdioContext()

    monkeypatch.setattr(mcp, "ClientSession", FakeClientSession, raising=False)
    monkeypatch.setattr(stdio, "stdio_client", fake_stdio_client, raising=False)
    return FakeClientSession, FakeStdioContext


@pytest.fixture
def install_path(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'civ6-mcp'\n")
    return tmp_path


def assert_no_runtime_leaks(
    client: Civ6McpClient,
    *,
    loop: asyncio.AbstractEventLoop | None,
    loop_thread: threading.Thread | None,
    session: type[FakeClientSession],
    stdio: type[FakeStdioContext],
) -> None:
    assert all(not process.running for process in stdio.processes)
    assert session.open_session_count == 0
    assert all(task.done() for task in session.background_tasks)
    assert loop_thread is None or not loop_thread.is_alive()
    assert loop is None or loop.is_closed()
    assert client._loop is None
    assert client._loop_thread is None
    assert client._session is None
    assert client._session_ctx is None
    assert client._stdio_ctx is None
    assert client.tool_names == set()


def test_start_initializes_session_lists_tools_and_shutdown_closes_contexts(
    fake_mcp_sdk,
    install_path: Path,
) -> None:
    session, stdio = fake_mcp_sdk
    config = Civ6McpConfig(install_path=install_path, launcher="python")
    client = Civ6McpClient(config)

    client.start()

    assert client.tool_names == {"get_game_overview", "end_turn"}
    assert client.tool_catalog["get_game_overview"]["description"] == "overview"
    assert client.tool_schemas()["get_game_overview"]["description"] == "overview"
    assert client.startup_health.ok is True
    assert client.startup_health.missing_required_tools == set()
    assert client.has_required_tools is True
    assert session.events[:3] == ["session_enter", "initialize", "list_tools"]
    assert stdio.events == ["stdio_enter"]
    assert stdio.entered_params.command == "python"
    assert stdio.entered_params.args == ["-m", "civ_mcp"]

    client.stop()

    assert session.events[-1] == "session_exit"
    assert stdio.events[-1] == "stdio_exit"
    assert client.tool_names == set()


def test_stop_is_idempotent_after_active_session_shutdown(
    fake_mcp_sdk,
    install_path: Path,
) -> None:
    session, stdio = fake_mcp_sdk
    client = Civ6McpClient(Civ6McpConfig(install_path=install_path, launcher="python"))

    client.start()
    client.stop()
    client.stop()

    assert session.events.count("session_exit") == 1
    assert stdio.events.count("stdio_exit") == 1
    assert client._loop is None
    assert client._loop_thread is None
    assert client.tool_names == set()


def test_stop_signals_event_loop_closes_it_and_joins_thread_with_timeout(
    fake_mcp_sdk,
    install_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Civ6McpClient(Civ6McpConfig(install_path=install_path, launcher="python"))

    client.start()
    loop = client._loop
    loop_thread = client._loop_thread
    assert loop is not None
    assert loop_thread is not None

    join_timeouts: list[float | None] = []
    original_join = threading.Thread.join

    def record_join(self: threading.Thread, timeout: float | None = None) -> None:
        if self is loop_thread:
            join_timeouts.append(timeout)
        original_join(self, timeout=timeout)

    monkeypatch.setattr(threading.Thread, "join", record_join)

    client.stop()

    assert join_timeouts == [LOOP_THREAD_JOIN_TIMEOUT_SECONDS]
    assert not loop_thread.is_alive()
    assert loop.is_closed()
    assert client._loop is None
    assert client._loop_thread is None


def test_stop_leaves_no_process_session_thread_or_pending_tasks_after_normal_startup(
    fake_mcp_sdk,
    install_path: Path,
) -> None:
    session, stdio = fake_mcp_sdk
    session.spawn_background_task_on_initialize = True
    client = Civ6McpClient(Civ6McpConfig(install_path=install_path, launcher="python"))

    client.start()
    loop = client._loop
    loop_thread = client._loop_thread

    assert stdio.processes
    assert all(process.running for process in stdio.processes)
    assert session.open_session_count == 1
    assert session.background_tasks
    assert any(not task.done() for task in session.background_tasks)
    assert loop_thread is not None and loop_thread.is_alive()

    client.stop()

    assert_no_runtime_leaks(
        client,
        loop=loop,
        loop_thread=loop_thread,
        session=session,
        stdio=stdio,
    )


def test_stop_closes_partially_initialized_contexts_when_not_marked_started(
    fake_mcp_sdk,
    install_path: Path,
) -> None:
    session_class, stdio_class = fake_mcp_sdk
    client = Civ6McpClient(Civ6McpConfig(install_path=install_path, launcher="python"))
    loop = asyncio.new_event_loop()
    loop_ready = threading.Event()
    loop_thread = threading.Thread(
        target=client._run_loop_forever,
        args=(loop, loop_ready),
        name="civ6-mcp-loop",
        daemon=True,
    )
    client._loop = loop
    client._loop_ready = loop_ready
    client._loop_thread = loop_thread
    client._stdio_ctx = stdio_class()
    client._session_ctx = session_class(object(), object())
    client._session = client._session_ctx
    client._tool_names = {"get_game_overview", "end_turn"}

    loop_thread.start()
    assert loop_ready.wait(timeout=1.0)

    client.stop()
    client.stop()

    assert session_class.events == ["session_exit"]
    assert stdio_class.events == ["stdio_exit"]
    assert client._loop is None
    assert client._loop_thread is None
    assert client._session is None
    assert client._session_ctx is None
    assert client._stdio_ctx is None
    assert client.tool_names == set()


def test_start_exposes_missing_required_tools_from_initial_catalog(
    fake_mcp_sdk,
    install_path: Path,
) -> None:
    session, _stdio = fake_mcp_sdk
    session.tools = [
        FakeTool("get_game_overview", "overview", {"type": "object"}),
    ]
    client = Civ6McpClient(Civ6McpConfig(install_path=install_path, launcher="python"))

    client.start()

    try:
        startup_health = client.startup_health

        assert startup_health.ok is False
        assert startup_health.started is True
        assert startup_health.initialized is True
        assert startup_health.tool_count == 1
        assert startup_health.missing_required_tools == {"end_turn"}
        assert client.has_required_tools is False
        assert client.missing_required_tools == {"end_turn"}
    finally:
        client.stop()


def test_start_cleans_up_stdio_and_session_when_initialize_fails(
    fake_mcp_sdk,
    install_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, stdio = fake_mcp_sdk

    async def fail_initialize(self) -> None:  # noqa: ANN001
        self.events.append("initialize")
        raise RuntimeError("initialize failed")

    monkeypatch.setattr(session, "initialize", fail_initialize)
    client = Civ6McpClient(Civ6McpConfig(install_path=install_path, launcher="python"))

    with pytest.raises(Civ6McpError, match="Failed to start civ6-mcp server"):
        client.start()

    assert client.health_check().started is False
    assert session.events == ["session_enter", "initialize", "session_exit"]
    assert stdio.events == ["stdio_enter", "stdio_exit"]
    assert client._loop is None
    assert client._loop_thread is None
    assert client.tool_names == set()


def test_start_failure_leaves_no_process_session_thread_or_pending_tasks(
    fake_mcp_sdk,
    install_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, stdio = fake_mcp_sdk
    session.spawn_background_task_on_initialize = True
    original_initialize = session.initialize

    async def fail_initialize(self) -> None:  # noqa: ANN001
        await original_initialize(self)
        raise RuntimeError("initialize failed after background task started")

    monkeypatch.setattr(session, "initialize", fail_initialize)
    client = Civ6McpClient(Civ6McpConfig(install_path=install_path, launcher="python"))

    with pytest.raises(Civ6McpError, match="Failed to start civ6-mcp server"):
        client.start()

    assert session.events == ["session_enter", "initialize", "session_exit"]
    assert stdio.events == ["stdio_enter", "stdio_exit"]
    assert_no_runtime_leaks(
        client,
        loop=session.initialize_loop,
        loop_thread=session.initialize_thread,
        session=session,
        stdio=stdio,
    )


def test_start_fails_and_cleans_up_when_initial_tool_catalog_refresh_fails(
    fake_mcp_sdk,
    install_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, stdio = fake_mcp_sdk

    async def fail_list_tools(self) -> FakeToolsResponse:  # noqa: ANN001
        self.events.append("list_tools")
        raise RuntimeError("list tools failed")

    monkeypatch.setattr(session, "list_tools", fail_list_tools)
    client = Civ6McpClient(Civ6McpConfig(install_path=install_path, launcher="python"))

    with pytest.raises(Civ6McpError, match="Failed to start civ6-mcp server"):
        client.start()

    assert session.events == ["session_enter", "initialize", "list_tools", "session_exit"]
    assert stdio.events == ["stdio_enter", "stdio_exit"]
    assert client._loop is None
    assert client._loop_thread is None
    assert client.tool_names == set()
    assert client.tool_catalog == {}


def test_start_timeout_reports_failure_and_closes_started_contexts(
    fake_mcp_sdk,
    install_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, stdio = fake_mcp_sdk

    async def never_initialize(self) -> None:  # noqa: ANN001
        self.events.append("initialize")
        await asyncio.Event().wait()

    monkeypatch.setattr(session, "initialize", never_initialize)
    config = Civ6McpConfig(
        install_path=install_path,
        launcher="python",
        startup_timeout_seconds=0.05,
    )
    client = Civ6McpClient(config)

    with pytest.raises(Civ6McpError, match="Timed out starting civ6-mcp server"):
        client.start()

    assert session.events == ["session_enter", "initialize", "session_exit"]
    assert stdio.events == ["stdio_enter", "stdio_exit"]
    assert client._loop is None
    assert client._loop_thread is None
    assert client.tool_names == set()


def test_start_interrupted_by_keyboard_interrupt_closes_started_contexts(
    fake_mcp_sdk,
    install_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, stdio = fake_mcp_sdk
    original_future_result = Future.result
    interrupt_raised = False

    async def never_initialize(self) -> None:  # noqa: ANN001
        self.events.append("initialize")
        await asyncio.Event().wait()

    def interrupt_startup_wait_once(self: Future[Any], timeout: float | None = None) -> Any:
        nonlocal interrupt_raised
        if not interrupt_raised:
            deadline = time.monotonic() + 1.0
            while "initialize" not in session.events and time.monotonic() < deadline:
                time.sleep(0.001)
            interrupt_raised = True
            raise KeyboardInterrupt
        return original_future_result(self, timeout=timeout)

    monkeypatch.setattr(session, "initialize", never_initialize)
    monkeypatch.setattr(Future, "result", interrupt_startup_wait_once)
    client = Civ6McpClient(Civ6McpConfig(install_path=install_path, launcher="python"))

    try:
        with pytest.raises(KeyboardInterrupt):
            client.start()

        assert interrupt_raised is True
        assert session.events == ["session_enter", "initialize", "session_exit"]
        assert stdio.events == ["stdio_enter", "stdio_exit"]
        assert_no_runtime_leaks(
            client,
            loop=session.initialize_loop,
            loop_thread=session.initialize_thread,
            session=session,
            stdio=stdio,
        )
    finally:
        client.stop()


def test_health_check_pings_session_and_refreshes_tool_catalog(fake_mcp_sdk, install_path: Path) -> None:
    session, _stdio = fake_mcp_sdk
    config = Civ6McpConfig(install_path=install_path, launcher="python")
    client = Civ6McpClient(config)
    client.start()

    session.tools = [
        FakeTool("get_game_overview", "overview", {"type": "object"}),
        FakeTool("end_turn", "end the turn", {"type": "object"}),
        FakeTool("get_units", "units", {"type": "object"}),
    ]
    health = client.health_check(required_tools={"get_game_overview", "end_turn"})

    assert health.ok is True
    assert health.started is True
    assert health.initialized is True
    assert health.tool_count == 3
    assert health.missing_required_tools == set()
    assert client.has_tool("get_units") is True
    assert "ping" in session.events

    client.stop()


def test_health_check_fails_when_required_tool_is_missing_from_refreshed_catalog(
    fake_mcp_sdk,
    install_path: Path,
) -> None:
    session, _stdio = fake_mcp_sdk
    client = Civ6McpClient(Civ6McpConfig(install_path=install_path, launcher="python"))
    client.start()

    session.tools = [
        FakeTool("get_game_overview", "overview", {"type": "object"}),
    ]
    health = client.health_check(required_tools={"get_game_overview", "end_turn"})

    assert health.ok is False
    assert health.started is True
    assert health.initialized is True
    assert health.tool_count == 1
    assert health.missing_required_tools == {"end_turn"}
    assert client.has_tool("end_turn") is False
    assert "end_turn" in health.message

    client.stop()


def test_health_check_fails_when_session_becomes_inactive_during_probe(
    fake_mcp_sdk,
    install_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _stdio = fake_mcp_sdk
    client = Civ6McpClient(Civ6McpConfig(install_path=install_path, launcher="python"))
    client.start()

    async def mark_inactive(self) -> None:  # noqa: ANN001
        self.events.append("ping")
        client._started = False

    monkeypatch.setattr(session, "ping", mark_inactive)
    health = client.health_check(required_tools={"get_game_overview", "end_turn"})

    assert health.ok is False
    assert health.started is False
    assert health.initialized is True
    assert health.missing_required_tools == set()
    assert "not active" in health.message

    client._started = True
    client.stop()


def test_health_check_reports_unstarted_client_without_spawning(install_path: Path) -> None:
    client = Civ6McpClient(Civ6McpConfig(install_path=install_path, launcher="python"))

    default_health = client.health_check()
    health = client.health_check(required_tools={"get_game_overview"})

    assert default_health.ok is False
    assert default_health.missing_required_tools == {"get_game_overview", "end_turn"}
    assert health.ok is False
    assert health.started is False
    assert health.initialized is False
    assert health.missing_required_tools == {"get_game_overview"}
    assert "not started" in health.message


def test_start_creates_named_background_event_loop_and_retains_state(
    fake_mcp_sdk,
    install_path: Path,
) -> None:
    session, _stdio = fake_mcp_sdk
    caller_thread = threading.current_thread().name
    client = Civ6McpClient(Civ6McpConfig(install_path=install_path, launcher="python"))

    client.start()

    try:
        loop = client._loop
        loop_thread = client._loop_thread

        assert loop is not None
        assert loop_thread is not None
        assert loop_thread.name == "civ6-mcp-loop"
        assert loop_thread.daemon is True
        assert loop_thread.is_alive()
        assert loop.is_running()
        assert session.initialize_thread_name == "civ6-mcp-loop"
        assert session.initialize_thread_name != caller_thread
        assert session.initialize_loop_id == id(loop)
    finally:
        client.stop()

    assert client._loop is None
    assert client._loop_thread is None


def test_start_surfaces_invalid_install_path_without_spawning(tmp_path: Path, monkeypatch) -> None:
    thread_started = False

    def fail_if_spawned(self) -> None:  # noqa: ANN001
        nonlocal thread_started
        thread_started = True
        raise AssertionError("civ6-mcp client must not spawn for invalid install path")

    monkeypatch.setattr("threading.Thread.start", fail_if_spawned)
    client = Civ6McpClient(Civ6McpConfig(install_path=tmp_path / "missing", launcher="python"))

    with pytest.raises(Civ6McpUnavailableError, match="Failed to start Civ6McpClient"):
        client.start()

    assert thread_started is False
    assert client.tool_names == set()


def test_start_surfaces_missing_launcher_without_spawning(
    install_path: Path,
    monkeypatch,
) -> None:
    thread_started = False

    def fail_if_spawned(self) -> None:  # noqa: ANN001
        nonlocal thread_started
        thread_started = True
        raise AssertionError("civ6-mcp client must not spawn for invalid launcher command")

    monkeypatch.setattr("threading.Thread.start", fail_if_spawned)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    client = Civ6McpClient(Civ6McpConfig(install_path=install_path, launcher="python"))

    with pytest.raises(Civ6McpUnavailableError, match="Failed to start Civ6McpClient"):
        client.start()

    assert thread_started is False
    assert client.tool_names == set()


def test_health_check_wraps_ping_failure(fake_mcp_sdk, install_path: Path, monkeypatch) -> None:
    session, _stdio = fake_mcp_sdk

    async def fail_ping(self) -> None:  # noqa: ANN001
        raise RuntimeError("transport closed")

    monkeypatch.setattr(session, "ping", fail_ping)
    client = Civ6McpClient(Civ6McpConfig(install_path=install_path, launcher="python"))
    client.start()

    with pytest.raises(Civ6McpError, match="health check failed"):
        client.health_check()

    client.stop()
