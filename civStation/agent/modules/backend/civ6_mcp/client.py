"""Stdio MCP client for the upstream civ6-mcp server.

The upstream project ships a FastMCP server (`civ-mcp`) that talks stdio
JSON-RPC. We launch it as a subprocess via `uv run --directory <path> civ-mcp`
and drive it with the official `mcp` Python SDK that civStation already
depends on.

Threading model:
    The `mcp` SDK is asyncio-native. We expose a synchronous facade
    (`Civ6McpClient.call_tool(...)`) that runs an internal asyncio loop on
    a dedicated background thread. This lets the existing main-thread turn
    loop call us the same way it calls pyautogui — no `async def` allowed
    to leak into turn_executor.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import threading
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from civStation.agent.modules.backend.civ6_mcp.response import (
    Civ6McpNormalizedResult,
    normalize_mcp_response_timeout,
    normalize_mcp_tool_result,
)

logger = logging.getLogger(__name__)


class Civ6McpError(RuntimeError):
    """Raised when the civ6-mcp server returns an error or fails to start."""


class Civ6McpUnavailableError(Civ6McpError):
    """Raised when the civ6-mcp install or `uv`/`python` runtime is missing."""


@dataclass(frozen=True)
class Civ6McpHealth:
    """Snapshot of the stdio MCP connection health."""

    ok: bool
    started: bool
    initialized: bool
    tool_count: int
    missing_required_tools: frozenset[str] = field(default_factory=frozenset)
    message: str = ""


DEFAULT_HEALTH_REQUIRED_TOOLS = frozenset({"get_game_overview", "end_turn"})
SUPPORTED_LAUNCHERS = frozenset({"uv", "python"})


@runtime_checkable
class Civ6McpClientProtocol(Protocol):
    """Public client surface consumed by civ6-mcp backend components."""

    @property
    def tool_names(self) -> set[str]:
        """Names of tools exposed by the connected civ6-mcp server."""
        ...

    def has_tool(self, name: str) -> bool:
        """Return whether the connected civ6-mcp server exposes ``name``."""
        ...

    def tool_schemas(self) -> dict[str, dict[str, Any]]:
        """Return MCP tool metadata keyed by tool name."""
        ...

    def health_check(self, required_tools: set[str] | frozenset[str] | None = None) -> Civ6McpHealth:
        """Return current MCP connection health after probing the session."""
        ...

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Invoke a civ6-mcp tool and return its textual response."""
        ...

    def call_tool_result(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Civ6McpNormalizedResult:
        """Invoke a civ6-mcp tool and return a normalized typed response."""
        ...


@dataclass
class Civ6McpConfig:
    """How to launch and talk to the upstream civ6-mcp server."""

    install_path: Path
    """Absolute path to the `civ6-mcp` checkout (where pyproject.toml lives)."""

    launcher: str = "uv"
    """Either `uv` (recommended) or `python` (uses civ6-mcp's own .venv)."""

    extra_args: list[str] = field(default_factory=list)
    """Extra CLI args passed after the launcher."""

    env_overrides: dict[str, str] = field(default_factory=dict)
    """Extra environment variables for the subprocess (CIV_MCP_*)."""

    client_name: str = "civStation"
    client_version: str = "0.1.0"

    startup_timeout_seconds: float = 30.0
    call_timeout_seconds: float = 120.0
    """`end_turn` can run 30–90s during late-game AI processing — set generous."""

    def _required_config_value(self, field_name: str) -> Any:
        value = getattr(self, field_name)
        if value is None or (isinstance(value, str) and not value.strip()):
            if field_name == "install_path":
                raise Civ6McpUnavailableError(
                    "civ6-mcp config missing required field: install_path. "
                    "Set --civ6-mcp-path or the CIV6_MCP_PATH env var to a local "
                    "github.com/lmwilki/civ6-mcp checkout."
                )
            if field_name == "launcher":
                raise Civ6McpUnavailableError(
                    "civ6-mcp config missing required field: launcher. "
                    "Set --civ6-mcp-launcher or the CIV6_MCP_LAUNCHER env var to either 'uv' or 'python'."
                )
            raise Civ6McpUnavailableError(f"civ6-mcp config missing required field: {field_name}")
        return value

    def _resolved_install_path(self) -> Path:
        raw_path = self._required_config_value("install_path")
        try:
            return Path(raw_path).expanduser().resolve()
        except TypeError as exc:
            raise Civ6McpUnavailableError(
                f"civ6-mcp config field install_path must be a path-like value, got {type(raw_path).__name__}. "
                "Pass --civ6-mcp-path or set CIV6_MCP_PATH to your civ6-mcp checkout."
            ) from exc

    def _unsupported_launcher_error(self, launcher: str) -> Civ6McpUnavailableError:
        supported = ", ".join(sorted(SUPPORTED_LAUNCHERS))
        return Civ6McpUnavailableError(
            f"Unsupported launcher for civ6-mcp: {launcher!r}. "
            f"Supported launchers: {supported}. "
            "Set --civ6-mcp-launcher or CIV6_MCP_LAUNCHER to one of those values."
        )

    def _validate_launcher_type(self, launcher: Any) -> str:
        if not isinstance(launcher, str):
            raise Civ6McpUnavailableError(
                f"civ6-mcp config field launcher must be a string, got {type(launcher).__name__}. "
                "Use --civ6-mcp-launcher uv|python or set CIV6_MCP_LAUNCHER."
            )
        return launcher

    @classmethod
    def from_environment(
        cls,
        *,
        install_path: str | Path | None = None,
        launcher: str | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> Civ6McpConfig:
        """Build a config from CLI/env hints with sensible fallbacks.

        Resolution order for install_path:
            1. explicit argument
            2. CIV6_MCP_PATH env var
            3. ~/civ6-mcp
        """
        path: Path
        if install_path is not None:
            path = Path(install_path).expanduser().resolve()
        elif os.environ.get("CIV6_MCP_PATH"):
            path = Path(os.environ["CIV6_MCP_PATH"]).expanduser().resolve()
        else:
            path = Path.home() / "civ6-mcp"

        chosen_launcher = launcher or os.environ.get("CIV6_MCP_LAUNCHER") or "uv"

        merged_env: dict[str, str] = {}
        for key in (
            "CIV_MCP_SAVE_FILE",
            "CIV_MCP_AGENT_MODEL",
            "CIV_MCP_METADATA",
            "CIV_MCP_TELEMETRY_BUCKET",
            "CIV_MCP_ALERT_WEBHOOK",
            "CIV_MCP_DISABLE_LUA",
        ):
            if os.environ.get(key):
                merged_env[key] = os.environ[key]
        if env_overrides:
            merged_env.update({k: v for k, v in env_overrides.items() if v is not None})

        return cls(install_path=path, launcher=chosen_launcher, env_overrides=merged_env)

    def validate(self) -> None:
        """Check the install path looks usable; raise Civ6McpUnavailableError otherwise."""
        install_path = self._resolved_install_path()
        launcher = self._required_config_value("launcher")
        launcher = self._validate_launcher_type(launcher)
        if launcher not in SUPPORTED_LAUNCHERS:
            raise self._unsupported_launcher_error(launcher)
        if not install_path.is_dir():
            raise Civ6McpUnavailableError(
                f"civ6-mcp install path does not exist: {install_path}. "
                "Set --civ6-mcp-path or the CIV6_MCP_PATH env var to your "
                "civ6-mcp checkout."
            )
        pyproject = install_path / "pyproject.toml"
        if not pyproject.is_file():
            raise Civ6McpUnavailableError(
                f"civ6-mcp install path missing pyproject.toml: {install_path}. "
                "Did you `git clone https://github.com/lmwilki/civ6-mcp` and `uv sync` there?"
            )
        if launcher == "uv" and shutil.which("uv") is None:
            raise Civ6McpUnavailableError(
                "Selected launcher is `uv` but the `uv` binary is not on PATH. "
                "Install astral-sh/uv or pass --civ6-mcp-launcher python."
            )

    def server_command(self) -> list[str]:
        """Build the argv used to launch the civ6-mcp stdio server."""
        launcher = self._required_config_value("launcher")
        launcher = self._validate_launcher_type(launcher)
        install_path = self._resolved_install_path()
        if launcher == "uv":
            return [
                "uv",
                "run",
                "--directory",
                str(install_path),
                "civ-mcp",
                *self.extra_args,
            ]
        if launcher == "python":
            return [
                "python",
                "-m",
                "civ_mcp",
                *self.extra_args,
            ]
        raise self._unsupported_launcher_error(launcher)


class Civ6McpClient:
    """Synchronous stdio client wrapper around the upstream MCP server.

    Lifecycle::

        client = Civ6McpClient(config)
        client.start()
        text = client.call_tool("get_game_overview")
        ...
        client.stop()

    Thread-safety: `call_tool` is safe to call from any thread; calls are
    serialized onto the internal event loop.
    """

    def __init__(self, config: Civ6McpConfig) -> None:
        self.config = config
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._session: Any | None = None  # mcp.ClientSession
        self._stdio_ctx: Any | None = None
        self._session_ctx: Any | None = None
        self._tool_names: set[str] = set()
        self._tool_schemas: dict[str, dict[str, Any]] = {}
        self._started = False
        self._lock = threading.RLock()

    # ----- lifecycle -------------------------------------------------

    def start(self) -> None:
        """Spawn the upstream server and complete the MCP handshake."""
        with self._lock:
            if self._started:
                return
            self.config.validate()

            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(
                target=self._run_loop_forever,
                name="civ6-mcp-loop",
                daemon=True,
            )
            self._loop_thread.start()

            try:
                self._submit(self._async_start()).result(timeout=self.config.startup_timeout_seconds)
            except Exception as exc:
                self._teardown_loop()
                raise Civ6McpError(f"Failed to start civ6-mcp server: {exc}") from exc

            self._started = True
            logger.info(
                "civ6-mcp server ready (path=%s, tools=%d)",
                self.config.install_path,
                len(self._tool_names),
            )

    def stop(self) -> None:
        """Tear down the MCP session and stop the event-loop thread."""
        with self._lock:
            if not self._started:
                self._teardown_loop()
                return
            try:
                self._submit(self._async_stop()).result(timeout=10.0)
            except Exception as exc:  # noqa: BLE001
                logger.warning("civ6-mcp stop encountered: %s", exc)
            finally:
                self._teardown_loop()
                self._started = False

    def __enter__(self) -> Civ6McpClient:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ----- tool surface ---------------------------------------------

    @property
    def tool_names(self) -> set[str]:
        return set(self._tool_names)

    def has_tool(self, name: str) -> bool:
        return name in self._tool_names

    def tool_schemas(self) -> dict[str, dict[str, Any]]:
        return dict(self._tool_schemas)

    def health_check(self, required_tools: set[str] | frozenset[str] | None = None) -> Civ6McpHealth:
        """Probe the MCP session and refresh the known tool catalog.

        The check is intentionally explicit. VLM callers never reach this
        method, and civ6-mcp startup failures remain civ6-mcp failures rather
        than falling back to computer-use.
        """
        required = frozenset(required_tools) if required_tools is not None else DEFAULT_HEALTH_REQUIRED_TOOLS
        if not self._started or self._session is None:
            missing = required.difference(self._tool_names)
            return Civ6McpHealth(
                ok=False,
                started=self._started,
                initialized=self._session is not None,
                tool_count=len(self._tool_names),
                missing_required_tools=missing,
                message="civ6-mcp client is not started or the MCP session is not initialized.",
            )

        future = self._submit(self._async_health_check(required))
        try:
            return future.result(timeout=self.config.startup_timeout_seconds)
        except FutureTimeoutError as exc:
            raise Civ6McpError(f"civ6-mcp health check timed out after {self.config.startup_timeout_seconds}s") from exc
        except Exception as exc:
            raise Civ6McpError(f"civ6-mcp health check failed: {exc}") from exc

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Invoke a server tool and return the textual result.

        The upstream server returns narrated text (see end_turn.py et al.).
        Errors arrive either as text-prefixed bodies (`"Error: ..."`) or as
        JSON-RPC `isError: true`. Both are surfaced; callers should pattern-
        match the prefixes documented in civ6-mcp/AGENTS.md.
        """
        result = self.call_tool_result(name, arguments)
        if result.timed_out:
            raise Civ6McpError(result.error)
        if result.is_error:
            raise Civ6McpError(f"civ6-mcp tool '{name}' returned isError: {result.error or '<empty>'}")
        return result.text

    def call_tool_result(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Civ6McpNormalizedResult:
        """Invoke a server tool and return a typed normalized result."""
        if not self._started:
            raise Civ6McpError("Civ6McpClient.start() must be called before call_tool().")
        request_arguments = arguments or {}
        future = self._submit(self._async_call_tool(name, request_arguments))
        try:
            return future.result(timeout=self.config.call_timeout_seconds)
        except FutureTimeoutError:
            future.cancel()
            logger.warning(
                "civ6-mcp tool timed out: tool=%s timeout=%ss",
                name,
                self.config.call_timeout_seconds,
            )
            return normalize_mcp_response_timeout(
                name,
                request_arguments,
                timeout_seconds=self.config.call_timeout_seconds,
            )

    # ----- internal asyncio plumbing --------------------------------

    def _run_loop_forever(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            try:
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                self._loop.run_until_complete(asyncio.sleep(0))
            except Exception:  # noqa: BLE001
                pass
            self._loop.close()

    def _submit(self, coro) -> Future:
        if self._loop is None:
            raise Civ6McpError("Event loop is not running.")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _teardown_loop(self) -> None:
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)
        self._loop = None
        self._loop_thread = None
        self._session = None
        self._stdio_ctx = None
        self._session_ctx = None
        self._tool_names = set()
        self._tool_schemas = {}

    async def _async_start(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        env: dict[str, str] = {**os.environ}
        env.update(self.config.env_overrides)
        server_command = self.config.server_command()

        params = StdioServerParameters(
            command=server_command[0],
            args=server_command[1:],
            env=env,
        )

        try:
            self._stdio_ctx = stdio_client(params)
            read_stream, write_stream = await self._stdio_ctx.__aenter__()

            # Some mcp SDK versions accept client_info; fall back gracefully if not.
            try:
                from mcp.types import Implementation  # type: ignore

                client_info = Implementation(
                    name=self.config.client_name,
                    version=self.config.client_version,
                )
                self._session_ctx = ClientSession(read_stream, write_stream, client_info=client_info)
            except Exception:  # noqa: BLE001
                self._session_ctx = ClientSession(read_stream, write_stream)

            self._session = await self._session_ctx.__aenter__()
            await self._session.initialize()
            await self._refresh_tool_catalog()
        except Exception:
            try:
                await self._async_stop()
            except Exception as stop_exc:  # noqa: BLE001
                logger.debug("civ6-mcp cleanup after failed start encountered: %s", stop_exc)
            self._session = None
            self._session_ctx = None
            self._stdio_ctx = None
            raise

    async def _refresh_tool_catalog(self) -> None:
        if self._session is None:
            raise Civ6McpError("MCP session not initialized.")
        tools_response = await self._session.list_tools()
        self._tool_names = set()
        self._tool_schemas = {}
        for tool in tools_response.tools:
            self._tool_names.add(tool.name)
            schema = getattr(tool, "inputSchema", None)
            if schema is not None:
                # `inputSchema` is a pydantic-style model on some SDKs, dict on others.
                if hasattr(schema, "model_dump"):
                    schema = schema.model_dump()
                elif hasattr(schema, "dict"):
                    schema = schema.dict()
            self._tool_schemas[tool.name] = {
                "description": getattr(tool, "description", "") or "",
                "input_schema": schema or {},
            }

    async def _async_health_check(self, required_tools: frozenset[str]) -> Civ6McpHealth:
        if self._session is None:
            missing = required_tools.difference(self._tool_names)
            return Civ6McpHealth(
                ok=False,
                started=self._started,
                initialized=False,
                tool_count=len(self._tool_names),
                missing_required_tools=missing,
                message="MCP session not initialized.",
            )

        ping = getattr(self._session, "ping", None)
        if callable(ping):
            await ping()
        await self._refresh_tool_catalog()

        missing = required_tools.difference(self._tool_names)
        ok = not missing
        message = "healthy" if ok else f"Missing required civ6-mcp tools: {sorted(missing)}"
        return Civ6McpHealth(
            ok=ok,
            started=self._started,
            initialized=True,
            tool_count=len(self._tool_names),
            missing_required_tools=missing,
            message=message,
        )

    async def _async_stop(self) -> None:
        try:
            if self._session_ctx is not None:
                await self._session_ctx.__aexit__(None, None, None)
        finally:
            if self._stdio_ctx is not None:
                await self._stdio_ctx.__aexit__(None, None, None)

    async def _async_call_tool(self, name: str, arguments: dict[str, Any]) -> Civ6McpNormalizedResult:
        if self._session is None:
            raise Civ6McpError("MCP session not initialized.")
        if name not in self._tool_names:
            raise Civ6McpError(f"Unknown civ6-mcp tool '{name}'. Available: {sorted(self._tool_names)[:8]}...")
        result = await self._session.call_tool(name, arguments)
        return normalize_mcp_tool_result(name, arguments, result)


__all__ = [
    "Civ6McpClient",
    "Civ6McpClientProtocol",
    "Civ6McpConfig",
    "Civ6McpError",
    "Civ6McpHealth",
    "Civ6McpNormalizedResult",
    "Civ6McpUnavailableError",
]
