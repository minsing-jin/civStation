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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Civ6McpError(RuntimeError):
    """Raised when the civ6-mcp server returns an error or fails to start."""


class Civ6McpUnavailableError(Civ6McpError):
    """Raised when the civ6-mcp install or `uv`/`python` runtime is missing."""


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
        if not self.install_path.is_dir():
            raise Civ6McpUnavailableError(
                f"civ6-mcp install path does not exist: {self.install_path}. "
                "Set --civ6-mcp-path or the CIV6_MCP_PATH env var to your "
                "civ6-mcp checkout."
            )
        pyproject = self.install_path / "pyproject.toml"
        if not pyproject.is_file():
            raise Civ6McpUnavailableError(
                f"civ6-mcp install path missing pyproject.toml: {self.install_path}. "
                "Did you `git clone https://github.com/lmwilki/civ6-mcp` and `uv sync` there?"
            )
        if self.launcher == "uv" and shutil.which("uv") is None:
            raise Civ6McpUnavailableError(
                "Selected launcher is `uv` but the `uv` binary is not on PATH. "
                "Install astral-sh/uv or pass --civ6-mcp-launcher python."
            )

    def server_command(self) -> list[str]:
        """Build the argv used to launch the civ6-mcp stdio server."""
        if self.launcher == "uv":
            return [
                "uv",
                "run",
                "--directory",
                str(self.install_path),
                "civ-mcp",
                *self.extra_args,
            ]
        if self.launcher == "python":
            return [
                "python",
                "-m",
                "civ_mcp",
                *self.extra_args,
            ]
        raise Civ6McpUnavailableError(f"Unsupported launcher: {self.launcher!r}")


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

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Invoke a server tool and return the textual result.

        The upstream server returns narrated text (see end_turn.py et al.).
        Errors arrive either as text-prefixed bodies (`"Error: ..."`) or as
        JSON-RPC `isError: true`. Both are surfaced; callers should pattern-
        match the prefixes documented in civ6-mcp/AGENTS.md.
        """
        if not self._started:
            raise Civ6McpError("Civ6McpClient.start() must be called before call_tool().")
        future = self._submit(self._async_call_tool(name, arguments or {}))
        try:
            return future.result(timeout=self.config.call_timeout_seconds)
        except TimeoutError as exc:
            raise Civ6McpError(f"civ6-mcp tool '{name}' timed out after {self.config.call_timeout_seconds}s") from exc

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

        params = StdioServerParameters(
            command=self.config.server_command()[0],
            args=self.config.server_command()[1:],
            env=env,
        )

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

    async def _async_stop(self) -> None:
        try:
            if self._session_ctx is not None:
                await self._session_ctx.__aexit__(None, None, None)
        finally:
            if self._stdio_ctx is not None:
                await self._stdio_ctx.__aexit__(None, None, None)

    async def _async_call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if self._session is None:
            raise Civ6McpError("MCP session not initialized.")
        if name not in self._tool_names:
            raise Civ6McpError(f"Unknown civ6-mcp tool '{name}'. Available: {sorted(self._tool_names)[:8]}...")
        result = await self._session.call_tool(name, arguments)

        is_error = getattr(result, "isError", False)
        text_parts: list[str] = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                text_parts.append(text)
            elif isinstance(block, dict) and "text" in block:
                text_parts.append(str(block["text"]))
        body = "\n".join(text_parts).strip()

        if is_error:
            raise Civ6McpError(f"civ6-mcp tool '{name}' returned isError: {body or '<empty>'}")
        return body
