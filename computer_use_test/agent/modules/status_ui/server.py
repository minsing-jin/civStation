"""
Status Server — FastAPI + Uvicorn daemon for the real-time dashboard.

Endpoints:
  GET  /            — HTML dashboard
  GET  /api/status  — JSON agent status snapshot
  POST /api/directive — Submit a directive from the web UI
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from computer_use_test.agent.modules.hitl.command_queue import CommandQueue, Directive, DirectiveType
from computer_use_test.agent.modules.status_ui.dashboard import DASHBOARD_HTML

if TYPE_CHECKING:
    from computer_use_test.agent.modules.status_ui.state_bridge import AgentStateBridge

logger = logging.getLogger(__name__)

# Keyword mapping for web UI quick commands
_QUICK_COMMANDS = {
    "stop": DirectiveType.STOP,
    "pause": DirectiveType.PAUSE,
    "resume": DirectiveType.RESUME,
}


class StatusServer:
    """
    FastAPI server running on a daemon thread.

    Provides the web dashboard and API endpoints for monitoring
    and sending directives to the running agent.
    """

    def __init__(
        self,
        state_bridge: AgentStateBridge,
        command_queue: CommandQueue,
        host: str = "0.0.0.0",
        port: int = 8765,
    ) -> None:
        self._bridge = state_bridge
        self._queue = command_queue
        self._host = host
        self._port = port
        self._thread: threading.Thread | None = None
        self._server = None

    def _create_app(self):
        """Build the FastAPI application."""
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse

        app = FastAPI(title="Civ6 Agent Status", docs_url=None, redoc_url=None)

        bridge = self._bridge
        queue = self._queue

        @app.get("/", response_class=HTMLResponse)
        async def dashboard():
            return DASHBOARD_HTML

        @app.get("/api/status")
        async def get_status():
            status = bridge.get_status()
            return JSONResponse(content=status.to_dict())

        @app.post("/api/directive")
        async def post_directive(body: dict):
            text = body.get("text", "").strip()
            if not text:
                return JSONResponse(content={"error": "empty text"}, status_code=400)

            lower = text.lower()
            dtype = _QUICK_COMMANDS.get(lower, DirectiveType.CHANGE_STRATEGY)
            directive = Directive(directive_type=dtype, payload=text, source="web_ui")
            queue.push(directive)
            return JSONResponse(content={"ok": True, "type": dtype.value})

        return app

    def start(self) -> None:
        """Start the server on a daemon thread."""
        if self._thread and self._thread.is_alive():
            return

        app = self._create_app()

        def _run():
            import uvicorn

            config = uvicorn.Config(app, host=self._host, port=self._port, log_level="warning")
            self._server = uvicorn.Server(config)
            self._server.run()

        self._thread = threading.Thread(target=_run, daemon=True, name="StatusServer")
        self._thread.start()
        logger.info(f"StatusServer started on http://{self._host}:{self._port}")

    def stop(self) -> None:
        """Signal the server to shut down."""
        if self._server:
            self._server.should_exit = True
        logger.info("StatusServer stop requested")
