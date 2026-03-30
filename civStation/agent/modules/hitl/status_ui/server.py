"""
Status Server — FastAPI + Uvicorn daemon for the real-time dashboard.

Endpoints:
  GET  /            — HTML dashboard
  GET  /api/status  — JSON agent status snapshot
  POST /api/directive — Submit a directive from the web UI
  WS   /ws          — WebSocket for real-time bidirectional communication
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import threading
from typing import TYPE_CHECKING

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from civStation.agent.modules.hitl.command_queue import CommandQueue, Directive, DirectiveType
from civStation.agent.modules.hitl.status_ui.dashboard import DASHBOARD_HTML
from civStation.agent.modules.hitl.status_ui.screen_streamer import ScreenStreamer
from civStation.agent.modules.hitl.status_ui.websocket_manager import WebSocketManager

if TYPE_CHECKING:
    from civStation.agent.modules.hitl.agent_gate import AgentGate
    from civStation.agent.modules.hitl.status_ui.state_bridge import AgentStateBridge
    from civStation.utils.chatapp.discussion.discussion_engine import StrategyDiscussion

logger = logging.getLogger(__name__)

# Keyword mapping for web UI quick commands
_QUICK_COMMANDS = {
    "stop": DirectiveType.STOP,
    "pause": DirectiveType.PAUSE,
    "resume": DirectiveType.RESUME,
}


def _get_local_ip() -> str:
    """Detect the machine's LAN IP address for mobile access."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


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
        ws_manager: WebSocketManager | None = None,
        agent_gate: AgentGate | None = None,
        discussion_engine: StrategyDiscussion | None = None,
        host: str = "0.0.0.0",
        port: int = 8765,
    ) -> None:
        self._bridge = state_bridge
        self._queue = command_queue
        self._ws_manager = ws_manager or WebSocketManager()
        self._agent_gate = agent_gate
        self._discussion_engine = discussion_engine
        self._host = host
        self._port = port
        self._thread: threading.Thread | None = None
        self._server = None
        self._screen_streamer = ScreenStreamer(self._ws_manager)

    def _create_app(self):
        """Build the FastAPI application."""
        app = FastAPI(title="Civ6 Agent Status", docs_url=None, redoc_url=None)

        bridge = self._bridge
        queue = self._queue
        ws_manager = self._ws_manager

        @app.get("/", response_class=HTMLResponse)
        async def dashboard():
            return DASHBOARD_HTML

        @app.get("/api/status")
        async def get_status():
            status = bridge.get_status()
            return JSONResponse(content=status.to_dict())

        port = self._port

        @app.get("/api/connection-info")
        async def connection_info():
            ip = _get_local_ip()
            url = f"http://{ip}:{port}"
            return JSONResponse(content={"ip": ip, "port": port, "url": url})

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

        # --- Agent control endpoints (for external controller) ---
        controller = self._agent_gate

        @app.get("/api/agent/state")
        async def agent_state():
            if not controller:
                return JSONResponse(content={"error": "controller not initialized"}, status_code=503)
            return JSONResponse(content={"state": controller.state.value})

        @app.post("/api/agent/start")
        async def agent_start():
            if not controller:
                return JSONResponse(content={"error": "controller not initialized"}, status_code=503)
            ok = controller.start()
            return JSONResponse(
                content={"ok": ok, "state": controller.state.value},
                status_code=200 if ok else 409,
            )

        @app.post("/api/agent/pause")
        async def agent_pause():
            if not controller:
                return JSONResponse(content={"error": "controller not initialized"}, status_code=503)
            ok = controller.pause()
            return JSONResponse(
                content={"ok": ok, "state": controller.state.value},
                status_code=200 if ok else 409,
            )

        @app.post("/api/agent/resume")
        async def agent_resume():
            if not controller:
                return JSONResponse(content={"error": "controller not initialized"}, status_code=503)
            ok = controller.resume()
            return JSONResponse(
                content={"ok": ok, "state": controller.state.value},
                status_code=200 if ok else 409,
            )

        @app.post("/api/agent/stop")
        async def agent_stop():
            if not controller:
                return JSONResponse(content={"error": "controller not initialized"}, status_code=503)
            ok = controller.stop()
            return JSONResponse(
                content={"ok": ok, "state": controller.state.value},
                status_code=200 if ok else 409,
            )

        # --- Discussion endpoints ---
        discussion = self._discussion_engine

        @app.post("/api/discuss")
        async def discuss(body: dict):
            if not discussion:
                return JSONResponse(content={"error": "discussion engine not initialized"}, status_code=503)

            user_id = body.get("user_id", "web_user")
            message = body.get("message", "").strip()
            mode_str = body.get("mode", "in_game")
            language = body.get("language", "ko")

            if not message:
                return JSONResponse(content={"error": "empty message"}, status_code=400)

            # Get or create session
            session = discussion.get_active_session(user_id)
            if not session:
                from civStation.utils.chatapp.discussion.discussion_schemas import DiscussionMode

                mode_map = {
                    "pre_game": DiscussionMode.PRE_GAME,
                    "in_game": DiscussionMode.IN_GAME,
                    "post_turn": DiscussionMode.POST_TURN,
                }
                mode = mode_map.get(mode_str, DiscussionMode.IN_GAME)
                session_id = discussion.create_session(user_id, mode)
            else:
                session_id = session.session_id

            # Run VLM call in thread pool to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, discussion.process_message, session_id, message, language)

            session = discussion._sessions.get(session_id)
            msg_count = len(session.messages) if session else 0

            return JSONResponse(content={"session_id": session_id, "response": response, "message_count": msg_count})

        @app.post("/api/discuss/finalize")
        async def discuss_finalize(body: dict):
            if not discussion:
                return JSONResponse(content={"error": "discussion engine not initialized"}, status_code=503)

            user_id = body.get("user_id", "web_user")
            session = discussion.get_active_session(user_id)
            if not session:
                return JSONResponse(content={"error": "no active session"}, status_code=404)

            loop = asyncio.get_running_loop()
            strategy = await loop.run_in_executor(None, discussion.finalize_session, session.session_id)

            strategy_str = str(strategy) if strategy else None
            return JSONResponse(content={"ok": True, "strategy": strategy_str})

        @app.get("/api/discuss/status")
        async def discuss_status(user_id: str = "web_user"):
            if not discussion:
                return JSONResponse(content={"error": "discussion engine not initialized"}, status_code=503)

            reference_snapshot = discussion.get_reference_snapshot()
            session = discussion.get_active_session(user_id)
            if not session:
                return JSONResponse(
                    content={
                        "active": False,
                        "messages": [],
                        "current_strategy": reference_snapshot["current_strategy"],
                        "combined_context": reference_snapshot["combined_context"],
                    }
                )

            messages = [{"role": m.role, "content": m.content} for m in session.messages]
            return JSONResponse(
                content={
                    "active": True,
                    "session_id": session.session_id,
                    "mode": session.mode.value,
                    "message_count": len(messages),
                    "messages": messages,
                    "current_strategy": reference_snapshot["current_strategy"],
                    "combined_context": reference_snapshot["combined_context"],
                }
            )

        screen_streamer = self._screen_streamer

        @app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            client = ws.client
            client_str = f"{client.host}:{client.port}" if client else "unknown"

            await ws_manager.connect(ws)
            logger.info(f"WS connected from {client_str}")

            # Auto-start screen streamer when first client connects
            if not screen_streamer.is_running:
                screen_streamer.start()

            # Send initial status snapshot
            try:
                status = bridge.get_status()
                await ws.send_json({"type": "status", "data": status.to_dict()})
            except Exception as e:
                logger.warning(f"WS initial status send failed ({client_str}): {e}")

            try:
                while True:
                    # Use receive_text() so a non-JSON frame doesn't kill the loop
                    try:
                        text = await ws.receive_text()
                    except WebSocketDisconnect:
                        logger.info(f"WS client disconnected ({client_str})")
                        break

                    logger.debug(f"WS recv ({client_str}): {text[:300]}")

                    # --- JSON parse (safe) ---
                    try:
                        raw = json.loads(text)
                    except (json.JSONDecodeError, ValueError):
                        logger.warning(f"WS malformed JSON from {client_str}: {text[:100]!r}")
                        try:
                            await ws.send_json({"type": "message", "message": "Invalid JSON payload"})
                        except Exception:
                            pass
                        continue

                    msg_type = raw.get("type", "")
                    mode = raw.get("mode", "")

                    # ── NEW: type-based protocol (bridge / Node.js ws) ──────────

                    if msg_type == "control":
                        # {"type":"control","action":"start|stop|pause|resume"}
                        action = str(raw.get("action", "")).lower()
                        if not controller:
                            logger.warning(f"WS control '{action}' ignored: AgentGate not initialized")
                            try:
                                await ws.send_json({"type": "error", "message": "AgentGate not initialized"})
                            except Exception:
                                pass
                            continue

                        _control_map = {
                            "start": controller.start,
                            "stop": controller.stop,
                            "pause": controller.pause,
                            "resume": controller.resume,
                        }
                        if action not in _control_map:
                            logger.warning(f"WS unknown control action: {action!r}")
                            try:
                                await ws.send_json({"type": "error", "message": f"Unknown action: {action}"})
                            except Exception:
                                pass
                            continue

                        ok = _control_map[action]()
                        logger.info(
                            f"WS control '{action}' ({client_str}) → {'ok' if ok else 'rejected: invalid state'}"
                        )

                        # Push updated status back immediately
                        try:
                            status = bridge.get_status()
                            await ws.send_json({"type": "status", "data": status.to_dict()})
                        except Exception:
                            pass
                        continue

                    if msg_type == "command":
                        # {"type":"command","content":"..."}
                        content = raw.get("content", "")
                        if not content and content != 0:
                            logger.debug(f"WS empty command from {client_str}")
                            try:
                                await ws.send_json({"type": "message", "message": "empty command"})
                            except Exception:
                                pass
                            continue

                        if not isinstance(content, str):
                            content = json.dumps(content)

                        directive = Directive(
                            directive_type=DirectiveType.CHANGE_STRATEGY,
                            payload=content,
                            source="websocket_bridge",
                        )
                        queue.push(directive)
                        logger.info(f"WS command queued ({client_str}): {content[:80]!r}")
                        continue

                    if msg_type == "ping":
                        # Client keepalive — silently ignored
                        continue

                    # ── Legacy: mode-based protocol (dashboard) ───────────────
                    content = raw.get("content", "")

                    if mode == "primitive":
                        directive = Directive(
                            directive_type=DirectiveType.PRIMITIVE_OVERRIDE,
                            payload=content if isinstance(content, str) else json.dumps(content),
                            source="websocket",
                        )
                    elif mode == "high_level":
                        directive = Directive(
                            directive_type=DirectiveType.CHANGE_STRATEGY,
                            payload=content,
                            source="websocket",
                        )
                    elif mode in ("stop", "pause", "resume"):
                        dtype = _QUICK_COMMANDS.get(mode, DirectiveType.CUSTOM)
                        directive = Directive(directive_type=dtype, payload=content, source="websocket")
                    else:
                        logger.debug(f"WS unhandled msg ({client_str}): type={msg_type!r} mode={mode!r}")
                        continue

                    queue.push(directive)

            except WebSocketDisconnect:
                logger.info(f"WS disconnected ({client_str})")
            except Exception as e:
                logger.warning(f"WS unexpected error ({client_str}): {e}", exc_info=True)
            finally:
                await ws_manager.disconnect(ws)
                logger.info(f"WS cleaned up ({client_str})")

        return app

    def start(self) -> None:
        """Start the server on a daemon thread."""
        if self._thread and self._thread.is_alive():
            return

        app = self._create_app()

        def _run():
            import uvicorn

            config = uvicorn.Config(
                app,
                host=self._host,
                port=self._port,
                log_level="warning",
                ws_ping_interval=20,  # send ping every 20s to keep connection alive
                ws_ping_timeout=30,  # drop connection if no pong within 30s
            )
            self._server = uvicorn.Server(config)
            self._server.run()

        self._thread = threading.Thread(target=_run, daemon=True, name="StatusServer")
        self._thread.start()
        logger.info(f"StatusServer started on http://{self._host}:{self._port}")

    def stop(self) -> None:
        """Signal the server to shut down."""
        self._screen_streamer.stop()
        if self._server:
            self._server.should_exit = True
        logger.info("StatusServer stop requested")
