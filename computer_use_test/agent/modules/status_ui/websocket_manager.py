"""
WebSocket Manager — Connection management and broadcast for real-time HITL.

Manages WebSocket connections from the dashboard, provides both async
(FastAPI event loop) and sync (agent main thread) broadcast methods.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """WebSocket connection manager with thread-safe broadcast."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, ws: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await ws.accept()
        async with self._lock:
            self._connections.add(ws)
            if self._loop is None:
                self._loop = asyncio.get_running_loop()
        logger.info(f"WebSocket connected (total: {len(self._connections)})")

    async def disconnect(self, ws: WebSocket) -> None:
        """Unregister a WebSocket connection."""
        async with self._lock:
            self._connections.discard(ws)
        logger.info(f"WebSocket disconnected (total: {len(self._connections)})")

    async def _async_broadcast(self, data: dict[str, Any]) -> None:
        """Broadcast data to all connected clients (called within asyncio loop)."""
        async with self._lock:
            stale: list[WebSocket] = []
            for ws in self._connections:
                try:
                    await ws.send_json(data)
                except Exception:
                    stale.append(ws)
            for ws in stale:
                self._connections.discard(ws)

    def broadcast(self, data: dict[str, Any]) -> None:
        """
        Synchronous broadcast — safe to call from the agent main thread.

        Schedules the async broadcast on the FastAPI event loop via
        ``asyncio.run_coroutine_threadsafe``.
        """
        if self._loop is None or not self._connections:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._async_broadcast(data), self._loop)
        except RuntimeError:
            pass  # Event loop closed — ignore

    @property
    def connection_count(self) -> int:
        """Number of active WebSocket connections."""
        return len(self._connections)
