"""
Relay Client — WebSocket client connecting to an external relay server.

The relay server bridges remote/mobile controllers to the local agent
when direct LAN access is not available (e.g., NAT traversal).

Flow:
  1. Connect  → wss://relay-host/ws
  2. Auth     → {"type":"auth","token":"<token>"}
  3. Pair QR  → {"type":"create_pair_qr"} → receive pairUrl → print QR
  4. Listen   → route incoming messages to AgentGate / CommandQueue
  5. Status   → send_status(data) broadcasts agent state to relay
  6. Reconnect on disconnect (5 s delay)

Incoming message protocol:
  {"type":"control","action":"start|stop|pause|resume"}  → AgentGate
  {"type":"command","content":"..."}                     → CommandQueue (CHANGE_STRATEGY)

Outgoing message protocol:
  {"type":"status","data":{...}}   → relay → mobile controller
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from civStation.agent.modules.hitl.agent_gate import AgentGate
    from civStation.agent.modules.hitl.command_queue import CommandQueue

logger = logging.getLogger(__name__)

_RECONNECT_DELAY = 5.0  # seconds between reconnect attempts


def _print_qr(url: str) -> None:
    """Print QR code to console. Falls back to URL display if qrcode not installed."""
    try:
        import qrcode  # type: ignore[import]

        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        print("\n" + "=" * 60)
        print("  Relay QR — scan to connect your mobile controller")
        print("=" * 60)
        qr.print_ascii(invert=True)
        print(f"  URL: {url}")
        print("=" * 60 + "\n")
    except ImportError:
        print("\n" + "=" * 60)
        print("  Relay QR pair URL (install 'qrcode' for ASCII QR)")
        print("=" * 60)
        print(f"  {url}")
        print("=" * 60 + "\n")


class RelayClient:
    """
    Async WebSocket client for the external relay server.

    Runs its own asyncio event loop in a daemon thread so it does not
    interfere with FastAPI's event loop or the main agent thread.

    Usage:
        client = RelayClient(url="wss://control.example.com/ws", token="xxx",
                             agent_gate=gate, command_queue=queue)
        client.start()   # non-blocking
        ...
        client.send_status(status_dict)  # thread-safe
        client.stop()
    """

    def __init__(
        self,
        url: str,
        token: str,
        agent_gate: AgentGate | None = None,
        command_queue: CommandQueue | None = None,
    ) -> None:
        self._url = url
        self._token = token
        self._agent_gate = agent_gate
        self._command_queue = command_queue

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Cached pair URL so state_bridge can expose it in status
        self.pair_url: str = ""

        # asyncio.Queue for outbound messages (thread-safe via put_nowait)
        self._outbound: asyncio.Queue[dict[str, Any]] | None = None

    # ------------------------------------------------------------------
    # Public API (called from main / agent threads)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the relay client in a daemon thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="RelayClient")
        self._thread.start()
        logger.info(f"RelayClient started → {self._url}")

    def stop(self) -> None:
        """Signal the relay client to stop."""
        self._stop_event.set()
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        logger.info("RelayClient stop requested")

    def send_status(self, data: dict[str, Any]) -> None:
        """
        Thread-safe: enqueue a status broadcast to the relay server.
        Safe to call from the main agent thread.
        """
        if self._outbound is None or self._loop is None or self._loop.is_closed():
            return
        try:
            self._loop.call_soon_threadsafe(self._outbound.put_nowait, {"type": "status", "data": data})
        except RuntimeError:
            pass  # loop closed

    # ------------------------------------------------------------------
    # Internal: asyncio event loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Entry point for the daemon thread — owns the asyncio loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._reconnect_loop())
        except Exception as e:
            logger.error(f"RelayClient loop exited with error: {e}")
        finally:
            self._loop.close()

    async def _reconnect_loop(self) -> None:
        """Keep reconnecting until stop() is called."""
        while not self._stop_event.is_set():
            try:
                await self._connect_and_run()
            except Exception as e:
                logger.warning(f"RelayClient disconnected: {e}")

            if self._stop_event.is_set():
                break
            logger.info(f"RelayClient reconnecting in {_RECONNECT_DELAY}s...")
            await asyncio.sleep(_RECONNECT_DELAY)

    async def _connect_and_run(self) -> None:
        """Connect, authenticate, pair, then run send/receive loops."""
        import websockets  # type: ignore[import]

        self._outbound = asyncio.Queue()

        async with websockets.connect(self._url) as ws:
            logger.info(f"RelayClient connected to {self._url}")

            # 1. Authenticate
            await ws.send(json.dumps({"type": "auth", "token": self._token}))
            raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
            auth_resp = json.loads(raw)
            if auth_resp.get("status") != "ok":
                raise RuntimeError(f"Relay auth failed: {auth_resp}")
            logger.info("RelayClient authenticated")

            # 2. Request pair QR
            await ws.send(json.dumps({"type": "create_pair_qr"}))
            raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
            qr_resp = json.loads(raw)
            pair_url = qr_resp.get("url", "")
            if pair_url:
                self.pair_url = pair_url
                _print_qr(pair_url)
            else:
                logger.warning(f"Unexpected pair_qr response: {qr_resp}")

            # 3. Run send + receive concurrently
            await asyncio.gather(
                self._recv_loop(ws),
                self._send_loop(ws),
            )

    async def _recv_loop(self, ws: Any) -> None:
        """Receive and dispatch incoming relay messages."""
        async for raw in ws:
            if self._stop_event.is_set():
                break
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(f"RelayClient: malformed JSON received: {raw!r}")
                continue

            msg_type = msg.get("type", "")

            if msg_type == "control":
                self._handle_control(msg.get("action", ""))
            elif msg_type == "command":
                self._handle_command(msg.get("content", ""))
            else:
                logger.debug(f"RelayClient: unknown message type '{msg_type}'")

    async def _send_loop(self, ws: Any) -> None:
        """Drain outbound queue and send to relay."""
        while not self._stop_event.is_set():
            try:
                payload = await asyncio.wait_for(self._outbound.get(), timeout=1.0)
                await ws.send(json.dumps(payload))
            except asyncio.TimeoutError:
                continue  # normal — no status update, keep waiting
            except Exception as e:
                logger.warning(f"RelayClient send error: {e}")
                break

    # ------------------------------------------------------------------
    # Message handlers (called from asyncio thread — thread-safe writes)
    # ------------------------------------------------------------------

    def _handle_control(self, action: str) -> None:
        """Route control actions to AgentGate."""
        if not self._agent_gate:
            logger.debug(f"RelayClient: control '{action}' ignored (no AgentGate)")
            return

        action = action.lower()
        if action == "start":
            ok = self._agent_gate.start()
        elif action == "stop":
            ok = self._agent_gate.stop()
        elif action == "pause":
            ok = self._agent_gate.pause()
        elif action == "resume":
            ok = self._agent_gate.resume()
        else:
            logger.warning(f"RelayClient: unknown control action '{action}'")
            return

        logger.info(f"RelayClient: control '{action}' → {'ok' if ok else 'rejected (invalid state)'}")

    def _handle_command(self, content: str) -> None:
        """Push text commands to CommandQueue as CHANGE_STRATEGY."""
        if not content:
            return
        if not self._command_queue:
            logger.debug("RelayClient: command ignored (no CommandQueue)")
            return

        from civStation.agent.modules.hitl.command_queue import Directive, DirectiveType

        directive = Directive(
            directive_type=DirectiveType.CHANGE_STRATEGY,
            payload=content,
            source="relay",
        )
        self._command_queue.push(directive)
        logger.info(f"RelayClient: command queued → '{content[:80]}'")
