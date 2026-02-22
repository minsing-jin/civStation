"""
Screen Streamer — Periodic screen capture → WebSocket video_frame broadcast.

Captures the main monitor using `mss`, resizes with Pillow, encodes as JPEG,
and sends base64-encoded frames via the WebSocketManager.

Environment variables:
    STREAM_FPS           Target frames per second (default 5)
    STREAM_WIDTH         Max frame width in pixels   (default 960)
    STREAM_JPEG_QUALITY  JPEG quality 1-95           (default 60)
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time

from PIL import Image

logger = logging.getLogger(__name__)

# ── Configuration from env ─────────────────────────────────────────
STREAM_FPS: int = int(os.environ.get("STREAM_FPS", "5"))
STREAM_WIDTH: int = int(os.environ.get("STREAM_WIDTH", "960"))
STREAM_JPEG_QUALITY: int = int(os.environ.get("STREAM_JPEG_QUALITY", "60"))


def _capture_and_encode() -> tuple[str, int, int]:
    """Capture the main monitor, resize, JPEG-encode, return (b64, w, h).

    Runs synchronously — intended to be called from a thread-pool executor
    so that the event loop is never blocked.
    """
    import mss

    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Primary monitor (0 = all monitors combined)
        raw = sct.grab(monitor)

    # mss → PIL (BGRA → RGB)
    img = Image.frombytes("RGB", raw.size, raw.rgb)

    # Resize keeping aspect ratio
    if img.width > STREAM_WIDTH:
        ratio = STREAM_WIDTH / img.width
        new_h = int(img.height * ratio)
        img = img.resize((STREAM_WIDTH, new_h), Image.LANCZOS)

    # JPEG encode
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=STREAM_JPEG_QUALITY)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return b64, img.width, img.height


class ScreenStreamer:
    """Async screen capture loop that broadcasts video_frame messages.

    Usage::

        streamer = ScreenStreamer(ws_manager)
        streamer.start()   # spawns the async task
        ...
        streamer.stop()    # cancels the task gracefully
    """

    def __init__(self, ws_manager) -> None:
        from computer_use_test.agent.modules.hitl.status_ui.websocket_manager import WebSocketManager

        self._ws: WebSocketManager = ws_manager
        self._future: asyncio.Future | None = None
        self._running = False

    # ── lifecycle ───────────────────────────────────────────────────

    def start(self) -> None:
        """Schedule the capture loop on the WebSocketManager's event loop."""
        if self._running:
            return

        loop = self._ws._loop
        if loop is None:
            logger.warning("ScreenStreamer: no event loop available yet, deferring start")
            return

        self._running = True
        self._future = asyncio.run_coroutine_threadsafe(self._loop(), loop)
        logger.info(f"ScreenStreamer started (fps={STREAM_FPS}, width={STREAM_WIDTH}, jpeg_q={STREAM_JPEG_QUALITY})")

    def stop(self) -> None:
        """Stop the capture loop."""
        self._running = False
        if self._future and not self._future.done():
            self._future.cancel()
            self._future = None
        logger.info("ScreenStreamer stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ── capture loop ────────────────────────────────────────────────

    async def _loop(self) -> None:
        interval = 1.0 / max(1, STREAM_FPS)

        while self._running:
            if self._ws.connection_count == 0:
                # No clients — sleep longer to save CPU
                await asyncio.sleep(0.5)
                continue

            t0 = time.monotonic()
            try:
                loop = asyncio.get_running_loop()
                b64, w, h = await loop.run_in_executor(None, _capture_and_encode)

                frame_msg = {
                    "type": "video_frame",
                    "mime": "image/jpeg",
                    "data": b64,
                    "width": w,
                    "height": h,
                    "ts": int(time.time() * 1000),
                }
                await self._ws._async_broadcast(frame_msg)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("ScreenStreamer: capture/send error")
                await asyncio.sleep(1.0)  # back-off on error
                continue

            elapsed = time.monotonic() - t0
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self._running = False
