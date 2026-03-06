"""
Turn Detector — Background worker that reads the in-game turn number.

Runs in a daemon thread using the same single-slot mailbox pattern as
ContextUpdater.  On each ``submit()``, the worker crops the turn-number
region from the screenshot and sends a small VLM call (max_tokens=64)
to read the digit.

Lazy calibration:
    The first ``submit()`` triggers an automatic calibration using the
    full screenshot.  This ensures the game screen is actually visible
    (unlike startup-time calibration which may capture a loading screen).

Self-correction:
    - If the cropped region yields ``null`` for ``_RECALIBRATION_THRESHOLD``
      consecutive frames, the detector tries a wider fallback crop first,
      then auto-recalibrates using the full screenshot.
    - If a detected turn *decreases* compared to the last confirmed value,
      the reading is treated as a misread and discarded.  After
      ``_ANOMALY_RETRY_LIMIT`` consecutive anomalies, a recalibration is
      triggered.
    - If a detected turn jumps by more than ``_MAX_TURN_JUMP``, the reading
      is held in a "pending" slot and only accepted after a second
      consecutive reading confirms the new value.

The main thread reads the result via the ``latest_turn`` property or
``check_new_turn()`` — both are effectively instant (no VLM call).
"""

from __future__ import annotations

import json
import logging
import threading

from computer_use_test.utils.llm_provider.parser import strip_markdown

logger = logging.getLogger(__name__)

_CALIBRATION_PROMPT = """\
이 문명6 스크린샷에서 턴 숫자가 표시된 UI 박스의 위치를 찾아라.
이미지는 게임 윈도우만 크롭된 상태이다 (데스크탑 요소 없음).
턴 숫자는 게임 화면 상단 바의 중앙~우측에 작은 숫자로 표시된다.
JSON만 응답: {"x1": 좌측비율, "y1": 상단비율, "x2": 우측비율, "y2": 하단비율}
비율은 0.0~1.0 (이미지 전체 대비). 턴 숫자 박스를 찾을 수 없으면 null."""

_TURN_DETECTION_PROMPT = '이 이미지에서 턴 숫자를 읽어라. JSON만 응답: {"turn_number": 숫자_또는_null}'

# Primary default: top-center-right area (Civ6 turn counter typical location)
_DEFAULT_CROP = {"x1": 0.42, "y1": 0.0, "x2": 0.58, "y2": 0.05}

# Wider fallback: entire top bar (used when narrow crop fails)
_WIDE_FALLBACK_CROP = {"x1": 0.30, "y1": 0.0, "x2": 1.0, "y2": 0.06}

# After this many consecutive null detections, try wider crop then recalibrate.
_RECALIBRATION_THRESHOLD = 3

# After this many consecutive anomalous readings, trigger auto-recalibration.
_ANOMALY_RETRY_LIMIT = 3

# If turn jumps by more than this in one frame, require confirmation.
_MAX_TURN_JUMP = 5


class TurnDetector:
    """
    Background worker that reads the in-game turn number from screenshots.

    Includes:
    - Lazy calibration on first submit (game screen guaranteed to be visible).
    - Wider fallback crop when narrow crop fails.
    - Auto-recalibration when crop yields null repeatedly.
    - Anomaly detection: ignores decreasing turns and requires confirmation
      for large jumps.

    Usage::

        detector = TurnDetector(vlm_provider)
        detector.start()

        # In the main loop (calibration happens automatically on first call):
        detector.submit(pil_image)

        # Instant reads:
        turn = detector.latest_turn       # int | None
        is_new = detector.check_new_turn()  # bool

        # On shutdown:
        detector.stop()
    """

    def __init__(self, vlm_provider) -> None:
        from computer_use_test.utils.llm_provider.base import BaseVLMProvider

        self._vlm: BaseVLMProvider = vlm_provider
        self._crop_box: dict[str, float] = dict(_DEFAULT_CROP)
        self._calibrated: bool = False

        # Turn tracking (confirmed values only)
        self._latest_turn: int | None = None
        self._previous_turn: int | None = None

        # --- Feedback / self-correction state ---
        self._consecutive_nulls: int = 0
        self._consecutive_anomalies: int = 0
        self._pending_turn: int | None = None
        self._recalibration_count: int = 0
        self._using_wide_fallback: bool = False

        # Single-slot mailbox (same pattern as ContextUpdater)
        self._pending_image = None
        self._last_full_image = None
        self._cond = threading.Condition()
        self._stopped = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background detection thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name="turn-detector", daemon=True)
        self._thread.start()
        logger.info("TurnDetector background worker started")

    def stop(self) -> None:
        """Signal the worker to stop and wait for it to finish."""
        with self._cond:
            self._stopped = True
            self._cond.notify()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("TurnDetector background worker stopped")

    def calibrate(self, pil_image) -> bool:
        """Locate the turn-number bounding box via a VLM call.

        Can be called explicitly at startup, or is called automatically
        on the first ``submit()``.

        Returns:
            True if calibration succeeded, False if using fallback.
        """
        try:
            content_parts = [
                self._vlm._build_pil_image_content(pil_image),
                self._vlm._build_text_content(_CALIBRATION_PROMPT),
            ]
            response = self._vlm._send_to_api(content_parts, temperature=0.0, max_tokens=128)
            content = strip_markdown(response.content)
            data = json.loads(content)

            if data is None:
                logger.warning("TurnDetector calibration: VLM returned null, using default crop")
                return False

            required_keys = {"x1", "y1", "x2", "y2"}
            if not required_keys.issubset(data.keys()):
                logger.warning(f"TurnDetector calibration: missing keys in {data}, using default crop")
                return False

            for key in required_keys:
                val = float(data[key])
                if not (0.0 <= val <= 1.0):
                    logger.warning(f"TurnDetector calibration: {key}={val} out of range, using default crop")
                    return False
                data[key] = val

            self._crop_box = {k: data[k] for k in required_keys}
            self._calibrated = True
            self._using_wide_fallback = False
            logger.info(f"Turn box calibrated: {self._crop_box}")
            return True

        except Exception as e:
            logger.warning(f"TurnDetector calibration failed: {e}, using default crop")
            return False

    def submit(self, pil_image) -> None:
        """Submit a screenshot for background turn detection (non-blocking)."""
        with self._cond:
            self._pending_image = pil_image
            self._cond.notify()

    @property
    def latest_turn(self) -> int | None:
        """Return the most recently confirmed turn number (instant read)."""
        return self._latest_turn

    def check_new_turn(self) -> bool:
        """Return True if the turn number has increased since the last check."""
        current = self._latest_turn
        previous = self._previous_turn

        if current is not None and previous is not None and current > previous:
            self._previous_turn = current
            return True

        if current is not None and previous is None:
            self._previous_turn = current

        return False

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def recalibration_count(self) -> int:
        """How many times auto-recalibration has been triggered."""
        return self._recalibration_count

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Background loop: wait for an image, crop, detect turn number."""
        while True:
            with self._cond:
                while self._pending_image is None and not self._stopped:
                    self._cond.wait()
                if self._stopped:
                    return
                image = self._pending_image
                self._pending_image = None

            self._last_full_image = image

            try:
                # Lazy calibration: on first real game screenshot
                if not self._calibrated:
                    logger.info("TurnDetector: lazy calibration on first submit")
                    self.calibrate(image)
                    # Even if calibrate fails, _calibrated stays False so
                    # recalibration can retry later. Mark as attempted.
                    self._calibrated = True

                self._detect_and_update(image)
            except Exception as e:
                logger.warning(f"TurnDetector analysis failed: {e}")

    def _detect_and_update(self, pil_image) -> None:
        """Crop the turn-number region, read digit via VLM, validate, and update."""
        import time as _time

        t0 = _time.monotonic()
        raw_turn = self._read_turn_from_crop(pil_image)
        elapsed = _time.monotonic() - t0

        # --- Null handling ---
        if raw_turn is None:
            self._consecutive_nulls += 1
            self._pending_turn = None
            logger.debug(f"TurnDetector: null detection ({self._consecutive_nulls}/{_RECALIBRATION_THRESHOLD})")

            if self._consecutive_nulls >= _RECALIBRATION_THRESHOLD:
                if not self._using_wide_fallback:
                    # Step 1: Try wider crop before full recalibration
                    self._crop_box = dict(_WIDE_FALLBACK_CROP)
                    self._using_wide_fallback = True
                    self._consecutive_nulls = 0
                    logger.warning("TurnDetector: narrow crop failed, switching to wide fallback crop")
                else:
                    # Step 2: Wide crop also failed → full recalibration
                    self._trigger_recalibration("wide fallback crop also failed")
            return

        # Got a number — reset null counter
        self._consecutive_nulls = 0

        # --- Validation against last confirmed turn ---
        confirmed = self._latest_turn

        if confirmed is not None:
            # Case 1: Turn decreased — misread, discard
            if raw_turn < confirmed:
                self._consecutive_anomalies += 1
                self._pending_turn = None
                logger.warning(
                    f"TurnDetector: turn DECREASED {confirmed} → {raw_turn} "
                    f"(anomaly {self._consecutive_anomalies}/{_ANOMALY_RETRY_LIMIT}), discarding"
                )
                if self._consecutive_anomalies >= _ANOMALY_RETRY_LIMIT:
                    self._trigger_recalibration("consecutive anomalous readings")
                return

            # Case 2: Large jump — hold for confirmation
            jump = raw_turn - confirmed
            if jump > _MAX_TURN_JUMP:
                if self._pending_turn == raw_turn:
                    logger.info(f"TurnDetector: large jump {confirmed} → {raw_turn} CONFIRMED (2nd reading)")
                    self._accept_turn(raw_turn, elapsed)
                else:
                    self._pending_turn = raw_turn
                    logger.warning(
                        f"TurnDetector: large jump {confirmed} → {raw_turn} (>{_MAX_TURN_JUMP}), awaiting confirmation"
                    )
                return

        # --- Normal case: accept ---
        self._accept_turn(raw_turn, elapsed)

    def _accept_turn(self, turn: int, duration: float | None = None) -> None:
        """Accept a validated turn number as the new confirmed value."""
        self._latest_turn = turn
        self._consecutive_anomalies = 0
        self._pending_turn = None
        logger.debug(f"TurnDetector: confirmed turn={turn}")

        # Update Rich dashboard with turn detection result
        from computer_use_test.utils.rich_logger import RichLogger

        RichLogger.get().update_turn_detector(turn, duration=duration)

    def _read_turn_from_crop(self, pil_image) -> int | None:
        """Crop the turn region and read the digit via VLM. Returns int or None."""
        w, h = pil_image.size
        x1 = int(w * self._crop_box["x1"])
        y1 = int(h * self._crop_box["y1"])
        x2 = int(w * self._crop_box["x2"])
        y2 = int(h * self._crop_box["y2"])

        cropped = pil_image.crop((x1, y1, x2, y2))

        content_parts = [
            self._vlm._build_pil_image_content(cropped),
            self._vlm._build_text_content(_TURN_DETECTION_PROMPT),
        ]

        response = self._vlm._send_to_api(content_parts, temperature=0.0, max_tokens=64)
        content = strip_markdown(response.content)

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"TurnDetector: invalid JSON: {e}")
            return None

        raw = data.get("turn_number")
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float) and raw == int(raw):
            return int(raw)
        return None

    def _trigger_recalibration(self, reason: str) -> None:
        """Run full-screenshot calibration inside the worker thread."""
        self._recalibration_count += 1
        logger.warning(f"TurnDetector: auto-recalibration #{self._recalibration_count} triggered ({reason})")

        self._consecutive_nulls = 0
        self._consecutive_anomalies = 0
        self._pending_turn = None
        self._using_wide_fallback = False

        full_image = self._last_full_image
        if full_image is None:
            logger.warning("TurnDetector: no full image available for recalibration")
            self._crop_box = dict(_DEFAULT_CROP)
            return

        success = self.calibrate(full_image)
        if not success:
            self._crop_box = dict(_DEFAULT_CROP)
            logger.warning("TurnDetector: recalibration failed, reverted to default crop")
