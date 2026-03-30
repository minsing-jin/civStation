"""
Context Updater — Background worker that analyzes screenshots independently.

Runs in a daemon thread. The main pipeline feeds it screenshots via
``submit()``, and it uses a VLM to extract game-state information and
write it back to the ContextManager. Because the worker owns a
dedicated thread, the main routing/planning pipeline is never blocked.

Read-path safety:
    The main thread reads context through ``ContextManager.get_context_for_primitive()``
    which only accesses Python-atomic attribute references on dataclass
    fields, so no explicit locking is required for reads.

Write-path safety:
    All writes go through ``ContextManager.update_game_observation()``
    which updates ``HighLevelContext`` fields (notes, threats,
    opportunities).  The updater is the *only* writer for these
    observation fields, so there is no write–write race.
"""

# TODO: Build an MCP server to extract native Civ6 game data and combine
#       it with VLM vision analysis for a hybrid context pipeline.
#       (MCP를 활용하여 게임 데이터 직접 추출 + VLM 비전 하이브리드)
# TODO: Migrate from unstructured VLM-extracted context to structured
#       context with hybrid approach.
#       (Unstructured → Structured context 하이브리드 접근 도입)
# TODO: Context length management — cap situation_summary length and
#       limit cumulative observations so they don't exceed max tokens.
#       (situation_summary 길이 제한 및 누적 observations 토큰 관리)

from __future__ import annotations

import json
import logging
import threading
from typing import TYPE_CHECKING

from civStation.utils.llm_provider.parser import strip_markdown

if TYPE_CHECKING:
    from civStation.agent.modules.context.context_manager import ContextManager
    from civStation.utils.llm_provider.base import BaseVLMProvider

logger = logging.getLogger(__name__)

_CONTEXT_EXTRACTION_PROMPT = """\
너는 문명6 게임 화면의 전략 상황 분석기야.
스크린샷을 보고 현재 게임 상황을 전략적 관점에서 요약해.

읽어야 할 정보:
1. 화면 상단 바: 과학/문화/골드/신앙 산출량
2. 현재 턴 숫자 (화면 오른쪽 위)
3. 현재 시대
4. 현재 연구 중인 기술 / 사회제도 (있으면)
5. 전쟁 상태, 적 유닛 위치, 도시 방어 상황 등 위협 요소
6. 건설 가능한 불가사의, 정착 가능한 좋은 위치, 약한 도시국가 등 기회 요소

응답 형식 (JSON만 출력):
{
    "current_turn": 숫자_또는_null,
    "game_era": "시대명_또는_null",
    "situation_summary": "2-3줄 전략 요약 (자원, 연구, 군사 상황)",
    "threats": ["위협요소1", "위협요소2"],
    "opportunities": ["기회요소1", "기회요소2"]
}

situation_summary 예시: "턴 85, 중세시대. 과학 +42/턴. 대학 연구 중(3턴). 아즈텍 전사 접근."
threats 예시: ["아즈텍 전사 2기 북쪽 국경 접근", "골드 수입 낮음 (-3/턴)"]
opportunities 예시: ["남쪽에 정착 가능한 좋은 위치", "콜로세움 건설 가능"]
"""


class ContextUpdater:
    """
    Background worker that extracts game state from screenshots via VLM
    and writes the results into ContextManager.

    Usage::

        updater = ContextUpdater(ctx, vlm_provider)
        updater.start()

        # In the main loop, after capturing a screenshot:
        updater.submit(pil_image)

        # The latest context is always available synchronously:
        ctx.get_context_for_primitive(primitive_name)

        # On shutdown:
        updater.stop()
    """

    def __init__(
        self,
        context_manager: ContextManager,
        vlm_provider: BaseVLMProvider,
        img_config=None,
    ) -> None:
        self._ctx = context_manager
        self._vlm = vlm_provider
        self._img_config = img_config

        # Single-slot "mailbox": holds the most recent screenshot.
        # If a new screenshot arrives before the previous one is processed,
        # the old one is simply replaced — we always want the freshest frame.
        self._pending_image = None
        self._cond = threading.Condition()  # guards _pending_image + _stopped
        self._stopped = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background analysis thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name="context-updater", daemon=True)
        self._thread.start()
        logger.info("ContextUpdater background worker started")

    def stop(self) -> None:
        """Signal the worker to stop and wait for it to finish."""
        with self._cond:
            self._stopped = True
            self._cond.notify()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("ContextUpdater background worker stopped")

    def submit(self, pil_image) -> None:
        """
        Submit a screenshot for background analysis.

        Non-blocking. If a previous screenshot has not been processed yet,
        it is replaced with the newer one (latest-wins policy).
        """
        with self._cond:
            self._pending_image = pil_image
            self._cond.notify()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Background loop: wait for an image, analyze it, update context."""
        while True:
            # Wait until there is work or we are told to stop.
            with self._cond:
                while self._pending_image is None and not self._stopped:
                    self._cond.wait()
                if self._stopped:
                    return
                image = self._pending_image
                self._pending_image = None  # consumed

            # Analyze outside the lock — this is the slow VLM call.
            try:
                self._analyze_and_update(image)
            except Exception as e:
                logger.warning(f"ContextUpdater analysis failed: {e}")

    def _analyze_and_update(self, pil_image) -> None:
        """Call VLM to extract strategic observations and write to high-level context."""
        import time as _time

        from civStation.utils.image_pipeline import CONTEXT_DEFAULT, process_image

        t0 = _time.monotonic()

        cfg = self._img_config or CONTEXT_DEFAULT
        result = process_image(pil_image, cfg)
        prepared = result.image
        jpeg_quality = cfg.jpeg_quality if cfg.jpeg_quality > 0 else None
        build_kwargs = {"jpeg_quality": jpeg_quality} if jpeg_quality else {}
        content_parts = [
            self._vlm._build_pil_image_content(prepared, **build_kwargs),
            self._vlm._build_text_content(_CONTEXT_EXTRACTION_PROMPT),
        ]

        response = self._vlm._send_to_api(content_parts, temperature=0.1)
        content = strip_markdown(response.content)

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"ContextUpdater: invalid JSON from VLM: {e}")
            return

        elapsed = _time.monotonic() - t0

        # --- Minimal GlobalContext update (turn tracking only) ---
        turn_val = data.get("current_turn")
        era_val = data.get("game_era")
        global_updates: dict = {}
        if isinstance(turn_val, int | float) and turn_val == int(turn_val):
            global_updates["current_turn"] = int(turn_val)
        if isinstance(era_val, str) and era_val:
            global_updates["game_era"] = era_val
        if global_updates:
            self._ctx.update_global_context(**global_updates)

        # --- Strategic observations → HighLevelContext ---
        situation = data.get("situation_summary", "")
        if not isinstance(situation, str) or len(situation) < 5:
            logger.debug("ContextUpdater: no meaningful situation summary, skipping")
            return

        threats = data.get("threats", [])
        if not isinstance(threats, list):
            threats = []
        threats = [t for t in threats if isinstance(t, str) and t]

        opportunities = data.get("opportunities", [])
        if not isinstance(opportunities, list):
            opportunities = []
        opportunities = [o for o in opportunities if isinstance(o, str) and o]

        self._ctx.update_game_observation(
            situation_summary=situation,
            threats=threats,
            opportunities=opportunities,
        )
        logger.debug(f"ContextUpdater wrote observation (threats={len(threats)}, opportunities={len(opportunities)})")

        # Update Rich dashboard with context results
        from civStation.utils.rich_logger import RichLogger

        RichLogger.get().update_context(
            turn=global_updates.get("current_turn"),
            era=global_updates.get("game_era"),
            threats=threats,
            opportunities=opportunities,
            duration=elapsed,
        )
