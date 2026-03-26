"""Shared runtime hooks for multi-step primitive recovery and verification."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from civStation.agent.modules.memory.short_term_memory import ShortTermMemory
from civStation.utils.llm_provider.base import BaseVLMProvider
from civStation.utils.llm_provider.parser import AgentAction, strip_markdown

logger = logging.getLogger(__name__)


@dataclass
class NoProgressResolution:
    """Shared recovery decision after a no-progress or semantic-failure step."""

    handled: bool = False
    reroute: bool = False
    error_message: str = ""


@dataclass
class SemanticVerifyResult:
    """Outcome of a semantic post-action verification hook."""

    handled: bool = False
    passed: bool = True
    reason: str = ""
    details: dict[str, object] = field(default_factory=dict)


class RetryFallbackHook:
    """Common retry/fallback policy shared by multi-step processes."""

    def __init__(self, retry_limit: int = 1, fallback_stage: str = "generic_fallback"):
        self.retry_limit = retry_limit
        self.fallback_stage = fallback_stage

    def handle_failure(
        self,
        memory: ShortTermMemory,
        *,
        stage_name: str,
        stage_key: str,
        on_first_retry=None,
        reroute_message: str | None = None,
    ) -> NoProgressResolution:
        failures = memory.increment_stage_failure(stage_key)

        if failures <= self.retry_limit:
            if on_first_retry is not None:
                try:
                    on_first_retry()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Stage-local retry hook failed for %s: %s", stage_key, exc)
            logger.info("Stage '%s' failed once -> retry", stage_key)
            return NoProgressResolution(handled=True)

        if not memory.has_stage_fallback_used(stage_key):
            memory.mark_stage_fallback_used(stage_key)
            memory.set_fallback_return_stage(stage_name, stage_key)
            memory.begin_stage(self.fallback_stage)
            logger.info("Stage '%s' exceeded retry budget -> generic fallback", stage_key)
            return NoProgressResolution(handled=True)

        return NoProgressResolution(
            handled=False,
            reroute=True,
            error_message=reroute_message or f"Stage '{stage_name}' failed after retry+fallback",
        )

    def reset(self, memory: ShortTermMemory, stage_key: str) -> None:
        memory.reset_stage_failure(stage_key)

    def on_fallback_success(self, memory: ShortTermMemory) -> None:
        return_stage, return_key = memory.consume_fallback_return_stage()
        if return_key:
            memory.reset_stage_failure(return_key)
        if return_stage:
            memory.begin_stage(return_stage)


class BaseSemanticVerifyHook:
    """Opt-in semantic verification hook."""

    def verify(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        action: AgentAction,
        *,
        img_config=None,
    ) -> SemanticVerifyResult:
        return SemanticVerifyResult(handled=False)


class NoopSemanticVerifyHook(BaseSemanticVerifyHook):
    """Default hook: no semantic verification."""


def _analyze_structured_json(
    provider: BaseVLMProvider,
    pil_image,
    prompt: str,
    *,
    img_config=None,
    max_tokens: int = 512,
) -> dict:
    prepared = provider._prepare_pil_image(pil_image, img_config=img_config)
    jpeg_quality = getattr(img_config, "jpeg_quality", 0) if img_config else 0
    build_kwargs = {"jpeg_quality": jpeg_quality} if jpeg_quality > 0 else {}
    content_parts = [
        provider._build_pil_image_content(prepared, **build_kwargs),
        provider._build_text_content(prompt),
    ]

    response = provider._send_to_api(content_parts, temperature=0.1, max_tokens=max_tokens, use_thinking=False)
    content = strip_markdown(response.content)
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError(f"expected dict JSON, got {type(data).__name__}")
    return data


class PolicySemanticVerifyHook(BaseSemanticVerifyHook):
    """Legacy hook kept for compatibility; policy drag verification is unused."""

    def verify(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        action: AgentAction,
        *,
        img_config=None,
    ) -> SemanticVerifyResult:
        return SemanticVerifyResult(handled=False)
