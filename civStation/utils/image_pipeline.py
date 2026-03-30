"""
Parameterized Image Preprocessing Pipeline.

Provides a single ``ImagePipelineConfig`` dataclass that controls every
preprocessing step (UI filter, colour policy, resize, transport encoding).
``process_image()`` applies the pipeline and returns a ``PipelineResult``.

Building blocks are reused from:
- ``ui_benchmarking.py`` — ``apply_ui_filter``, ``apply_color_policy``,
  ``simulate_transport_encoding``
- ``screen.py`` — ``resize_for_vlm``

Presets (``ROUTER_DEFAULT``, ``PLANNER_DEFAULT``, …) give sensible
per-call-site defaults.  ``config_from_args()`` lets CLI / config.yaml
override any parameter without touching source code.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config & Result
# ---------------------------------------------------------------------------


@dataclass
class ImagePipelineConfig:
    """All knobs for a single image-preprocessing pass."""

    max_long_edge: int = 1024  # 0 = no resize
    ui_filter_mode: str = "none"  # none|ui_contrast|ui_quantized|ui_bg_blur|ui_bg_blur_contrast
    color_policy: str = "preserve"  # preserve|grayscale|adaptive_gray
    encode_mode: str = "none"  # none|jpeg_like|webp_like|avif_like_if_supported
    jpeg_quality: int = 0  # 0 = provider default (VLM_JPEG_QUALITY)


@dataclass
class PipelineResult:
    """Output of ``process_image()``."""

    image: Image.Image
    elapsed_ms: float
    payload_bytes: int = 0
    payload_format: str = "pil"


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

ROUTER_DEFAULT = ImagePipelineConfig(
    max_long_edge=768,
    ui_filter_mode="ui_bg_blur_contrast",
    color_policy="grayscale",
    encode_mode="none",
    jpeg_quality=60,
)

PLANNER_DEFAULT = ImagePipelineConfig(
    max_long_edge=1024,
    ui_filter_mode="ui_contrast",
    color_policy="preserve",
    encode_mode="none",
    jpeg_quality=75,
)

CONTEXT_DEFAULT = ImagePipelineConfig(
    max_long_edge=1280,
    ui_filter_mode="none",
    color_policy="preserve",
    encode_mode="none",
    jpeg_quality=0,
)

TURN_DETECTOR_DEFAULT = ImagePipelineConfig(
    max_long_edge=0,
    ui_filter_mode="none",
    color_policy="preserve",
    encode_mode="none",
    jpeg_quality=0,
)

PLANNER_HIGH_QUALITY = ImagePipelineConfig(
    max_long_edge=1280,
    ui_filter_mode="ui_contrast",
    color_policy="preserve",
    encode_mode="none",
    jpeg_quality=85,
)

OBSERVATION_FAST = ImagePipelineConfig(
    max_long_edge=768,
    ui_filter_mode="ui_contrast",
    color_policy="preserve",
    encode_mode="none",
    jpeg_quality=70,
)

POLICY_TAB_CHECK_FAST = ImagePipelineConfig(
    max_long_edge=640,
    ui_filter_mode="ui_contrast",
    color_policy="preserve",
    encode_mode="none",
    jpeg_quality=65,
)

CITY_PRODUCTION_FOLLOWUP_FAST = ImagePipelineConfig(
    max_long_edge=640,
    ui_filter_mode="ui_contrast",
    color_policy="preserve",
    encode_mode="none",
    jpeg_quality=60,
)

CITY_PRODUCTION_PLACEMENT_FAST = ImagePipelineConfig(
    max_long_edge=768,
    ui_filter_mode="ui_contrast",
    color_policy="preserve",
    encode_mode="none",
    jpeg_quality=65,
)

PRESETS: dict[str, ImagePipelineConfig] = {
    "router_default": ROUTER_DEFAULT,
    "planner_default": PLANNER_DEFAULT,
    "context_default": CONTEXT_DEFAULT,
    "turn_detector_default": TURN_DETECTOR_DEFAULT,
    "planner_high_quality": PLANNER_HIGH_QUALITY,
    "observation_fast": OBSERVATION_FAST,
    "policy_tab_check_fast": POLICY_TAB_CHECK_FAST,
    "city_production_followup_fast": CITY_PRODUCTION_FOLLOWUP_FAST,
    "city_production_placement_fast": CITY_PRODUCTION_PLACEMENT_FAST,
}

# Map CLI site prefix → preset name
_SITE_TO_PRESET: dict[str, str] = {
    "router": "router_default",
    "planner": "planner_default",
    "context": "context_default",
    "turn_detector": "turn_detector_default",
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def process_image(
    pil_image: Image.Image,
    config: ImagePipelineConfig | None = None,
) -> PipelineResult:
    """Apply the full preprocessing pipeline.

    Order: UI filter → colour policy → resize → transport encoding.
    """
    if config is None:
        config = ImagePipelineConfig()

    t0 = time.perf_counter()
    img = pil_image

    # 1. UI filter
    if config.ui_filter_mode != "none":
        from civStation.utils.ui_benchmarking import apply_ui_filter

        img = apply_ui_filter(img, config.ui_filter_mode)

    # 2. Colour policy
    if config.color_policy != "preserve":
        from civStation.utils.ui_benchmarking import apply_color_policy

        img = apply_color_policy(img, config.color_policy)

    # 3. Resize
    if config.max_long_edge > 0:
        from civStation.utils.screen import resize_for_vlm

        img = resize_for_vlm(img, max_long_edge=config.max_long_edge)

    # 4. Transport encoding
    payload_bytes = 0
    payload_format = "pil"
    if config.encode_mode != "none":
        from civStation.utils.ui_benchmarking import simulate_transport_encoding

        img, payload_bytes, payload_format, _enc_ms = simulate_transport_encoding(img, config.encode_mode)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return PipelineResult(
        image=img,
        elapsed_ms=elapsed_ms,
        payload_bytes=payload_bytes,
        payload_format=payload_format,
    )


# ---------------------------------------------------------------------------
# CLI / config.yaml helpers
# ---------------------------------------------------------------------------


def config_from_args(args: object, site: str) -> ImagePipelineConfig:
    """Build an ``ImagePipelineConfig`` for *site* from parsed CLI args.

    Starts from the site's default preset, then applies any per-field
    CLI overrides (``--{site}-img-{field}``).
    """
    # Normalise site name: "turn-detector" → "turn_detector"
    site_key = site.replace("-", "_")
    attr_prefix = site_key + "_img_"

    # 1. Resolve preset
    preset_attr = attr_prefix + "preset"
    preset_name = getattr(args, preset_attr, None) or _SITE_TO_PRESET.get(site_key, "planner_default")
    base = PRESETS.get(preset_name)
    if base is None:
        logger.warning(f"Unknown preset '{preset_name}', falling back to planner_default")
        base = PLANNER_DEFAULT
    cfg = ImagePipelineConfig(
        max_long_edge=base.max_long_edge,
        ui_filter_mode=base.ui_filter_mode,
        color_policy=base.color_policy,
        encode_mode=base.encode_mode,
        jpeg_quality=base.jpeg_quality,
    )

    # 2. Apply individual overrides
    override_map = {
        "max_long_edge": int,
        "ui_filter": str,
        "color": str,
        "encode": str,
        "jpeg_quality": int,
    }
    field_map = {
        "ui_filter": "ui_filter_mode",
        "color": "color_policy",
        "encode": "encode_mode",
    }

    for cli_suffix, cast in override_map.items():
        attr_name = attr_prefix + cli_suffix
        raw = getattr(args, attr_name, None)
        if raw is not None:
            field_name = field_map.get(cli_suffix, cli_suffix)
            setattr(cfg, field_name, cast(raw))

    return cfg


def from_preprocess_spec(spec) -> ImagePipelineConfig:
    """Convert a ``PreprocessSpec`` (from ``ui_benchmarking``) to ``ImagePipelineConfig``."""
    return ImagePipelineConfig(
        max_long_edge=0,
        ui_filter_mode=spec.ui_filter_mode,
        color_policy=spec.color_policy,
        encode_mode=spec.encode_mode,
        jpeg_quality=0,
    )
