from __future__ import annotations

import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import PIL.Image
import PIL.ImageEnhance
import PIL.ImageFilter
import PIL.ImageOps
from PIL import features

from civStation.agent.models.schema import ClickAction, DragAction, KeyPressAction, WaitAction
from civStation.evaluation.evaluator.action_eval.bbox_eval.schema import CaseResult, DatasetCase
from civStation.evaluation.evaluator.action_eval.bbox_eval.scorer import select_best_gt_set

ALLOWED_UI_FILTER_MODES = {
    "none",
    "ui_contrast",
    "ui_quantized",
    "ui_bg_blur",
    "ui_bg_blur_contrast",
}
ALLOWED_COLOR_POLICIES = {"preserve", "grayscale", "adaptive_gray"}
ALLOWED_ENCODE_MODES = {"none", "jpeg_like", "webp_like", "avif_like_if_supported"}


@dataclass(frozen=True)
class PreprocessSpec:
    name: str
    ui_filter_mode: str
    color_policy: str
    encode_mode: str


@dataclass(frozen=True)
class PreparedImageResult:
    image: PIL.Image.Image
    payload_bytes: int
    payload_format: str
    encode_latency_ms: float
    preprocess_latency_ms: float


class UnsupportedEncodingModeError(RuntimeError):
    """Raised when a requested encoded transport format is not available."""


STAGED_PREPROCESS_SPECS = (
    PreprocessSpec("baseline", "none", "preserve", "none"),
    PreprocessSpec("ui_contrast", "ui_contrast", "preserve", "none"),
    PreprocessSpec("ui_quantized_webp", "ui_quantized", "preserve", "webp_like"),
    PreprocessSpec("ui_bg_blur_webp", "ui_bg_blur", "preserve", "webp_like"),
    PreprocessSpec("ui_bg_blur_gray_webp", "ui_bg_blur", "grayscale", "webp_like"),
    PreprocessSpec("ui_contrast_adaptive_jpeg", "ui_contrast", "adaptive_gray", "jpeg_like"),
    PreprocessSpec(
        "ui_bg_blur_contrast_avif",
        "ui_bg_blur_contrast",
        "adaptive_gray",
        "avif_like_if_supported",
    ),
)


def build_preprocess_specs(
    ui_filter_modes: list[str],
    color_policies: list[str],
    encode_modes: list[str],
    experiment_mode: str,
) -> list[PreprocessSpec]:
    allow_avif = features.check("avif")

    if experiment_mode == "staged":
        return [
            spec
            for spec in STAGED_PREPROCESS_SPECS
            if spec.ui_filter_mode in ui_filter_modes
            and spec.color_policy in color_policies
            and spec.encode_mode in encode_modes
            and (spec.encode_mode != "avif_like_if_supported" or allow_avif)
        ]

    specs: list[PreprocessSpec] = []
    for ui_filter_mode in ui_filter_modes:
        for color_policy in color_policies:
            for encode_mode in encode_modes:
                if encode_mode == "avif_like_if_supported" and not allow_avif:
                    continue
                specs.append(
                    PreprocessSpec(
                        name=f"{ui_filter_mode}__{color_policy}__{encode_mode}",
                        ui_filter_mode=ui_filter_mode,
                        color_policy=color_policy,
                        encode_mode=encode_mode,
                    )
                )
    return specs


def prepare_benchmark_image(image: PIL.Image.Image, spec: PreprocessSpec) -> PreparedImageResult:
    start = time.perf_counter()
    processed = apply_ui_filter(image, spec.ui_filter_mode)
    processed = apply_color_policy(processed, spec.color_policy)
    encoded_image, payload_bytes, payload_format, encode_latency_ms = simulate_transport_encoding(
        processed,
        spec.encode_mode,
    )
    preprocess_latency_ms = (time.perf_counter() - start) * 1000
    return PreparedImageResult(
        image=encoded_image,
        payload_bytes=payload_bytes,
        payload_format=payload_format,
        encode_latency_ms=encode_latency_ms,
        preprocess_latency_ms=preprocess_latency_ms,
    )


def apply_ui_filter(image: PIL.Image.Image, mode: str) -> PIL.Image.Image:
    source = image.convert("RGB")
    if mode == "none":
        return source.copy()
    if mode == "ui_contrast":
        return _enhance_ui_contrast(source)
    if mode == "ui_quantized":
        quantized = (
            PIL.ImageOps.autocontrast(source, cutoff=1)
            .quantize(colors=256, method=PIL.Image.Quantize.MEDIANCUT, dither=PIL.Image.Dither.NONE)
            .convert("RGB")
        )
        return quantized.filter(PIL.ImageFilter.UnsharpMask(radius=1.1, percent=135, threshold=2))
    if mode == "ui_bg_blur":
        return _blur_background_keep_edges(source)
    if mode == "ui_bg_blur_contrast":
        return _enhance_ui_contrast(_blur_background_keep_edges(source))
    raise ValueError(f"Unknown ui filter mode: {mode}")


def apply_color_policy(image: PIL.Image.Image, policy: str) -> PIL.Image.Image:
    source = image.convert("RGB")
    if policy == "preserve":
        return source.copy()
    if policy == "grayscale":
        return PIL.ImageOps.grayscale(source).convert("RGB")
    if policy == "adaptive_gray":
        gray = PIL.ImageOps.grayscale(source).convert("RGB")
        _h, saturation, _v = source.convert("HSV").split()
        keep_color_mask = saturation.filter(PIL.ImageFilter.GaussianBlur(radius=1.2)).point(
            lambda px: 255 if px >= 72 else 0
        )
        return PIL.Image.composite(source, gray, keep_color_mask)
    raise ValueError(f"Unknown color policy: {policy}")


def simulate_transport_encoding(
    image: PIL.Image.Image,
    encode_mode: str,
) -> tuple[PIL.Image.Image, int, str, float]:
    source = image.convert("RGB")
    start = time.perf_counter()

    if encode_mode == "none":
        encoded = _encode_image(source, "PNG", optimize=True)
        encode_latency_ms = (time.perf_counter() - start) * 1000
        return source.copy(), len(encoded), "png", encode_latency_ms

    if encode_mode == "jpeg_like":
        encoded = _encode_image(source, "JPEG", quality=48, optimize=True, subsampling=0)
        decoded = _decode_rgb_image(encoded)
        encode_latency_ms = (time.perf_counter() - start) * 1000
        return decoded, len(encoded), "jpeg", encode_latency_ms

    if encode_mode == "webp_like":
        encoded = _encode_image(source, "WEBP", quality=50, method=6)
        decoded = _decode_rgb_image(encoded)
        encode_latency_ms = (time.perf_counter() - start) * 1000
        return decoded, len(encoded), "webp", encode_latency_ms

    if encode_mode == "avif_like_if_supported":
        if not features.check("avif"):
            raise UnsupportedEncodingModeError("AVIF encoding is not available in the current Pillow build")
        encoded = _encode_image(source, "AVIF", quality=45, speed=6)
        decoded = _decode_rgb_image(encoded)
        encode_latency_ms = (time.perf_counter() - start) * 1000
        return decoded, len(encoded), "avif", encode_latency_ms

    raise ValueError(f"Unknown encode mode: {encode_mode}")


def convert_actions_for_bbox_eval(actions: list[dict]) -> list[ClickAction | DragAction | KeyPressAction | WaitAction]:
    converted: list[ClickAction | DragAction | KeyPressAction | WaitAction] = []
    for raw_action in actions:
        action_type = str(raw_action.get("action", "")).strip().lower()
        description = str(raw_action.get("reasoning", "")).strip() or None

        if action_type == "click":
            converted.append(
                ClickAction(
                    x=int(raw_action.get("x", 0)),
                    y=int(raw_action.get("y", 0)),
                    button=str(raw_action.get("button", "left") or "left"),
                    description=description,
                )
            )
            continue

        if action_type == "drag":
            converted.append(
                DragAction(
                    start_x=int(raw_action.get("x", 0)),
                    start_y=int(raw_action.get("y", 0)),
                    end_x=int(raw_action.get("end_x", 0)),
                    end_y=int(raw_action.get("end_y", 0)),
                    button=str(raw_action.get("button", "left") or "left"),
                    description=description,
                )
            )
            continue

        if action_type == "press":
            keys = _normalize_keys(str(raw_action.get("key", "")))
            if not keys:
                raise ValueError("press action did not include any usable key value")
            converted.append(KeyPressAction(keys=keys, description=description))
            continue

        if action_type == "wait":
            duration = float(raw_action.get("duration", 1.0))
            converted.append(WaitAction(duration=duration, description=description))
            continue

        raise ValueError(f"Unsupported action type for bbox evaluation: {action_type or 'empty'}")

    return converted


def score_actions_against_case(
    case: DatasetCase,
    actions: list[dict],
    ignore_wait: bool = False,
) -> CaseResult:
    try:
        converted_actions = convert_actions_for_bbox_eval(actions)
    except ValueError as exc:
        return CaseResult(case_id=case.case_id, error=str(exc))

    gt_action_sets = [action_set.actions for action_set in case.action_sets]
    best_sequence, gt_index = select_best_gt_set(gt_action_sets, converted_actions, ignore_wait=ignore_wait)
    return CaseResult(
        case_id=case.case_id,
        best_sequence=best_sequence,
        agent_actions_count=len(converted_actions),
        gt_set_index=gt_index,
    )


def resolve_dataset_screenshot_path(dataset_path: str | Path, screenshot_path: str) -> Path:
    dataset_path = Path(dataset_path)
    screenshot = Path(screenshot_path)
    if screenshot.is_absolute():
        return screenshot
    return (dataset_path.parent / screenshot).resolve()


def _enhance_ui_contrast(image: PIL.Image.Image) -> PIL.Image.Image:
    enhanced = PIL.ImageOps.autocontrast(image, cutoff=1)
    enhanced = PIL.ImageEnhance.Contrast(enhanced).enhance(1.18)
    return enhanced.filter(PIL.ImageFilter.UnsharpMask(radius=1.3, percent=165, threshold=2))


def _blur_background_keep_edges(image: PIL.Image.Image) -> PIL.Image.Image:
    edges = PIL.ImageOps.autocontrast(
        PIL.ImageOps.grayscale(image).filter(PIL.ImageFilter.FIND_EDGES),
        cutoff=2,
    )
    mask = edges.filter(PIL.ImageFilter.GaussianBlur(radius=1.8)).point(
        lambda px: max(0, min(255, int((px / 255.0) ** 0.7 * 255)))
    )
    blurred = image.filter(PIL.ImageFilter.GaussianBlur(radius=4.2))
    return PIL.Image.composite(image, blurred, mask)


def _encode_image(image: PIL.Image.Image, fmt: str, **save_kwargs: object) -> bytes:
    buf = BytesIO()
    image.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


def _decode_rgb_image(encoded: bytes) -> PIL.Image.Image:
    buf = BytesIO(encoded)
    image = PIL.Image.open(buf)
    image.load()
    return image.convert("RGB")


def _normalize_keys(raw_key: str) -> list[str]:
    cleaned = raw_key.strip().lower()
    if not cleaned or cleaned == "none":
        return []
    if "+" in cleaned:
        return [part.strip() for part in cleaned.split("+") if part.strip()]
    if "," in cleaned:
        return [part.strip() for part in cleaned.split(",") if part.strip()]
    return [cleaned]
