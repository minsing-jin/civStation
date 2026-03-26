import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path

import PIL.Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from civStation.evaluation.evaluator.action_eval.bbox_eval.dataset_loader import load_dataset
from civStation.evaluation.evaluator.action_eval.bbox_eval.scorer import aggregate_results
from civStation.utils.prompts.primitive_prompt import (
    MULTI_ACTION_SEQUENCE_JSON_FORMAT_INSTRUCTION,
    POLICY_PROMPT,
)
from civStation.utils.ui_benchmarking import (
    ALLOWED_COLOR_POLICIES,
    ALLOWED_ENCODE_MODES,
    ALLOWED_UI_FILTER_MODES,
    PreprocessSpec,
    UnsupportedEncodingModeError,
    build_preprocess_specs,
    prepare_benchmark_image,
    resolve_dataset_screenshot_path,
    score_actions_against_case,
)

try:
    import pyautogui
except Exception:  # pragma: no cover - depends on GUI availability
    pyautogui = None

load_dotenv()

console = Console()

ALLOWED_PROMPT_VARIANTS = {"long_prompt", "concise_prompt"}
ALLOWED_SIZE_MODES = {"raw", "compressed", "downscale_restore", "compressed_downscale_restore"}
ALLOWED_BACKGROUND_MODES = {"none", "color"}


def _require_pyautogui():
    if pyautogui is None:
        raise RuntimeError("pyautogui is unavailable in this environment")
    return pyautogui


@dataclass(frozen=True)
class ImageVariant:
    name: str
    size_mode: str
    background_mode: str
    image: PIL.Image.Image
    # (left, top, width, height): 원본 UI가 variant 이미지 내에서 차지하는 영역
    content_box: tuple[int, int, int, int]


@dataclass(frozen=True)
class PreparedImageVariant:
    name: str
    size_mode: str
    background_mode: str
    ui_filter_mode: str
    color_policy: str
    encode_mode: str
    image: PIL.Image.Image
    content_box: tuple[int, int, int, int]
    preprocess_latency_ms: float
    encode_latency_ms: float
    payload_bytes: int
    payload_format: str


def build_action_schema(normalizing_range: int) -> dict:
    return {
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["click", "press", "drag", "type"]},
                "x": {"type": "integer", "minimum": 0, "maximum": normalizing_range},
                "y": {"type": "integer", "minimum": 0, "maximum": normalizing_range},
                "end_x": {"type": "integer", "minimum": 0, "maximum": normalizing_range},
                "end_y": {"type": "integer", "minimum": 0, "maximum": normalizing_range},
                "button": {"type": "string", "enum": ["left", "right"]},
                "key": {"type": "string"},
                "text": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["action", "x", "y", "end_x", "end_y", "button", "key", "text", "reasoning"],
            "additionalProperties": False,
        },
    }


def build_long_prompt(normalizing_range: int) -> str:
    return f"""
{MULTI_ACTION_SEQUENCE_JSON_FORMAT_INSTRUCTION.format(normalizing_range=normalizing_range)}

{POLICY_PROMPT}

추가 지시:
- 반드시 실행 가능한 action sequence만 반환하라.
- 좌표는 모두 0~{normalizing_range} 정규화 좌표를 사용하라.
- 실행 불가한 추측은 하지 말고, 보이는 UI 근거를 reasoning에 짧게 남겨라.
""".strip()


def build_concise_prompt(normalizing_range: int) -> str:
    return f"""
문명6 정책 화면에 대해 다음 action space로 실행 계획(JSON 배열)을 반환하라.
- action: click | press | drag | type
- 좌표는 0~{normalizing_range} 정규화
- 정책 화면 우선순위:
  1) 확인 팝업이면 확인/취소 처리
  2) 정부 선택이면 전략에 맞는 정부 선택
  3) 정책 관리면 슬롯 채우기/교체/탭 전환
- 출력은 JSON 배열만 반환
""".strip()


def normalize_screenshot_for_primitive(img: PIL.Image.Image, mode: str) -> PIL.Image.Image:
    """
    Primitive 입력용 screenshot normalization.
    - raw: 원본 유지
    - compressed: 최대 1280으로 축소 + JPEG 재인코딩(품질 저하)
    """
    if mode == "raw":
        return img.copy()

    if mode != "compressed":
        raise ValueError(f"Unknown image mode: {mode}")

    comp = img.convert("RGB").copy()
    comp.thumbnail((1280, 1280), PIL.Image.Resampling.LANCZOS)

    # JPEG 재인코딩으로 화질/용량 축소 효과를 강제
    buf = BytesIO()
    comp.save(buf, format="JPEG", quality=45, optimize=True)
    buf.seek(0)
    return PIL.Image.open(buf).convert("RGB")


def resize_long_edge(img: PIL.Image.Image, max_long_edge: int) -> PIL.Image.Image:
    src = img.convert("RGB")
    w, h = src.size
    long_edge = max(w, h)
    if long_edge <= max_long_edge:
        return src.copy()

    scale = max_long_edge / long_edge
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return src.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)


def add_background_canvas(
    img: PIL.Image.Image,
    background_color: tuple[int, int, int],
    padding_ratio: float,
) -> tuple[PIL.Image.Image, tuple[int, int, int, int]]:
    src = img.convert("RGB")
    w, h = src.size
    pad_x = max(1, int(round(w * padding_ratio)))
    pad_y = max(1, int(round(h * padding_ratio)))
    canvas_w = w + pad_x * 2
    canvas_h = h + pad_y * 2

    canvas = PIL.Image.new("RGB", (canvas_w, canvas_h), background_color)
    canvas.paste(src, (pad_x, pad_y))
    return canvas, (pad_x, pad_y, w, h)


def parse_background_color(value: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--background-color 형식은 'R,G,B' 여야 합니다. 예: 32,32,32")

    rgb: list[int] = []
    for part in parts:
        channel = int(part)
        if not (0 <= channel <= 255):
            raise ValueError(f"RGB 채널은 0~255 이어야 합니다: {channel}")
        rgb.append(channel)
    return rgb[0], rgb[1], rgb[2]


def parse_csv_modes(raw: str, allowed: set[str], arg_name: str) -> list[str]:
    items = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{arg_name}는 최소 1개 이상 지정해야 합니다.")

    unknown = sorted(set(items) - allowed)
    if unknown:
        raise ValueError(f"{arg_name}에 알 수 없는 값이 있습니다: {', '.join(unknown)}")
    return items


def parse_csv_ints(raw: str, arg_name: str) -> list[int]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{arg_name}는 최소 1개 이상 지정해야 합니다.")

    values: list[int] = []
    for item in items:
        value = int(item)
        if value <= 0:
            raise ValueError(f"{arg_name} 값은 1 이상이어야 합니다: {value}")
        values.append(value)
    return values


def build_instruction_prompt(prompt_variant: str, normalizing_range: int, instruction: str) -> str:
    if prompt_variant == "long_prompt":
        return f"""
{MULTI_ACTION_SEQUENCE_JSON_FORMAT_INSTRUCTION.format(normalizing_range=normalizing_range)}

사용자 지시:
- {instruction}

추가 지시:
- 반드시 실행 가능한 action sequence만 반환하라.
- 좌표는 모두 0~{normalizing_range} 정규화 좌표를 사용하라.
- 실행 불가한 추측은 하지 말고, 보이는 UI 근거를 reasoning에 짧게 남겨라.
""".strip()
    if prompt_variant == "concise_prompt":
        return f"""
게임 UI 스크린샷에 대해 사용자 지시를 수행하는 실행 계획(JSON 배열)을 반환하라.
- instruction: {instruction}
- action: click | press | drag | type
- 좌표는 0~{normalizing_range} 정규화
- 출력은 JSON 배열만 반환
""".strip()
    raise ValueError(f"Unknown prompt variant: {prompt_variant}")


def build_prompt(prompt_variant: str, normalizing_range: int, instruction: str | None = None) -> str:
    if instruction is not None:
        return build_instruction_prompt(prompt_variant, normalizing_range, instruction)
    if prompt_variant == "long_prompt":
        return build_long_prompt(normalizing_range)
    if prompt_variant == "concise_prompt":
        return build_concise_prompt(normalizing_range)
    raise ValueError(f"Unknown prompt variant: {prompt_variant}")


def build_image_variant(
    base_image: PIL.Image.Image,
    size_mode: str,
    background_mode: str,
    downscale_long_edge: int,
    background_color: tuple[int, int, int],
    background_padding_ratio: float,
) -> ImageVariant:
    if size_mode == "raw":
        content = normalize_screenshot_for_primitive(base_image, "raw")
    elif size_mode == "compressed":
        content = normalize_screenshot_for_primitive(base_image, "compressed")
    elif size_mode == "downscale_restore":
        content = resize_long_edge(base_image, downscale_long_edge)
    elif size_mode == "compressed_downscale_restore":
        # downscale_restore와 동일하게 축소 후 JPEG 재인코딩까지 추가 적용
        content = normalize_screenshot_for_primitive(
            resize_long_edge(base_image, downscale_long_edge),
            "compressed",
        )
    else:
        raise ValueError(f"Unknown size mode: {size_mode}")

    if background_mode == "none":
        variant_image = content
        content_box = (0, 0, content.size[0], content.size[1])
    elif background_mode == "color":
        variant_image, content_box = add_background_canvas(
            content,
            background_color=background_color,
            padding_ratio=background_padding_ratio,
        )
    else:
        raise ValueError(f"Unknown background mode: {background_mode}")

    return ImageVariant(
        name=f"{size_mode}+bg_{background_mode}",
        size_mode=size_mode,
        background_mode=background_mode,
        image=variant_image,
        content_box=content_box,
    )


def prepare_image_variant(image_variant: ImageVariant, preprocess_spec: PreprocessSpec) -> PreparedImageVariant:
    prepared = prepare_benchmark_image(image_variant.image, preprocess_spec)
    return PreparedImageVariant(
        name=f"{image_variant.name}+{preprocess_spec.name}",
        size_mode=image_variant.size_mode,
        background_mode=image_variant.background_mode,
        ui_filter_mode=preprocess_spec.ui_filter_mode,
        color_policy=preprocess_spec.color_policy,
        encode_mode=preprocess_spec.encode_mode,
        image=prepared.image,
        content_box=image_variant.content_box,
        preprocess_latency_ms=prepared.preprocess_latency_ms,
        encode_latency_ms=prepared.encode_latency_ms,
        payload_bytes=prepared.payload_bytes,
        payload_format=prepared.payload_format,
    )


def select_experiment_axes(
    experiment_mode: str,
    normalizing_ranges: tuple[int, ...],
    prompt_variants: tuple[str, ...],
    size_modes: tuple[str, ...],
    background_modes: tuple[str, ...],
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if experiment_mode == "full":
        return normalizing_ranges, prompt_variants, size_modes, background_modes

    staged_ranges = tuple(value for value in normalizing_ranges if value in {250, 500}) or normalizing_ranges[:1]
    staged_prompts = (
        tuple(value for value in prompt_variants if value in {"long_prompt", "concise_prompt"}) or (prompt_variants[:1])
    )
    staged_sizes = tuple(
        value for value in size_modes if value in {"compressed", "compressed_downscale_restore", "downscale_restore"}
    )
    if "raw" in size_modes:
        staged_sizes = tuple(dict.fromkeys((*staged_sizes, "raw")))
    staged_sizes = staged_sizes or size_modes[:1]
    staged_backgrounds = tuple(value for value in background_modes if value == "none") or background_modes[:1]
    return staged_ranges, staged_prompts, staged_sizes, staged_backgrounds


def build_prepared_variants(
    base_image: PIL.Image.Image,
    size_modes: tuple[str, ...],
    background_modes: tuple[str, ...],
    preprocess_specs: list[PreprocessSpec],
    downscale_long_edge: int,
    background_color: tuple[int, int, int],
    background_padding_ratio: float,
) -> list[PreparedImageVariant]:
    prepared_variants: list[PreparedImageVariant] = []
    for size_mode in size_modes:
        for background_mode in background_modes:
            base_variant = build_image_variant(
                base_image=base_image,
                size_mode=size_mode,
                background_mode=background_mode,
                downscale_long_edge=downscale_long_edge,
                background_color=background_color,
                background_padding_ratio=background_padding_ratio,
            )
            for preprocess_spec in preprocess_specs:
                try:
                    prepared_variants.append(prepare_image_variant(base_variant, preprocess_spec))
                except UnsupportedEncodingModeError as exc:
                    console.print(
                        f"[yellow]skip[/yellow] variant={base_variant.name}+{preprocess_spec.name} reason={exc}"
                    )
    return prepared_variants


def make_variant_key(row: dict) -> tuple[str, int, str, str, str, str, str, str]:
    return (
        row["benchmark_type"],
        row["normalizing_range"],
        row["prompt_variant"],
        row["size_variant"],
        row["background_variant"],
        row["ui_filter_mode"],
        row["color_policy"],
        row["encode_mode"],
    )


def _restore_norm_coord(
    value: int,
    variant_extent: int,
    content_offset: int,
    content_extent: int,
    normalizing_range: int,
) -> int:
    clamped = max(0, min(normalizing_range, int(value)))
    if variant_extent <= 0 or content_extent <= 0 or normalizing_range <= 0:
        return clamped

    variant_px = (clamped / normalizing_range) * variant_extent
    content_px = variant_px - content_offset
    content_px = max(0.0, min(float(content_extent), content_px))

    restored = int(round((content_px / content_extent) * normalizing_range))
    return max(0, min(normalizing_range, restored))


def restore_actions_to_base_norm(
    actions: list[dict],
    variant: ImageVariant,
    normalizing_range: int,
) -> list[dict]:
    left, top, content_w, content_h = variant.content_box
    variant_w, variant_h = variant.image.size

    restored_actions: list[dict] = []
    for action in actions:
        restored = dict(action)

        if "x" in restored:
            restored["x"] = _restore_norm_coord(
                value=restored["x"],
                variant_extent=variant_w,
                content_offset=left,
                content_extent=content_w,
                normalizing_range=normalizing_range,
            )
        if "y" in restored:
            restored["y"] = _restore_norm_coord(
                value=restored["y"],
                variant_extent=variant_h,
                content_offset=top,
                content_extent=content_h,
                normalizing_range=normalizing_range,
            )
        if "end_x" in restored:
            restored["end_x"] = _restore_norm_coord(
                value=restored["end_x"],
                variant_extent=variant_w,
                content_offset=left,
                content_extent=content_w,
                normalizing_range=normalizing_range,
            )
        if "end_y" in restored:
            restored["end_y"] = _restore_norm_coord(
                value=restored["end_y"],
                variant_extent=variant_h,
                content_offset=top,
                content_extent=content_h,
                normalizing_range=normalizing_range,
            )

        restored_actions.append(restored)

    return restored_actions


def capture_live_screenshot(countdown_sec: int = 3, save_path: str | None = None) -> PIL.Image.Image:
    gui = _require_pyautogui()
    if countdown_sec > 0:
        print(f"[INFO] {countdown_sec}초 후 현재 화면을 캡처합니다. 문명6 화면으로 전환하세요.")
        time.sleep(countdown_sec)

    screenshot = gui.screenshot()
    screenshot = screenshot.convert("RGB")

    if save_path:
        screenshot.save(save_path)
        print(f"[INFO] 캡처 이미지 저장: {save_path}")

    return screenshot


def run_single(
    client: genai.Client,
    model: str,
    image: PIL.Image.Image,
    prompt: str,
    action_schema: dict,
) -> tuple[list[dict], float]:
    start = time.perf_counter()
    resp = client.models.generate_content(
        model=model,
        contents=[prompt, image],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=action_schema,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    elapsed = time.perf_counter() - start
    payload = json.loads(resp.text)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected response type: {type(payload)}")
    return payload, elapsed


def benchmark_grid(
    image_path: str | None,
    model: str,
    runs: int = 1,
    capture_live: bool = True,
    capture_countdown_sec: int = 3,
    save_capture_path: str | None = None,
    normalizing_ranges: tuple[int, ...] = (1000,),
    prompt_variants: tuple[str, ...] = ("long_prompt", "concise_prompt"),
    size_modes: tuple[str, ...] = ("raw", "compressed", "downscale_restore", "compressed_downscale_restore"),
    background_modes: tuple[str, ...] = ("none", "color"),
    preprocess_specs: list[PreprocessSpec] | None = None,
    downscale_long_edge: int = 960,
    background_color: tuple[int, int, int] = (32, 32, 32),
    background_padding_ratio: float = 0.12,
    experiment_mode: str = "staged",
) -> list[dict]:
    api_key = os.getenv("GENAI_API_KEY")
    if not api_key:
        raise ValueError("GENAI_API_KEY is not set")

    client = genai.Client(api_key=api_key)
    if capture_live:
        base_image = capture_live_screenshot(
            countdown_sec=capture_countdown_sec,
            save_path=save_capture_path,
        )
    else:
        if not image_path:
            raise ValueError("capture_live=False 인 경우 --image 경로가 필요합니다")
        base_image = PIL.Image.open(image_path).convert("RGB")

    selected_ranges, selected_prompts, selected_sizes, selected_backgrounds = select_experiment_axes(
        experiment_mode=experiment_mode,
        normalizing_ranges=normalizing_ranges,
        prompt_variants=prompt_variants,
        size_modes=size_modes,
        background_modes=background_modes,
    )
    preprocess_specs = preprocess_specs or [PreprocessSpec("baseline", "none", "preserve", "none")]
    prepared_variants = build_prepared_variants(
        base_image=base_image,
        size_modes=selected_sizes,
        background_modes=selected_backgrounds,
        preprocess_specs=preprocess_specs,
        downscale_long_edge=downscale_long_edge,
        background_color=background_color,
        background_padding_ratio=background_padding_ratio,
    )

    results: list[dict] = []
    total_calls = len(selected_ranges) * len(selected_prompts) * len(prepared_variants) * runs
    done_calls = 0

    with Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Inference Progress", total=total_calls)

        for normalizing_range in selected_ranges:
            action_schema = build_action_schema(normalizing_range)
            prompts = {prompt_name: build_prompt(prompt_name, normalizing_range) for prompt_name in selected_prompts}

            for prompt_name, prompt in prompts.items():
                for image_variant in prepared_variants:
                    variant_tag = f"{normalizing_range}/{prompt_name}/{image_variant.name}"
                    console.print(f"[bold]실험 시작[/bold] variant={variant_tag}, runs={runs}")
                    inference_times: list[float] = []
                    last_payload_raw: list[dict] = []
                    last_payload_restored: list[dict] = []
                    for _ in range(runs):
                        run_idx = len(inference_times) + 1
                        console.print(
                            f"[dim]  -> run {run_idx}/{runs} | variant={variant_tag} | calling model...[/dim]"
                        )
                        payload_raw, elapsed = run_single(
                            client,
                            model,
                            image_variant.image,
                            prompt,
                            action_schema=action_schema,
                        )
                        payload_restored = restore_actions_to_base_norm(
                            payload_raw,
                            image_variant,
                            normalizing_range=normalizing_range,
                        )
                        last_payload_raw = payload_raw
                        last_payload_restored = payload_restored
                        inference_times.append(elapsed)
                        done_calls += 1
                        progress.update(task_id, completed=done_calls)
                        console.print(f"[dim]     done run {run_idx}/{runs} | elapsed={elapsed:.3f}s[/dim]")

                    avg_inference = sum(inference_times) / len(inference_times)
                    preprocess_sec = image_variant.preprocess_latency_ms / 1000
                    first_action = last_payload_restored[0]["action"] if last_payload_restored else "n/a"
                    result = {
                        "benchmark_type": "latency_static",
                        "prompt_variant": prompt_name,
                        "image_variant": image_variant.name,
                        "size_variant": image_variant.size_mode,
                        "background_variant": image_variant.background_mode,
                        "ui_filter_mode": image_variant.ui_filter_mode,
                        "color_policy": image_variant.color_policy,
                        "encode_mode": image_variant.encode_mode,
                        "payload_format": image_variant.payload_format,
                        "payload_bytes": image_variant.payload_bytes,
                        "variant_image_size": f"{image_variant.image.size[0]}x{image_variant.image.size[1]}",
                        "content_box": image_variant.content_box,
                        "normalizing_range": normalizing_range,
                        "runs": runs,
                        "preprocess_latency_ms": round(image_variant.preprocess_latency_ms, 3),
                        "encode_latency_ms": round(image_variant.encode_latency_ms, 3),
                        "avg_inference_latency_sec": round(avg_inference, 3),
                        "avg_latency_sec": round(avg_inference + preprocess_sec, 3),
                        "min_latency_sec": round(min(inference_times) + preprocess_sec, 3),
                        "max_latency_sec": round(max(inference_times) + preprocess_sec, 3),
                        "num_actions": len(last_payload_restored),
                        "first_action": first_action,
                        "result": last_payload_restored,
                        "result_raw": last_payload_raw,
                    }
                    results.append(result)
                    console.print(
                        f"[green]실험 완료[/green] variant={variant_tag} "
                        f"| total_avg={result['avg_latency_sec']:.3f}s "
                        f"| infer_avg={avg_inference:.3f}s payload={image_variant.payload_bytes}B"
                    )

    show_experiment_summary(results)
    show_action_preview(results)
    return results


def benchmark_quality_dataset(
    dataset_path: str,
    model: str,
    normalizing_ranges: tuple[int, ...],
    prompt_variants: tuple[str, ...],
    size_modes: tuple[str, ...],
    background_modes: tuple[str, ...],
    preprocess_specs: list[PreprocessSpec],
    downscale_long_edge: int,
    background_color: tuple[int, int, int],
    background_padding_ratio: float,
    experiment_mode: str,
    ignore_wait: bool = True,
    case_limit: int | None = None,
) -> list[dict]:
    api_key = os.getenv("GENAI_API_KEY")
    if not api_key:
        raise ValueError("GENAI_API_KEY is not set")

    client = genai.Client(api_key=api_key)
    cases = load_dataset(dataset_path)
    if case_limit is not None:
        cases = cases[:case_limit]

    selected_ranges, selected_prompts, selected_sizes, selected_backgrounds = select_experiment_axes(
        experiment_mode=experiment_mode,
        normalizing_ranges=normalizing_ranges,
        prompt_variants=prompt_variants,
        size_modes=size_modes,
        background_modes=background_modes,
    )
    results: list[dict] = []
    total_calls = len(selected_ranges) * len(selected_prompts) * len(selected_sizes) * len(selected_backgrounds)
    total_calls *= len(preprocess_specs) * len(cases)
    done_calls = 0

    with Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Quality Eval Progress", total=total_calls)

        for normalizing_range in selected_ranges:
            action_schema = build_action_schema(normalizing_range)
            for prompt_variant in selected_prompts:
                for size_mode in selected_sizes:
                    for background_mode in selected_backgrounds:
                        for preprocess_spec in preprocess_specs:
                            case_results = []
                            inference_times: list[float] = []
                            preprocess_times_ms: list[float] = []
                            encode_times_ms: list[float] = []
                            payload_sizes: list[int] = []
                            sample_image_size = "n/a"

                            for case in cases:
                                screenshot_path = resolve_dataset_screenshot_path(dataset_path, case.screenshot_path)
                                base_image = PIL.Image.open(screenshot_path).convert("RGB")
                                base_variant = build_image_variant(
                                    base_image=base_image,
                                    size_mode=size_mode,
                                    background_mode=background_mode,
                                    downscale_long_edge=downscale_long_edge,
                                    background_color=background_color,
                                    background_padding_ratio=background_padding_ratio,
                                )
                                try:
                                    prepared_variant = prepare_image_variant(base_variant, preprocess_spec)
                                except UnsupportedEncodingModeError as exc:
                                    console.print(
                                        "[yellow]skip[/yellow] quality "
                                        f"variant={base_variant.name}+{preprocess_spec.name} "
                                        f"reason={exc}"
                                    )
                                    break

                                prompt = build_prompt(
                                    prompt_variant,
                                    normalizing_range,
                                    instruction=case.instruction,
                                )
                                payload_raw, elapsed = run_single(
                                    client=client,
                                    model=model,
                                    image=prepared_variant.image,
                                    prompt=prompt,
                                    action_schema=action_schema,
                                )
                                payload_restored = restore_actions_to_base_norm(
                                    payload_raw,
                                    prepared_variant,
                                    normalizing_range=normalizing_range,
                                )
                                case_result = score_actions_against_case(
                                    case=case,
                                    actions=payload_restored,
                                    ignore_wait=ignore_wait,
                                )
                                case_results.append(case_result)
                                inference_times.append(elapsed)
                                preprocess_times_ms.append(prepared_variant.preprocess_latency_ms)
                                encode_times_ms.append(prepared_variant.encode_latency_ms)
                                payload_sizes.append(prepared_variant.payload_bytes)
                                sample_image_size = f"{prepared_variant.image.size[0]}x{prepared_variant.image.size[1]}"
                                done_calls += 1
                                progress.update(task_id, completed=done_calls)

                            else:
                                aggregate = aggregate_results(case_results)
                                avg_inference = sum(inference_times) / len(inference_times) if inference_times else 0.0
                                avg_preprocess_ms = (
                                    sum(preprocess_times_ms) / len(preprocess_times_ms) if preprocess_times_ms else 0.0
                                )
                                avg_encode_ms = sum(encode_times_ms) / len(encode_times_ms) if encode_times_ms else 0.0
                                avg_payload_bytes = (
                                    int(round(sum(payload_sizes) / len(payload_sizes))) if payload_sizes else 0
                                )
                                results.append(
                                    {
                                        "benchmark_type": "quality_dataset",
                                        "prompt_variant": prompt_variant,
                                        "image_variant": f"{size_mode}+bg_{background_mode}+{preprocess_spec.name}",
                                        "size_variant": size_mode,
                                        "background_variant": background_mode,
                                        "ui_filter_mode": preprocess_spec.ui_filter_mode,
                                        "color_policy": preprocess_spec.color_policy,
                                        "encode_mode": preprocess_spec.encode_mode,
                                        "payload_format": "mixed",
                                        "payload_bytes": avg_payload_bytes,
                                        "variant_image_size": sample_image_size,
                                        "content_box": None,
                                        "normalizing_range": normalizing_range,
                                        "runs": len(case_results),
                                        "preprocess_latency_ms": round(avg_preprocess_ms, 3),
                                        "encode_latency_ms": round(avg_encode_ms, 3),
                                        "avg_inference_latency_sec": round(avg_inference, 3),
                                        "avg_latency_sec": round(avg_inference + (avg_preprocess_ms / 1000), 3),
                                        "min_latency_sec": round(
                                            min(inference_times) + (avg_preprocess_ms / 1000),
                                            3,
                                        )
                                        if inference_times
                                        else 0.0,
                                        "max_latency_sec": round(
                                            max(inference_times) + (avg_preprocess_ms / 1000),
                                            3,
                                        )
                                        if inference_times
                                        else 0.0,
                                        "num_actions": 0,
                                        "first_action": "n/a",
                                        "result": [],
                                        "result_raw": [],
                                        "total_cases": aggregate.total_cases,
                                        "strict_success_rate": round(aggregate.strict_success_rate, 4),
                                        "avg_step_accuracy": round(aggregate.avg_step_accuracy, 4),
                                        "avg_prefix_len": round(aggregate.avg_prefix_len, 4),
                                        "error_count": aggregate.error_count,
                                        "timeout_count": aggregate.timeout_count,
                                        "per_action_type": [
                                            metric.model_dump() for metric in aggregate.per_action_type
                                        ],
                                        "case_results": [case_result.model_dump() for case_result in case_results],
                                    }
                                )

    apply_quality_gates(results)
    show_quality_summary(results)
    return results


def apply_quality_gates(rows: list[dict]) -> None:
    baseline_rows = {}
    for row in rows:
        if row["ui_filter_mode"] == "none" and row["color_policy"] == "preserve" and row["encode_mode"] == "none":
            baseline_rows[
                (row["normalizing_range"], row["prompt_variant"], row["size_variant"], row["background_variant"])
            ] = row

    for row in rows:
        baseline = baseline_rows.get(
            (row["normalizing_range"], row["prompt_variant"], row["size_variant"], row["background_variant"])
        )
        if baseline is None:
            row["quality_gate_pass"] = True
            continue
        row["quality_gate_pass"] = row["avg_step_accuracy"] >= max(0.0, baseline["avg_step_accuracy"] - 0.05) and row[
            "strict_success_rate"
        ] >= max(0.0, baseline["strict_success_rate"] - 0.05)


def show_experiment_summary(results: list[dict]) -> None:
    console.rule("[bold cyan]VLM Policy Action Benchmark")
    prompt_set = sorted({r["prompt_variant"] for r in results})
    image_set = sorted({r["image_variant"] for r in results})
    console.print(f"[bold]비교군[/bold]: {len(prompt_set)}개 prompt x {len(image_set)}개 image variant")

    table = Table(title="Latency / Action Summary", show_lines=False)
    table.add_column("Range", justify="right")
    table.add_column("Prompt", style="magenta")
    table.add_column("Image", style="cyan")
    table.add_column("Size", style="yellow")
    table.add_column("BG", style="blue")
    table.add_column("UI", style="green")
    table.add_column("Color", style="green")
    table.add_column("Enc", style="green")
    table.add_column("Payload", justify="right")
    table.add_column("ImgSize", style="white")
    table.add_column("Prep(ms)", justify="right")
    table.add_column("Infer(s)", justify="right")
    table.add_column("Total(s)", justify="right")
    table.add_column("Actions", justify="right")
    table.add_column("First", style="green")

    for row in results:
        table.add_row(
            str(row["normalizing_range"]),
            row["prompt_variant"],
            row["image_variant"],
            row["size_variant"],
            row["background_variant"],
            row["ui_filter_mode"],
            row["color_policy"],
            row["encode_mode"],
            str(row["payload_bytes"]),
            row["variant_image_size"],
            f"{row['preprocess_latency_ms']:.1f}",
            f"{row['avg_inference_latency_sec']:.3f}",
            f"{row['avg_latency_sec']:.3f}",
            str(row["num_actions"]),
            row["first_action"],
        )
    console.print(table)


def show_quality_summary(results: list[dict]) -> None:
    if not results:
        return

    console.rule("[bold cyan]Quality Benchmark Summary")
    table = Table(title="Quality / Latency Summary", show_lines=False)
    table.add_column("Range", justify="right")
    table.add_column("Prompt", style="magenta")
    table.add_column("Image", style="cyan")
    table.add_column("UI", style="green")
    table.add_column("Color", style="green")
    table.add_column("Enc", style="green")
    table.add_column("Total(s)", justify="right")
    table.add_column("Strict", justify="right")
    table.add_column("StepAcc", justify="right")
    table.add_column("Prefix", justify="right")
    table.add_column("Gate", justify="center")

    best_rows = sorted(
        results,
        key=lambda row: (
            not row.get("quality_gate_pass", False),
            -row["avg_step_accuracy"],
            row["avg_latency_sec"],
        ),
    )[:12]
    for row in best_rows:
        table.add_row(
            str(row["normalizing_range"]),
            row["prompt_variant"],
            row["image_variant"],
            row["ui_filter_mode"],
            row["color_policy"],
            row["encode_mode"],
            f"{row['avg_latency_sec']:.3f}",
            f"{row['strict_success_rate']:.3f}",
            f"{row['avg_step_accuracy']:.3f}",
            f"{row['avg_prefix_len']:.3f}",
            "pass" if row.get("quality_gate_pass", False) else "fail",
        )
    console.print(table)


def show_action_preview(results: list[dict], limit: int = 8) -> None:
    preview = Table(title="Action Preview (readable)", show_lines=True)
    preview.add_column("Variant", style="yellow")
    preview.add_column("Step", justify="right")
    preview.add_column("Action", style="cyan")
    preview.add_column("Coords", overflow="fold")
    preview.add_column("Button/Key/Text", overflow="fold")
    preview.add_column("Reasoning", overflow="fold")

    preview_rows = sorted(results, key=lambda row: row.get("avg_latency_sec", 0.0))[:limit]
    for row in preview_rows:
        variant = (
            f"{row['prompt_variant']} + {row['image_variant']}"
            f" (size={row['size_variant']}, bg={row['background_variant']}, "
            f"ui={row['ui_filter_mode']}, color={row['color_policy']}, enc={row['encode_mode']})"
        )
        actions = row["result"]
        if not actions:
            preview.add_row(variant, "-", "-", "-", "-", "-")
            continue
        for idx, action in enumerate(actions, start=1):
            coords = (
                f"({action.get('x', 0)}, {action.get('y', 0)}) -> ({action.get('end_x', 0)}, {action.get('end_y', 0)})"
            )
            bkt = f"b={action.get('button', '')} | k={action.get('key', '')} | t={action.get('text', '')}"
            preview.add_row(
                variant,
                str(idx),
                action.get("action", ""),
                coords,
                bkt,
                action.get("reasoning", ""),
            )
    console.print(preview)


def save_reports(
    report_dir: str,
    report_prefix: str,
    config: dict,
    latency_results: list[dict],
    quality_results: list[dict],
) -> tuple[Path, Path]:
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{report_prefix}_{timestamp}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"

    payload = {
        "config": config,
        "latency_results": latency_results,
        "quality_results": quality_results,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(build_markdown_report(config, latency_results, quality_results), encoding="utf-8")
    return json_path, md_path


def build_markdown_report(config: dict, latency_results: list[dict], quality_results: list[dict]) -> str:
    lines = [
        "# VLM UI Benchmark Report",
        "",
        "## Setup",
        f"- Model: `{config['model']}`",
        f"- Experiment mode: `{config['experiment_mode']}`",
        f"- Ranges: `{', '.join(map(str, config['normalizing_ranges']))}`",
        f"- Prompts: `{', '.join(config['prompt_variants'])}`",
        f"- Size modes: `{', '.join(config['size_modes'])}`",
        f"- Background modes: `{', '.join(config['background_modes'])}`",
        f"- UI filters: `{', '.join(config['ui_filter_modes'])}`",
        f"- Color policies: `{', '.join(config['color_policies'])}`",
        f"- Encode modes: `{', '.join(config['encode_modes'])}`",
    ]

    if latency_results:
        lines.extend(
            [
                "",
                "## Latency Leaders",
                "| range | prompt | image_variant | total_latency_sec | payload_bytes |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in sorted(latency_results, key=lambda item: item["avg_latency_sec"])[:10]:
            lines.append(
                f"| {row['normalizing_range']} | {row['prompt_variant']} | {row['image_variant']} | "
                f"{row['avg_latency_sec']:.3f} | {row['payload_bytes']} |"
            )

    if quality_results:
        lines.extend(
            [
                "",
                "## Quality Leaders",
                "| range | prompt | image_variant | total_latency_sec | "
                "strict_success_rate | avg_step_accuracy | gate |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in sorted(
            quality_results,
            key=lambda item: (
                not item.get("quality_gate_pass", False),
                -item["avg_step_accuracy"],
                item["avg_latency_sec"],
            ),
        )[:10]:
            lines.append(
                f"| {row['normalizing_range']} | {row['prompt_variant']} | {row['image_variant']} | "
                f"{row['avg_latency_sec']:.3f} | {row['strict_success_rate']:.3f} | "
                f"{row['avg_step_accuracy']:.3f} | {'pass' if row.get('quality_gate_pass', False) else 'fail'} |"
            )

    return "\n".join(lines) + "\n"


def norm_to_real(norm_value: int, screen_extent: int, normalizing_range: int) -> int:
    if normalizing_range <= 0:
        raise ValueError("normalizing_range must be > 0")
    clamped = max(0, min(normalizing_range, int(norm_value)))
    return int((clamped / normalizing_range) * screen_extent)


def execute_action_sequence_live(
    actions: list[dict],
    normalizing_range: int,
    action_delay_sec: float = 0.25,
    move_duration_sec: float = 0.2,
    drag_hold_sec: float = 0.08,
    drag_drop_hold_sec: float = 0.05,
) -> None:
    gui = _require_pyautogui()
    screen_w, screen_h = gui.size()

    for idx, action in enumerate(actions, start=1):
        action_type = str(action.get("action", "")).strip()
        console.print(f"[bold green]실행[/bold green] step={idx} action={action_type}")

        if action_type == "click":
            x = norm_to_real(action.get("x", 0), screen_w, normalizing_range)
            y = norm_to_real(action.get("y", 0), screen_h, normalizing_range)
            button = str(action.get("button", "left") or "left")
            gui.moveTo(x, y, duration=max(0.0, move_duration_sec))
            gui.click(button=button)

        elif action_type == "drag":
            start_x = norm_to_real(action.get("x", 0), screen_w, normalizing_range)
            start_y = norm_to_real(action.get("y", 0), screen_h, normalizing_range)
            end_x = norm_to_real(action.get("end_x", 0), screen_w, normalizing_range)
            end_y = norm_to_real(action.get("end_y", 0), screen_h, normalizing_range)
            button = str(action.get("button", "left") or "left")
            console.print(f"[cyan]drag[/cyan] start=({start_x}, {start_y}) end=({end_x}, {end_y}) button={button}")
            gui.moveTo(start_x, start_y, duration=max(0.0, move_duration_sec))
            gui.mouseDown(x=start_x, y=start_y, button=button)
            if drag_hold_sec > 0:
                time.sleep(drag_hold_sec)
            gui.moveTo(end_x, end_y, duration=max(0.0, move_duration_sec))
            if drag_drop_hold_sec > 0:
                time.sleep(drag_drop_hold_sec)
            gui.mouseUp(x=end_x, y=end_y, button=button)

        elif action_type == "press":
            key = str(action.get("key", "")).strip()
            if key and key.lower() != "none":
                gui.press(key)
            else:
                console.print("[yellow]skip[/yellow] press action has empty key")

        elif action_type == "type":
            text = str(action.get("text", ""))
            if text and text.lower() != "none":
                gui.write(text, interval=0.06)
            else:
                console.print("[yellow]skip[/yellow] type action has empty text")

        else:
            console.print(f"[yellow]skip[/yellow] unsupported action type: {action_type}")

        if action_delay_sec > 0:
            time.sleep(action_delay_sec)


def run_live_variant(
    client: genai.Client,
    model: str,
    normalizing_range: int,
    prompt_variant: str,
    size_mode: str,
    background_mode: str,
    preprocess_spec: PreprocessSpec,
    downscale_long_edge: int,
    background_color: tuple[int, int, int],
    background_padding_ratio: float,
    capture_countdown_sec: int,
    execute_actions: bool,
    confirm_before_exec: bool,
    pre_exec_countdown_sec: int,
    action_delay_sec: float,
    move_duration_sec: float,
    drag_hold_sec: float,
    drag_drop_hold_sec: float,
    save_path: str | None = None,
    variant_label: str | None = None,
) -> dict:
    action_schema = build_action_schema(normalizing_range)
    prompt = build_prompt(prompt_variant, normalizing_range)

    if variant_label:
        console.rule(f"[bold cyan]{variant_label}")

    base_image = capture_live_screenshot(
        countdown_sec=capture_countdown_sec,
        save_path=save_path,
    )
    image_variant = build_image_variant(
        base_image=base_image,
        size_mode=size_mode,
        background_mode=background_mode,
        downscale_long_edge=downscale_long_edge,
        background_color=background_color,
        background_padding_ratio=background_padding_ratio,
    )
    prepared_variant = prepare_image_variant(image_variant, preprocess_spec)

    payload_raw, elapsed = run_single(
        client=client,
        model=model,
        image=prepared_variant.image,
        prompt=prompt,
        action_schema=action_schema,
    )
    payload_restored = restore_actions_to_base_norm(
        payload_raw,
        prepared_variant,
        normalizing_range=normalizing_range,
    )

    row = {
        "benchmark_type": "live_single",
        "prompt_variant": prompt_variant,
        "image_variant": f"{image_variant.name}+{preprocess_spec.name}",
        "size_variant": image_variant.size_mode,
        "background_variant": image_variant.background_mode,
        "ui_filter_mode": preprocess_spec.ui_filter_mode,
        "color_policy": preprocess_spec.color_policy,
        "encode_mode": preprocess_spec.encode_mode,
        "variant_image_size": f"{prepared_variant.image.size[0]}x{prepared_variant.image.size[1]}",
        "content_box": prepared_variant.content_box,
        "normalizing_range": normalizing_range,
        "runs": 1,
        "payload_format": prepared_variant.payload_format,
        "payload_bytes": prepared_variant.payload_bytes,
        "preprocess_latency_ms": round(prepared_variant.preprocess_latency_ms, 3),
        "encode_latency_ms": round(prepared_variant.encode_latency_ms, 3),
        "avg_inference_latency_sec": round(elapsed, 3),
        "avg_latency_sec": round(elapsed + (prepared_variant.preprocess_latency_ms / 1000), 3),
        "min_latency_sec": round(elapsed + (prepared_variant.preprocess_latency_ms / 1000), 3),
        "max_latency_sec": round(elapsed + (prepared_variant.preprocess_latency_ms / 1000), 3),
        "num_actions": len(payload_restored),
        "first_action": payload_restored[0]["action"] if payload_restored else "n/a",
        "result": payload_restored,
        "result_raw": payload_raw,
    }

    console.print(
        f"[green]추론 완료[/green] total={row['avg_latency_sec']:.3f}s "
        f"infer={elapsed:.3f}s payload={prepared_variant.payload_bytes}B"
    )
    show_action_preview([row])

    if execute_actions and payload_restored:
        should_run = True
        if confirm_before_exec:
            answer = console.input("[bold yellow]이 액션을 실제 실행하려면 'yes' 입력: [/bold yellow]").strip().lower()
            should_run = answer == "yes"

        if should_run:
            if pre_exec_countdown_sec > 0:
                console.print(f"[red]실행까지 {pre_exec_countdown_sec}초[/red]")
                time.sleep(pre_exec_countdown_sec)
            execute_action_sequence_live(
                payload_restored,
                normalizing_range=normalizing_range,
                action_delay_sec=action_delay_sec,
                move_duration_sec=move_duration_sec,
                drag_hold_sec=drag_hold_sec,
                drag_drop_hold_sec=drag_drop_hold_sec,
            )
        else:
            console.print("[yellow]실행 스킵[/yellow] 사용자 확인 미통과")

    return row


def run_live_civ6_test(
    model: str,
    normalizing_range: int,
    prompt_variant: str,
    size_mode: str,
    background_mode: str,
    preprocess_spec: PreprocessSpec,
    downscale_long_edge: int,
    background_color: tuple[int, int, int],
    background_padding_ratio: float,
    live_steps: int = 1,
    capture_countdown_sec: int = 3,
    execute_actions: bool = False,
    confirm_before_exec: bool = False,
    pre_exec_countdown_sec: int = 3,
    action_delay_sec: float = 0.25,
    move_duration_sec: float = 0.2,
    drag_hold_sec: float = 0.08,
    drag_drop_hold_sec: float = 0.05,
    step_interval_sec: float = 1.0,
    save_live_captures_dir: str | None = None,
) -> list[dict]:
    api_key = os.getenv("GENAI_API_KEY")
    if not api_key:
        raise ValueError("GENAI_API_KEY is not set")

    if prompt_variant not in ALLOWED_PROMPT_VARIANTS:
        raise ValueError(f"Unknown prompt variant: {prompt_variant}")

    client = genai.Client(api_key=api_key)

    capture_dir = None
    if save_live_captures_dir:
        capture_dir = os.path.abspath(save_live_captures_dir)
        os.makedirs(capture_dir, exist_ok=True)

    results: list[dict] = []
    console.rule("[bold cyan]Live Civ6 Test")
    console.print(
        f"[bold]설정[/bold] prompt={prompt_variant}, size={size_mode}, bg={background_mode}, "
        f"ui={preprocess_spec.ui_filter_mode}, color={preprocess_spec.color_policy}, "
        f"enc={preprocess_spec.encode_mode}, "
        f"range={normalizing_range}, steps={live_steps}, execute_actions={execute_actions}"
    )
    console.print("[yellow]주의[/yellow] 마우스를 화면 왼쪽 위 모서리로 이동하면 PyAutoGUI failsafe로 중단됩니다.")

    for step in range(1, live_steps + 1):
        save_path = None
        if capture_dir:
            save_path = os.path.join(capture_dir, f"live_step_{step:02d}.png")

        row = run_live_variant(
            client=client,
            model=model,
            normalizing_range=normalizing_range,
            prompt_variant=prompt_variant,
            size_mode=size_mode,
            background_mode=background_mode,
            preprocess_spec=preprocess_spec,
            downscale_long_edge=downscale_long_edge,
            background_color=background_color,
            background_padding_ratio=background_padding_ratio,
            capture_countdown_sec=capture_countdown_sec,
            execute_actions=execute_actions,
            confirm_before_exec=confirm_before_exec,
            pre_exec_countdown_sec=pre_exec_countdown_sec,
            action_delay_sec=action_delay_sec,
            move_duration_sec=move_duration_sec,
            drag_hold_sec=drag_hold_sec,
            drag_drop_hold_sec=drag_drop_hold_sec,
            save_path=save_path,
            variant_label=f"Live Civ6 Test {step}/{live_steps}",
        )
        results.append(row)

        if step < live_steps and step_interval_sec > 0:
            time.sleep(step_interval_sec)

    console.rule("[bold cyan]Live Test Summary")
    show_experiment_summary(results)
    return results


def run_live_civ6_grid(
    model: str,
    normalizing_ranges: tuple[int, ...],
    prompt_variants: tuple[str, ...],
    size_modes: tuple[str, ...],
    background_modes: tuple[str, ...],
    preprocess_spec: PreprocessSpec,
    downscale_long_edge: int,
    background_color: tuple[int, int, int],
    background_padding_ratio: float,
    capture_countdown_sec: int = 3,
    execute_actions: bool = False,
    confirm_before_exec: bool = False,
    pre_exec_countdown_sec: int = 3,
    action_delay_sec: float = 0.25,
    move_duration_sec: float = 0.2,
    drag_hold_sec: float = 0.08,
    drag_drop_hold_sec: float = 0.05,
    step_interval_sec: float = 1.0,
    save_live_captures_dir: str | None = None,
) -> list[dict]:
    api_key = os.getenv("GENAI_API_KEY")
    if not api_key:
        raise ValueError("GENAI_API_KEY is not set")

    client = genai.Client(api_key=api_key)

    capture_dir = None
    if save_live_captures_dir:
        capture_dir = os.path.abspath(save_live_captures_dir)
        os.makedirs(capture_dir, exist_ok=True)

    total_variants = len(normalizing_ranges) * len(prompt_variants) * len(size_modes) * len(background_modes)
    results: list[dict] = []

    console.rule("[bold cyan]Live Civ6 Grid Test")
    console.print(
        "[bold]설정[/bold] "
        f"ranges={list(normalizing_ranges)}, prompts={list(prompt_variants)}, "
        f"size_modes={list(size_modes)}, bg_modes={list(background_modes)}, "
        f"ui={preprocess_spec.ui_filter_mode}, color={preprocess_spec.color_policy}, "
        f"enc={preprocess_spec.encode_mode}, "
        f"execute_actions={execute_actions}"
    )
    console.print(f"[yellow]총 조합[/yellow] {total_variants}")

    variant_index = 0
    for normalizing_range in normalizing_ranges:
        for prompt_variant in prompt_variants:
            for size_mode in size_modes:
                for background_mode in background_modes:
                    variant_index += 1
                    label = (
                        f"Grid {variant_index}/{total_variants} "
                        f"| range={normalizing_range} "
                        f"| prompt={prompt_variant} "
                        f"| size={size_mode} "
                        f"| bg={background_mode}"
                    )
                    save_path = None
                    if capture_dir:
                        save_path = os.path.join(capture_dir, f"grid_step_{variant_index:02d}.png")

                    row = run_live_variant(
                        client=client,
                        model=model,
                        normalizing_range=normalizing_range,
                        prompt_variant=prompt_variant,
                        size_mode=size_mode,
                        background_mode=background_mode,
                        preprocess_spec=preprocess_spec,
                        downscale_long_edge=downscale_long_edge,
                        background_color=background_color,
                        background_padding_ratio=background_padding_ratio,
                        capture_countdown_sec=capture_countdown_sec,
                        execute_actions=execute_actions,
                        confirm_before_exec=confirm_before_exec,
                        pre_exec_countdown_sec=pre_exec_countdown_sec,
                        action_delay_sec=action_delay_sec,
                        move_duration_sec=move_duration_sec,
                        drag_hold_sec=drag_hold_sec,
                        drag_drop_hold_sec=drag_drop_hold_sec,
                        save_path=save_path,
                        variant_label=label,
                    )
                    results.append(row)

                    if variant_index < total_variants and step_interval_sec > 0:
                        time.sleep(step_interval_sec)

    console.rule("[bold cyan]Live Grid Summary")
    show_experiment_summary(results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Path to screenshot image (옵션)")
    parser.add_argument("--model", default="gemini-3-flash-preview", help="Gemini model")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per variant")
    parser.add_argument("--normalizing-range", type=int, default=1000, help="정규화 좌표 범위 (예: 1000)")
    parser.add_argument(
        "--size-modes",
        default="raw,compressed,downscale_restore,compressed_downscale_restore",
        help="이미지 크기 변인 CSV (raw,compressed,downscale_restore,compressed_downscale_restore)",
    )
    parser.add_argument(
        "--background-modes",
        default="none,color",
        help="배경 변인 CSV (none,color)",
    )
    parser.add_argument(
        "--benchmark-grid-ranges",
        default=None,
        help="정적 benchmark에서 사용할 normalizing_range CSV (미지정 시 --normalizing-range 단일값 사용)",
    )
    parser.add_argument(
        "--prompt-variants",
        default="long_prompt,concise_prompt",
        help="정적 benchmark에서 사용할 prompt 변인 CSV",
    )
    parser.add_argument(
        "--ui-filter-modes",
        default="none,ui_contrast,ui_quantized,ui_bg_blur,ui_bg_blur_contrast",
        help="UI 전처리 변인 CSV",
    )
    parser.add_argument(
        "--color-policies",
        default="preserve,grayscale,adaptive_gray",
        help="색상 정책 변인 CSV",
    )
    parser.add_argument(
        "--encode-modes",
        default="none,jpeg_like,webp_like,avif_like_if_supported",
        help="전송 인코딩 변인 CSV",
    )
    parser.add_argument(
        "--experiment-mode",
        choices=["staged", "full"],
        default="staged",
        help="staged는 추천 조합만, full은 전체 조합을 탐색",
    )
    parser.add_argument(
        "--quality-dataset",
        default=None,
        help="bbox_eval 형식 JSONL 데이터셋 경로. 지정 시 quality benchmark도 함께 실행",
    )
    parser.add_argument(
        "--quality-case-limit",
        type=int,
        default=None,
        help="quality benchmark에서 사용할 최대 case 수",
    )
    parser.add_argument(
        "--quality-ignore-wait",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="quality benchmark에서 wait 액션 무시 여부",
    )
    parser.add_argument(
        "--report-dir",
        default="tests/rough_test/reports",
        help="benchmark report 저장 디렉토리",
    )
    parser.add_argument(
        "--report-prefix",
        default="vlm_ui_benchmark",
        help="benchmark report 파일 prefix",
    )
    parser.add_argument(
        "--downscale-long-edge",
        type=int,
        default=960,
        help="size_mode=downscale_restore일 때 long edge 크기",
    )
    parser.add_argument(
        "--background-color",
        default="32,32,32",
        help="background_mode=color일 때 RGB (예: 32,32,32)",
    )
    parser.add_argument(
        "--background-padding-ratio",
        type=float,
        default=0.12,
        help="background_mode=color일 때 패딩 비율 (0.12=12%%)",
    )
    parser.add_argument(
        "--live-test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="실제 Civ6 화면에서 단일 변인으로 실시간 테스트 실행",
    )
    parser.add_argument(
        "--live-grid-test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="실제 Civ6 화면에서 전체 조합을 순회하며 실시간 테스트 실행",
    )
    parser.add_argument("--live-steps", type=int, default=1, help="live-test에서 반복 step 수")
    parser.add_argument(
        "--live-prompt-variant",
        choices=["long_prompt", "concise_prompt"],
        default="concise_prompt",
        help="live-test에서 사용할 프롬프트 변인",
    )
    parser.add_argument(
        "--live-size-mode",
        choices=sorted(ALLOWED_SIZE_MODES),
        default="compressed_downscale_restore",
        help="live-test에서 사용할 size 변인",
    )
    parser.add_argument(
        "--live-background-mode",
        choices=sorted(ALLOWED_BACKGROUND_MODES),
        default="none",
        help="live-test에서 사용할 배경 변인",
    )
    parser.add_argument(
        "--live-ui-filter-mode",
        choices=sorted(ALLOWED_UI_FILTER_MODES),
        default="ui_bg_blur_contrast",
        help="live-test에서 사용할 UI 전처리 변인",
    )
    parser.add_argument(
        "--live-color-policy",
        choices=sorted(ALLOWED_COLOR_POLICIES),
        default="adaptive_gray",
        help="live-test에서 사용할 색상 정책",
    )
    parser.add_argument(
        "--live-encode-mode",
        choices=sorted(ALLOWED_ENCODE_MODES),
        default="webp_like",
        help="live-test에서 사용할 전송 인코딩",
    )
    parser.add_argument(
        "--live-grid-ranges",
        default="250,500,1000",
        help="live-grid-test에서 사용할 normalizing_range CSV",
    )
    parser.add_argument(
        "--live-grid-prompt-variants",
        default="long_prompt,concise_prompt",
        help="live-grid-test에서 사용할 prompt 변인 CSV",
    )
    parser.add_argument(
        "--execute-actions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="live-test에서 추론 결과 action을 실제 입력으로 실행",
    )
    parser.add_argument(
        "--confirm-before-exec",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="live-test 실행 전 'yes' 확인 입력 요구",
    )
    parser.add_argument(
        "--pre-exec-countdown-sec",
        type=int,
        default=3,
        help="실행 직전 카운트다운 초",
    )
    parser.add_argument(
        "--action-delay-sec",
        type=float,
        default=0.25,
        help="action step 간 지연 시간",
    )
    parser.add_argument(
        "--move-duration-sec",
        type=float,
        default=0.2,
        help="마우스 이동/드래그 duration",
    )
    parser.add_argument(
        "--drag-hold-sec",
        type=float,
        default=0.08,
        help="drag에서 mouseDown 후 유지 시간",
    )
    parser.add_argument(
        "--drag-drop-hold-sec",
        type=float,
        default=0.05,
        help="drag 종료 지점에서 mouseUp 전 유지 시간",
    )
    parser.add_argument(
        "--step-interval-sec",
        type=float,
        default=1.0,
        help="live-test step 간 대기 시간",
    )
    parser.add_argument(
        "--save-live-captures-dir",
        default=None,
        help="live-test 캡처 저장 디렉토리 (옵션)",
    )
    parser.add_argument(
        "--capture-live",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="실시간 화면 캡처 사용 여부 (기본: 사용)",
    )
    parser.add_argument("--capture-countdown-sec", type=int, default=3, help="실시간 캡처 전 대기 초")
    parser.add_argument(
        "--save-capture-path",
        default=None,
        help="실시간 캡처 원본 저장 경로 (옵션, 예: civ6_live.png)",
    )
    args = parser.parse_args()

    size_modes = parse_csv_modes(args.size_modes, ALLOWED_SIZE_MODES, "--size-modes")
    prompt_variants = parse_csv_modes(args.prompt_variants, ALLOWED_PROMPT_VARIANTS, "--prompt-variants")
    background_modes = parse_csv_modes(args.background_modes, ALLOWED_BACKGROUND_MODES, "--background-modes")
    ui_filter_modes = parse_csv_modes(args.ui_filter_modes, ALLOWED_UI_FILTER_MODES, "--ui-filter-modes")
    color_policies = parse_csv_modes(args.color_policies, ALLOWED_COLOR_POLICIES, "--color-policies")
    encode_modes = parse_csv_modes(args.encode_modes, ALLOWED_ENCODE_MODES, "--encode-modes")
    background_color = parse_background_color(args.background_color)
    if args.normalizing_range <= 0:
        raise ValueError("--normalizing-range는 1 이상이어야 합니다.")
    if args.downscale_long_edge <= 0:
        raise ValueError("--downscale-long-edge는 1 이상이어야 합니다.")
    if args.background_padding_ratio < 0:
        raise ValueError("--background-padding-ratio는 0 이상이어야 합니다.")
    if args.live_steps <= 0:
        raise ValueError("--live-steps는 1 이상이어야 합니다.")
    if args.pre_exec_countdown_sec < 0:
        raise ValueError("--pre-exec-countdown-sec는 0 이상이어야 합니다.")
    if args.action_delay_sec < 0:
        raise ValueError("--action-delay-sec는 0 이상이어야 합니다.")
    if args.move_duration_sec < 0:
        raise ValueError("--move-duration-sec는 0 이상이어야 합니다.")
    if args.drag_hold_sec < 0:
        raise ValueError("--drag-hold-sec는 0 이상이어야 합니다.")
    if args.drag_drop_hold_sec < 0:
        raise ValueError("--drag-drop-hold-sec는 0 이상이어야 합니다.")
    if args.step_interval_sec < 0:
        raise ValueError("--step-interval-sec는 0 이상이어야 합니다.")
    if args.quality_case_limit is not None and args.quality_case_limit <= 0:
        raise ValueError("--quality-case-limit은 1 이상이어야 합니다.")

    normalizing_ranges = (
        tuple(parse_csv_ints(args.benchmark_grid_ranges, "--benchmark-grid-ranges"))
        if args.benchmark_grid_ranges
        else (args.normalizing_range,)
    )
    live_grid_ranges = tuple(parse_csv_ints(args.live_grid_ranges, "--live-grid-ranges"))
    live_grid_prompt_variants = tuple(
        parse_csv_modes(args.live_grid_prompt_variants, ALLOWED_PROMPT_VARIANTS, "--live-grid-prompt-variants")
    )
    preprocess_specs = build_preprocess_specs(
        ui_filter_modes=ui_filter_modes,
        color_policies=color_policies,
        encode_modes=encode_modes,
        experiment_mode=args.experiment_mode,
    )
    if not preprocess_specs:
        raise ValueError("선택한 staged/full 조합으로 생성된 preprocess spec이 없습니다.")
    live_preprocess_spec = PreprocessSpec(
        name="live_custom",
        ui_filter_mode=args.live_ui_filter_mode,
        color_policy=args.live_color_policy,
        encode_mode=args.live_encode_mode,
    )

    if args.live_test:
        run_live_civ6_test(
            model=args.model,
            normalizing_range=args.normalizing_range,
            prompt_variant=args.live_prompt_variant,
            size_mode=args.live_size_mode,
            background_mode=args.live_background_mode,
            preprocess_spec=live_preprocess_spec,
            downscale_long_edge=args.downscale_long_edge,
            background_color=background_color,
            background_padding_ratio=args.background_padding_ratio,
            live_steps=args.live_steps,
            capture_countdown_sec=args.capture_countdown_sec,
            execute_actions=args.execute_actions,
            confirm_before_exec=args.confirm_before_exec,
            pre_exec_countdown_sec=args.pre_exec_countdown_sec,
            action_delay_sec=args.action_delay_sec,
            move_duration_sec=args.move_duration_sec,
            drag_hold_sec=args.drag_hold_sec,
            drag_drop_hold_sec=args.drag_drop_hold_sec,
            step_interval_sec=args.step_interval_sec,
            save_live_captures_dir=args.save_live_captures_dir,
        )
        return

    if args.live_grid_test:
        run_live_civ6_grid(
            model=args.model,
            normalizing_ranges=live_grid_ranges,
            prompt_variants=live_grid_prompt_variants,
            size_modes=tuple(size_modes),
            background_modes=tuple(background_modes),
            preprocess_spec=live_preprocess_spec,
            downscale_long_edge=args.downscale_long_edge,
            background_color=background_color,
            background_padding_ratio=args.background_padding_ratio,
            capture_countdown_sec=args.capture_countdown_sec,
            execute_actions=args.execute_actions,
            confirm_before_exec=args.confirm_before_exec,
            pre_exec_countdown_sec=args.pre_exec_countdown_sec,
            action_delay_sec=args.action_delay_sec,
            move_duration_sec=args.move_duration_sec,
            drag_hold_sec=args.drag_hold_sec,
            drag_drop_hold_sec=args.drag_drop_hold_sec,
            step_interval_sec=args.step_interval_sec,
            save_live_captures_dir=args.save_live_captures_dir,
        )
        return

    latency_results = benchmark_grid(
        image_path=args.image,
        model=args.model,
        runs=args.runs,
        capture_live=args.capture_live,
        capture_countdown_sec=args.capture_countdown_sec,
        save_capture_path=args.save_capture_path,
        normalizing_ranges=normalizing_ranges,
        prompt_variants=tuple(prompt_variants),
        size_modes=tuple(size_modes),
        background_modes=tuple(background_modes),
        preprocess_specs=preprocess_specs,
        downscale_long_edge=args.downscale_long_edge,
        background_color=background_color,
        background_padding_ratio=args.background_padding_ratio,
        experiment_mode=args.experiment_mode,
    )
    quality_results = []
    if args.quality_dataset:
        quality_results = benchmark_quality_dataset(
            dataset_path=args.quality_dataset,
            model=args.model,
            normalizing_ranges=normalizing_ranges,
            prompt_variants=tuple(prompt_variants),
            size_modes=tuple(size_modes),
            background_modes=tuple(background_modes),
            preprocess_specs=preprocess_specs,
            downscale_long_edge=args.downscale_long_edge,
            background_color=background_color,
            background_padding_ratio=args.background_padding_ratio,
            experiment_mode=args.experiment_mode,
            ignore_wait=args.quality_ignore_wait,
            case_limit=args.quality_case_limit,
        )

    json_path, md_path = save_reports(
        report_dir=args.report_dir,
        report_prefix=args.report_prefix,
        config={
            "model": args.model,
            "experiment_mode": args.experiment_mode,
            "normalizing_ranges": list(normalizing_ranges),
            "prompt_variants": prompt_variants,
            "size_modes": size_modes,
            "background_modes": background_modes,
            "ui_filter_modes": ui_filter_modes,
            "color_policies": color_policies,
            "encode_modes": encode_modes,
            "quality_dataset": args.quality_dataset,
        },
        latency_results=latency_results,
        quality_results=quality_results,
    )
    console.print(f"[green]report saved[/green] json={json_path} md={md_path}")


if __name__ == "__main__":
    main()
