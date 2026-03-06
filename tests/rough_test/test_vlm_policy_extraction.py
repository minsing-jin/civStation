import argparse
import json
import os
import time
from dataclasses import dataclass
from io import BytesIO

import PIL.Image
import pyautogui
from dotenv import load_dotenv
from google import genai
from google.genai import types
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from computer_use_test.utils.prompts.primitive_prompt import (
    MULTI_ACTION_SEQUENCE_JSON_FORMAT_INSTRUCTION,
    POLICY_PROMPT,
)

load_dotenv()

console = Console()

ALLOWED_PROMPT_VARIANTS = {"long_prompt", "concise_prompt"}
ALLOWED_SIZE_MODES = {"raw", "compressed", "downscale_restore", "compressed_downscale_restore"}
ALLOWED_BACKGROUND_MODES = {"none", "color"}


@dataclass(frozen=True)
class ImageVariant:
    name: str
    size_mode: str
    background_mode: str
    image: PIL.Image.Image
    # (left, top, width, height): 원본 UI가 variant 이미지 내에서 차지하는 영역
    content_box: tuple[int, int, int, int]


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


def build_prompt(prompt_variant: str, normalizing_range: int) -> str:
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
    if countdown_sec > 0:
        print(f"[INFO] {countdown_sec}초 후 현재 화면을 캡처합니다. 문명6 화면으로 전환하세요.")
        time.sleep(countdown_sec)

    screenshot = pyautogui.screenshot()
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


def benchmark(
    image_path: str | None,
    model: str,
    runs: int = 1,
    capture_live: bool = True,
    capture_countdown_sec: int = 3,
    save_capture_path: str | None = None,
    normalizing_range: int = 1000,
    size_modes: tuple[str, ...] = ("raw", "compressed", "downscale_restore", "compressed_downscale_restore"),
    background_modes: tuple[str, ...] = ("none", "color"),
    downscale_long_edge: int = 960,
    background_color: tuple[int, int, int] = (32, 32, 32),
    background_padding_ratio: float = 0.12,
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

    prompt_variants = {
        "long_prompt": build_long_prompt(normalizing_range),
        "concise_prompt": build_concise_prompt(normalizing_range),
    }
    action_schema = build_action_schema(normalizing_range)

    image_variants: list[ImageVariant] = []
    for size_mode in size_modes:
        for background_mode in background_modes:
            image_variants.append(
                build_image_variant(
                    base_image=base_image,
                    size_mode=size_mode,
                    background_mode=background_mode,
                    downscale_long_edge=downscale_long_edge,
                    background_color=background_color,
                    background_padding_ratio=background_padding_ratio,
                )
            )

    results: list[dict] = []

    total_calls = len(prompt_variants) * len(image_variants) * runs
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

        for prompt_name, prompt in prompt_variants.items():
            for image_variant in image_variants:
                variant_tag = f"{prompt_name}+{image_variant.name}"
                console.print(f"[bold]실험 시작[/bold] variant={variant_tag}, runs={runs}")
                times: list[float] = []
                last_payload_raw: list[dict] = []
                last_payload_restored: list[dict] = []
                for _ in range(runs):
                    run_idx = len(times) + 1
                    console.print(f"[dim]  -> run {run_idx}/{runs} | variant={variant_tag} | calling model...[/dim]")
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
                    times.append(elapsed)
                    done_calls += 1
                    progress.update(task_id, completed=done_calls)
                    console.print(f"[dim]     done run {run_idx}/{runs} | elapsed={elapsed:.3f}s[/dim]")

                avg_time = sum(times) / len(times)
                first_action = last_payload_restored[0]["action"] if last_payload_restored else "n/a"
                result = {
                    "prompt_variant": prompt_name,
                    "image_variant": image_variant.name,
                    "size_variant": image_variant.size_mode,
                    "background_variant": image_variant.background_mode,
                    "variant_image_size": f"{image_variant.image.size[0]}x{image_variant.image.size[1]}",
                    "content_box": image_variant.content_box,
                    "normalizing_range": normalizing_range,
                    "runs": runs,
                    "avg_latency_sec": round(avg_time, 3),
                    "min_latency_sec": round(min(times), 3),
                    "max_latency_sec": round(max(times), 3),
                    "num_actions": len(last_payload_restored),
                    "first_action": first_action,
                    "result": last_payload_restored,
                    "result_raw": last_payload_raw,
                }
                results.append(result)
                console.print(
                    f"[green]실험 완료[/green] variant={variant_tag} "
                    f"| avg={avg_time:.3f}s min={min(times):.3f}s max={max(times):.3f}s"
                )

    show_experiment_summary(results)
    show_action_preview(results)

    return results


def show_experiment_summary(results: list[dict]) -> None:
    console.rule("[bold cyan]VLM Policy Action Benchmark")
    prompt_set = sorted({r["prompt_variant"] for r in results})
    image_set = sorted({r["image_variant"] for r in results})
    console.print(f"[bold]비교군[/bold]: {len(prompt_set)}개 prompt x {len(image_set)}개 image variant")

    table = Table(title="Latency / Action Summary", show_lines=False)
    table.add_column("Prompt", style="magenta")
    table.add_column("Image", style="cyan")
    table.add_column("Size", style="yellow")
    table.add_column("BG", style="blue")
    table.add_column("ImgSize", style="white")
    table.add_column("Avg(s)", justify="right")
    table.add_column("Min(s)", justify="right")
    table.add_column("Max(s)", justify="right")
    table.add_column("Actions", justify="right")
    table.add_column("First", style="green")

    for row in results:
        table.add_row(
            row["prompt_variant"],
            row["image_variant"],
            row["size_variant"],
            row["background_variant"],
            row["variant_image_size"],
            f"{row['avg_latency_sec']:.3f}",
            f"{row['min_latency_sec']:.3f}",
            f"{row['max_latency_sec']:.3f}",
            str(row["num_actions"]),
            row["first_action"],
        )
    console.print(table)


def show_action_preview(results: list[dict]) -> None:
    preview = Table(title="Action Preview (readable)", show_lines=True)
    preview.add_column("Variant", style="yellow")
    preview.add_column("Step", justify="right")
    preview.add_column("Action", style="cyan")
    preview.add_column("Coords", overflow="fold")
    preview.add_column("Button/Key/Text", overflow="fold")
    preview.add_column("Reasoning", overflow="fold")

    for row in results:
        variant = (
            f"{row['prompt_variant']} + {row['image_variant']}"
            f" (size={row['size_variant']}, bg={row['background_variant']})"
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
    screen_w, screen_h = pyautogui.size()

    for idx, action in enumerate(actions, start=1):
        action_type = str(action.get("action", "")).strip()
        console.print(f"[bold green]실행[/bold green] step={idx} action={action_type}")

        if action_type == "click":
            x = norm_to_real(action.get("x", 0), screen_w, normalizing_range)
            y = norm_to_real(action.get("y", 0), screen_h, normalizing_range)
            button = str(action.get("button", "left") or "left")
            pyautogui.moveTo(x, y, duration=max(0.0, move_duration_sec))
            pyautogui.click(button=button)

        elif action_type == "drag":
            start_x = norm_to_real(action.get("x", 0), screen_w, normalizing_range)
            start_y = norm_to_real(action.get("y", 0), screen_h, normalizing_range)
            end_x = norm_to_real(action.get("end_x", 0), screen_w, normalizing_range)
            end_y = norm_to_real(action.get("end_y", 0), screen_h, normalizing_range)
            button = str(action.get("button", "left") or "left")
            console.print(f"[cyan]drag[/cyan] start=({start_x}, {start_y}) end=({end_x}, {end_y}) button={button}")
            pyautogui.moveTo(start_x, start_y, duration=max(0.0, move_duration_sec))
            pyautogui.mouseDown(x=start_x, y=start_y, button=button)
            if drag_hold_sec > 0:
                time.sleep(drag_hold_sec)
            pyautogui.moveTo(end_x, end_y, duration=max(0.0, move_duration_sec))
            if drag_drop_hold_sec > 0:
                time.sleep(drag_drop_hold_sec)
            pyautogui.mouseUp(x=end_x, y=end_y, button=button)

        elif action_type == "press":
            key = str(action.get("key", "")).strip()
            if key and key.lower() != "none":
                pyautogui.press(key)
            else:
                console.print("[yellow]skip[/yellow] press action has empty key")

        elif action_type == "type":
            text = str(action.get("text", ""))
            if text and text.lower() != "none":
                pyautogui.write(text, interval=0.06)
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

    payload_raw, elapsed = run_single(
        client=client,
        model=model,
        image=image_variant.image,
        prompt=prompt,
        action_schema=action_schema,
    )
    payload_restored = restore_actions_to_base_norm(
        payload_raw,
        image_variant,
        normalizing_range=normalizing_range,
    )

    row = {
        "prompt_variant": prompt_variant,
        "image_variant": image_variant.name,
        "size_variant": image_variant.size_mode,
        "background_variant": image_variant.background_mode,
        "variant_image_size": f"{image_variant.image.size[0]}x{image_variant.image.size[1]}",
        "content_box": image_variant.content_box,
        "normalizing_range": normalizing_range,
        "runs": 1,
        "avg_latency_sec": round(elapsed, 3),
        "min_latency_sec": round(elapsed, 3),
        "max_latency_sec": round(elapsed, 3),
        "num_actions": len(payload_restored),
        "first_action": payload_restored[0]["action"] if payload_restored else "n/a",
        "result": payload_restored,
        "result_raw": payload_raw,
    }

    console.print(f"[green]추론 완료[/green] latency={elapsed:.3f}s")
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
    background_modes = parse_csv_modes(args.background_modes, ALLOWED_BACKGROUND_MODES, "--background-modes")
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

    live_grid_ranges = tuple(parse_csv_ints(args.live_grid_ranges, "--live-grid-ranges"))
    live_grid_prompt_variants = tuple(
        parse_csv_modes(args.live_grid_prompt_variants, ALLOWED_PROMPT_VARIANTS, "--live-grid-prompt-variants")
    )

    if args.live_test:
        run_live_civ6_test(
            model=args.model,
            normalizing_range=args.normalizing_range,
            prompt_variant=args.live_prompt_variant,
            size_mode=args.live_size_mode,
            background_mode=args.live_background_mode,
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

    benchmark(
        image_path=args.image,
        model=args.model,
        runs=args.runs,
        capture_live=args.capture_live,
        capture_countdown_sec=args.capture_countdown_sec,
        save_capture_path=args.save_capture_path,
        normalizing_range=args.normalizing_range,
        size_modes=tuple(size_modes),
        background_modes=tuple(background_modes),
        downscale_long_edge=args.downscale_long_edge,
        background_color=background_color,
        background_padding_ratio=args.background_padding_ratio,
    )


if __name__ == "__main__":
    main()
