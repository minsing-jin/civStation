import argparse
import json
import os
import time
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

ACTION_SCHEMA = {
    "type": "array",
    "minItems": 1,
    "items": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["click", "press", "drag", "type"]},
            "x": {"type": "integer", "minimum": 0, "maximum": 1000},
            "y": {"type": "integer", "minimum": 0, "maximum": 1000},
            "end_x": {"type": "integer", "minimum": 0, "maximum": 1000},
            "end_y": {"type": "integer", "minimum": 0, "maximum": 1000},
            "button": {"type": "string", "enum": ["left", "right"]},
            "key": {"type": "string"},
            "text": {"type": "string"},
            "reasoning": {"type": "string"},
        },
        "required": ["action", "x", "y", "end_x", "end_y", "button", "key", "text", "reasoning"],
        "additionalProperties": False,
    },
}

LONG_PROMPT = f"""
{MULTI_ACTION_SEQUENCE_JSON_FORMAT_INSTRUCTION.format(normalizing_range=1000)}

{POLICY_PROMPT}

추가 지시:
- 반드시 실행 가능한 action sequence만 반환하라.
- 좌표는 모두 0~1000 정규화 좌표를 사용하라.
- 실행 불가한 추측은 하지 말고, 보이는 UI 근거를 reasoning에 짧게 남겨라.
""".strip()

CONCISE_PROMPT = """
문명6 정책 화면에 대해 다음 action space로 실행 계획(JSON 배열)을 반환하라.
- action: click | press | drag | type
- 좌표는 0~1000 정규화
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
    - compressed: 해상도 축소 + JPEG 재인코딩(품질 저하)
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
) -> tuple[list[dict], float]:
    start = time.perf_counter()
    resp = client.models.generate_content(
        model=model,
        contents=[prompt, image],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=ACTION_SCHEMA,
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
        base_image = PIL.Image.open(image_path)

    prompt_variants = {
        "long_prompt": LONG_PROMPT,
        "concise_prompt": CONCISE_PROMPT,
    }
    image_variants = {
        "raw": normalize_screenshot_for_primitive(base_image, "raw"),
        "compressed": normalize_screenshot_for_primitive(base_image, "compressed"),
    }

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
            for image_name, image in image_variants.items():
                console.print(f"[bold]실험 시작[/bold] variant={prompt_name}+{image_name}, runs={runs}")
                times: list[float] = []
                last_payload: list[dict] = []
                for _ in range(runs):
                    run_idx = len(times) + 1
                    console.print(
                        f"[dim]  -> run {run_idx}/{runs} | variant={prompt_name}+{image_name} | calling model...[/dim]"
                    )
                    payload, elapsed = run_single(client, model, image, prompt)
                    last_payload = payload
                    times.append(elapsed)
                    done_calls += 1
                    progress.update(task_id, completed=done_calls)
                    console.print(f"[dim]     done run {run_idx}/{runs} | elapsed={elapsed:.3f}s[/dim]")

                avg_time = sum(times) / len(times)
                first_action = last_payload[0]["action"] if last_payload else "n/a"
                result = {
                    "prompt_variant": prompt_name,
                    "image_variant": image_name,
                    "runs": runs,
                    "avg_latency_sec": round(avg_time, 3),
                    "min_latency_sec": round(min(times), 3),
                    "max_latency_sec": round(max(times), 3),
                    "num_actions": len(last_payload),
                    "first_action": first_action,
                    "result": last_payload,
                }
                results.append(result)
                console.print(
                    f"[green]실험 완료[/green] variant={prompt_name}+{image_name} "
                    f"| avg={avg_time:.3f}s min={min(times):.3f}s max={max(times):.3f}s"
                )

    show_experiment_summary(results)
    show_action_preview(results)

    return results


def show_experiment_summary(results: list[dict]) -> None:
    console.rule("[bold cyan]VLM Policy Action Benchmark")
    console.print(
        "[bold]비교군[/bold]: "
        "2개 프롬프트(long_policy_action_prompt, concise_policy_action_prompt) "
        "x 2개 스크린샷(normalization raw, compressed)"
    )

    table = Table(title="Latency / Action Summary", show_lines=False)
    table.add_column("Prompt", style="magenta")
    table.add_column("Image", style="cyan")
    table.add_column("Avg(s)", justify="right")
    table.add_column("Min(s)", justify="right")
    table.add_column("Max(s)", justify="right")
    table.add_column("Actions", justify="right")
    table.add_column("First", style="green")

    for row in results:
        table.add_row(
            row["prompt_variant"],
            row["image_variant"],
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
        variant = f"{row['prompt_variant']} + {row['image_variant']}"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Path to screenshot image (옵션)")
    parser.add_argument("--model", default="gemini-3-flash-preview", help="Gemini model")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per variant")
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

    benchmark(
        image_path=args.image,
        model=args.model,
        runs=args.runs,
        capture_live=args.capture_live,
        capture_countdown_sec=args.capture_countdown_sec,
        save_capture_path=args.save_capture_path,
    )


if __name__ == "__main__":
    main()
