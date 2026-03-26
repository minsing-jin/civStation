from __future__ import annotations

import importlib.util
import json
import tempfile
import time
from pathlib import Path

from civStation.evaluation.evaluator.action_eval.bbox_eval.scorer import aggregate_results
from civStation.utils.llm_provider import create_provider
from civStation.utils.llm_provider.parser import parse_to_agent_plan, strip_markdown

REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = REPO_ROOT / "tests/rough_test/test_vlm_policy_extraction.py"
RESULTS_DIR = PAPER_ROOT / "arxiv/results"
TRADEOFF_IMAGE = PAPER_ROOT / "arxiv/benchmarks/synthetic_ui/hero_screenshot.png"
QUALITY_DATASET = PAPER_ROOT / "arxiv/benchmarks/synthetic_ui/synthetic_bbox_dataset.jsonl"
PROVIDER_NAME = "gpt"
MODEL_NAME = "gpt-4o-mini"

PROMPT_TEMPLATE = """You are evaluating a game-like screenshot. Analyze the image and follow this instruction:

{instruction}

The image size is {width}x{height}. Use normalized coordinates in the range 0-{max_coord}.

Respond with JSON only (no markdown fences):
{{
  "reasoning": "your reasoning here",
  "actions": [
    {{"type": "click", "x": <int>, "y": <int>, "button": "left"}},
    {{"type": "press", "keys": ["enter"]}},
    {{"type": "drag", "start_x": <int>, "start_y": <int>, "end_x": <int>, "end_y": <int>}}
  ]
}}
"""


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("paper_tradeoff_benchmark", BENCHMARK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load benchmark module from {BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def action_to_raw_dict(action, reasoning: str) -> dict:
    action_type = getattr(action, "type", "")
    if action_type == "click":
        return {
            "action": "click",
            "x": int(action.x),
            "y": int(action.y),
            "end_x": 0,
            "end_y": 0,
            "button": getattr(action, "button", "left"),
            "key": "",
            "text": "",
            "reasoning": reasoning,
        }
    if action_type == "drag":
        return {
            "action": "drag",
            "x": int(action.start_x),
            "y": int(action.start_y),
            "end_x": int(action.end_x),
            "end_y": int(action.end_y),
            "button": getattr(action, "button", "left"),
            "key": "",
            "text": "",
            "reasoning": reasoning,
        }
    if action_type == "press":
        keys = getattr(action, "keys", [])
        return {
            "action": "press",
            "x": 0,
            "y": 0,
            "end_x": 0,
            "end_y": 0,
            "button": "left",
            "key": "+".join(keys),
            "text": "",
            "reasoning": reasoning,
        }
    if action_type == "wait":
        return {
            "action": "wait",
            "x": 0,
            "y": 0,
            "end_x": 0,
            "end_y": 0,
            "button": "left",
            "key": "",
            "text": "",
            "reasoning": reasoning,
        }
    raise ValueError(f"Unsupported action type: {action_type}")


def _coerce_raw_actions(payload, reasoning: str) -> list[dict]:
    if isinstance(payload, dict):
        if "actions" in payload:
            reasoning = str(payload.get("reasoning", reasoning) or reasoning)
            payload = payload["actions"]
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected response type: {type(payload)}")

    normalized: list[dict] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        raw = dict(item)
        for key in ("x", "y", "end_x", "end_y"):
            if key in raw and raw[key] not in (None, ""):
                raw[key] = int(round(float(raw[key])))
        if "button" not in raw:
            raw["button"] = "left"
        if "key" not in raw:
            raw["key"] = ""
        if "text" not in raw:
            raw["text"] = ""
        if "reasoning" not in raw:
            raw["reasoning"] = reasoning
        normalized.append(raw)
    return normalized


def build_eval_prompt(instruction: str, width: int, height: int, max_coord: int) -> str:
    return PROMPT_TEMPLATE.format(
        instruction=instruction,
        width=width,
        height=height,
        max_coord=max_coord,
    )


def call_provider_with_pil(provider, pil_image, prompt: str, primitive_name: str = "eval") -> tuple[list[dict], float]:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
        pil_image.save(tmp.name, format="PNG")
        start = time.perf_counter()
        response = provider.call_vlm(prompt=prompt, image_path=tmp.name, temperature=0.0, max_tokens=2048)
        elapsed = time.perf_counter() - start
    try:
        plan = parse_to_agent_plan(response.content, primitive_name)
        payload = [action_to_raw_dict(action, plan.reasoning) for action in plan.actions]
    except Exception:
        payload = _coerce_raw_actions(json.loads(strip_markdown(response.content)), reasoning="")
    return payload, elapsed


def run_latency_grid(benchmark) -> list[dict]:
    base_image = benchmark.PIL.Image.open(TRADEOFF_IMAGE).convert("RGB")
    provider = create_provider(PROVIDER_NAME, model=MODEL_NAME)
    preprocess_specs = [
        benchmark.PreprocessSpec("baseline", "none", "preserve", "none"),
        benchmark.PreprocessSpec("contrast_jpeg", "ui_contrast", "adaptive_gray", "jpeg_like"),
    ]
    prepared_variants = benchmark.build_prepared_variants(
        base_image=base_image,
        size_modes=("raw", "compressed", "downscale_restore"),
        background_modes=("none",),
        preprocess_specs=preprocess_specs,
        downscale_long_edge=960,
        background_color=(32, 32, 32),
        background_padding_ratio=0.12,
    )

    results: list[dict] = []
    for normalizing_range in (250, 500):
        for prompt_variant in ("concise_prompt", "long_prompt"):
            for image_variant in prepared_variants:
                prompt = build_eval_prompt(
                    instruction=(
                        "Click the green Confirm Policies button if it is visible. "
                        "Otherwise choose the most actionable visible UI control."
                    ),
                    width=image_variant.image.size[0],
                    height=image_variant.image.size[1],
                    max_coord=normalizing_range,
                )
                payload_raw, elapsed = call_provider_with_pil(provider, image_variant.image, prompt)
                payload_restored = benchmark.restore_actions_to_base_norm(
                    payload_raw,
                    image_variant,
                    normalizing_range=normalizing_range,
                )
                first_action = payload_restored[0]["action"] if payload_restored else "none"
                results.append(
                    {
                        "benchmark_type": "latency_grid",
                        "provider": PROVIDER_NAME,
                        "model": MODEL_NAME,
                        "prompt_variant": prompt_variant,
                        "image_variant": image_variant.name,
                        "size_variant": image_variant.size_mode,
                        "background_variant": image_variant.background_mode,
                        "ui_filter_mode": image_variant.ui_filter_mode,
                        "color_policy": image_variant.color_policy,
                        "encode_mode": image_variant.encode_mode,
                        "payload_format": image_variant.payload_format,
                        "payload_bytes": image_variant.payload_bytes,
                        "variant_image_size": f"{image_variant.image.size[0]}x{image_variant.image.size[1]}",
                        "normalizing_range": normalizing_range,
                        "preprocess_latency_ms": round(image_variant.preprocess_latency_ms, 3),
                        "encode_latency_ms": round(image_variant.encode_latency_ms, 3),
                        "avg_inference_latency_sec": round(elapsed, 3),
                        "avg_latency_sec": round(elapsed + image_variant.preprocess_latency_ms / 1000, 3),
                        "num_actions": len(payload_restored),
                        "first_action": first_action,
                    }
                )
    return results


def run_quality_grid(benchmark) -> list[dict]:
    provider = create_provider(PROVIDER_NAME, model=MODEL_NAME)
    cases = benchmark.load_dataset(str(QUALITY_DATASET))
    preprocess_specs = [
        benchmark.PreprocessSpec("baseline", "none", "preserve", "none"),
        benchmark.PreprocessSpec("contrast_jpeg", "ui_contrast", "adaptive_gray", "jpeg_like"),
    ]
    rows: list[dict] = []

    for normalizing_range in (250, 500):
        for prompt_variant in ("concise_prompt",):
            for size_mode in ("compressed", "downscale_restore"):
                for preprocess_spec in preprocess_specs:
                    case_results = []
                    latencies = []
                    payload_sizes = []
                    for case in cases:
                        screenshot_path = benchmark.resolve_dataset_screenshot_path(
                            str(QUALITY_DATASET), case.screenshot_path
                        )
                        base_image = benchmark.PIL.Image.open(screenshot_path).convert("RGB")
                        base_variant = benchmark.build_image_variant(
                            base_image=base_image,
                            size_mode=size_mode,
                            background_mode="none",
                            downscale_long_edge=960,
                            background_color=(32, 32, 32),
                            background_padding_ratio=0.12,
                        )
                        prepared_variant = benchmark.prepare_image_variant(base_variant, preprocess_spec)
                        prompt = build_eval_prompt(
                            instruction=case.instruction,
                            width=prepared_variant.image.size[0],
                            height=prepared_variant.image.size[1],
                            max_coord=normalizing_range,
                        )
                        payload_raw, elapsed = call_provider_with_pil(provider, prepared_variant.image, prompt)
                        payload_restored = benchmark.restore_actions_to_base_norm(
                            payload_raw, prepared_variant, normalizing_range=normalizing_range
                        )
                        case_results.append(
                            benchmark.score_actions_against_case(case=case, actions=payload_restored, ignore_wait=True)
                        )
                        latencies.append(elapsed + prepared_variant.preprocess_latency_ms / 1000)
                        payload_sizes.append(prepared_variant.payload_bytes)

                    aggregate = aggregate_results(case_results)
                    rows.append(
                        {
                            "benchmark_type": "quality_dataset",
                            "provider": PROVIDER_NAME,
                            "model": MODEL_NAME,
                            "prompt_variant": prompt_variant,
                            "image_variant": f"{size_mode}+bg_none+{preprocess_spec.name}",
                            "size_variant": size_mode,
                            "background_variant": "none",
                            "ui_filter_mode": preprocess_spec.ui_filter_mode,
                            "color_policy": preprocess_spec.color_policy,
                            "encode_mode": preprocess_spec.encode_mode,
                            "payload_bytes": int(round(sum(payload_sizes) / len(payload_sizes)))
                            if payload_sizes
                            else 0,
                            "normalizing_range": normalizing_range,
                            "avg_latency_sec": round(sum(latencies) / len(latencies), 3),
                            "strict_success_rate": round(aggregate.strict_success_rate, 4),
                            "avg_step_accuracy": round(aggregate.avg_step_accuracy, 4),
                            "avg_prefix_len": round(aggregate.avg_prefix_len, 4),
                            "error_count": aggregate.error_count,
                            "timeout_count": aggregate.timeout_count,
                        }
                    )

    # quality gate relative to no-preprocess baseline
    baseline_rows = {}
    for row in rows:
        if row["ui_filter_mode"] == "none" and row["color_policy"] == "preserve" and row["encode_mode"] == "none":
            baseline_rows[(row["normalizing_range"], row["prompt_variant"], row["size_variant"])] = row
    for row in rows:
        base = baseline_rows.get((row["normalizing_range"], row["prompt_variant"], row["size_variant"]))
        if base is None:
            row["quality_gate_pass"] = True
        else:
            row["quality_gate_pass"] = row["avg_step_accuracy"] >= max(0.0, base["avg_step_accuracy"] - 0.05) and row[
                "strict_success_rate"
            ] >= max(0.0, base["strict_success_rate"] - 0.05)
    return rows


def build_markdown(payload: dict) -> str:
    latency = payload["latency_results"]
    quality = payload["quality_results"]
    lines = [
        "# Trade-off benchmark",
        "",
        f"- Provider: `{payload['config']['provider']}`",
        f"- Model: `{payload['config']['model']}`",
        f"- Dataset: `{payload['config']['quality_dataset']}`",
        "",
        "## Latency Leaders",
        "| range | prompt | image_variant | avg_latency_sec | payload_bytes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in sorted(latency, key=lambda item: item["avg_latency_sec"])[:8]:
        lines.append(
            f"| {row['normalizing_range']} | {row['prompt_variant']} | {row['image_variant']} | "
            f"{row['avg_latency_sec']:.3f} | {row['payload_bytes']} |"
        )
    lines.extend(
        [
            "",
            "## Quality Leaders",
            "| range | prompt | image_variant | avg_latency_sec | strict_success_rate | avg_step_accuracy | gate |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in sorted(
        quality,
        key=lambda item: (
            not item.get("quality_gate_pass", False),
            -item["avg_step_accuracy"],
            item["avg_latency_sec"],
        ),
    )[:8]:
        lines.append(
            f"| {row['normalizing_range']} | {row['prompt_variant']} | {row['image_variant']} | "
            f"{row['avg_latency_sec']:.3f} | {row['strict_success_rate']:.3f} | "
            f"{row['avg_step_accuracy']:.3f} | {'pass' if row.get('quality_gate_pass', False) else 'fail'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    benchmark = load_benchmark_module()

    latency_results = run_latency_grid(benchmark)
    quality_results = run_quality_grid(benchmark)

    payload = {
        "config": {
            "provider": PROVIDER_NAME,
            "model": MODEL_NAME,
            "quality_dataset": str(QUALITY_DATASET.relative_to(REPO_ROOT)),
        },
        "latency_results": latency_results,
        "quality_results": quality_results,
    }
    json_path = RESULTS_DIR / "gpt_tradeoff_benchmark.json"
    md_path = RESULTS_DIR / "gpt_tradeoff_benchmark.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(build_markdown(payload), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
