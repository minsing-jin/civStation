from __future__ import annotations

import importlib.util
from pathlib import Path

import PIL.Image

from civStation.evaluation.evaluator.action_eval.bbox_eval.schema import DatasetCase
from civStation.utils.ui_benchmarking import (
    PreprocessSpec,
    build_preprocess_specs,
    convert_actions_for_bbox_eval,
    prepare_benchmark_image,
    score_actions_against_case,
)

BENCHMARK_PATH = Path(__file__).with_name("test_vlm_policy_extraction.py")
BENCHMARK_SPEC = importlib.util.spec_from_file_location("rough_benchmark_mod", BENCHMARK_PATH)
assert BENCHMARK_SPEC is not None
assert BENCHMARK_SPEC.loader is not None
benchmark_mod = importlib.util.module_from_spec(BENCHMARK_SPEC)
BENCHMARK_SPEC.loader.exec_module(benchmark_mod)


def test_build_preprocess_specs_staged_contains_baseline() -> None:
    specs = build_preprocess_specs(
        ui_filter_modes=["none", "ui_contrast", "ui_quantized", "ui_bg_blur", "ui_bg_blur_contrast"],
        color_policies=["preserve", "grayscale", "adaptive_gray"],
        encode_modes=["none", "jpeg_like", "webp_like", "avif_like_if_supported"],
        experiment_mode="staged",
    )

    assert specs
    assert specs[0].name == "baseline"
    assert any(spec.encode_mode == "webp_like" for spec in specs)


def test_prepare_benchmark_image_records_payload_metrics() -> None:
    image = PIL.Image.new("RGB", (96, 64), color=(220, 50, 50))
    spec = PreprocessSpec(
        name="contrast_webp",
        ui_filter_mode="ui_contrast",
        color_policy="adaptive_gray",
        encode_mode="webp_like",
    )

    prepared = prepare_benchmark_image(image, spec)

    assert prepared.image.size == (96, 64)
    assert prepared.payload_bytes > 0
    assert prepared.payload_format == "webp"
    assert prepared.preprocess_latency_ms >= prepared.encode_latency_ms >= 0


def test_convert_actions_for_bbox_eval_maps_supported_actions() -> None:
    actions = convert_actions_for_bbox_eval(
        [
            {"action": "click", "x": 111, "y": 222, "button": "left", "reasoning": "click target"},
            {"action": "drag", "x": 10, "y": 20, "end_x": 30, "end_y": 40, "button": "left"},
            {"action": "press", "key": "enter"},
        ]
    )

    assert actions[0].type == "click"
    assert actions[1].type == "drag"
    assert actions[2].type == "press"


def test_score_actions_against_case_accepts_matching_click() -> None:
    case = DatasetCase.model_validate(
        {
            "case_id": "case_click",
            "instruction": "Click confirm",
            "screenshot_path": str(Path("screenshots/test.png")),
            "image_size": {"width": 1000, "height": 1000},
            "action_sets": [
                {
                    "actions": [
                        {
                            "type": "click",
                            "target_bbox": {"x_min": 100, "y_min": 100, "x_max": 140, "y_max": 140},
                            "button": "left",
                        }
                    ]
                }
            ],
            "metadata": {},
        }
    )

    result = score_actions_against_case(
        case,
        [{"action": "click", "x": 120, "y": 120, "button": "left", "reasoning": "hit"}],
    )

    assert result.error is None
    assert result.best_sequence is not None
    assert result.best_sequence.strict_success is True


def test_benchmark_grid_runs_with_stubbed_model(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "screen.png"
    PIL.Image.new("RGB", (160, 100), color=(30, 30, 30)).save(image_path)

    monkeypatch.setenv("GENAI_API_KEY", "dummy-key")
    monkeypatch.setattr(benchmark_mod.genai, "Client", lambda api_key: object())

    def fake_run_single(client, model, image, prompt, action_schema):
        return [
            {
                "action": "click",
                "x": 120,
                "y": 130,
                "end_x": 0,
                "end_y": 0,
                "button": "left",
                "key": "",
                "text": "",
                "reasoning": "stub action",
            }
        ], 0.25

    monkeypatch.setattr(benchmark_mod, "run_single", fake_run_single)

    results = benchmark_mod.benchmark_grid(
        image_path=str(image_path),
        model="stub-model",
        runs=1,
        capture_live=False,
        normalizing_ranges=(250,),
        prompt_variants=("concise_prompt",),
        size_modes=("compressed",),
        background_modes=("none",),
        preprocess_specs=[PreprocessSpec("baseline", "none", "preserve", "none")],
        downscale_long_edge=960,
        background_color=(32, 32, 32),
        background_padding_ratio=0.12,
        experiment_mode="staged",
    )

    assert len(results) == 1
    assert results[0]["avg_latency_sec"] >= 0.25
    assert results[0]["ui_filter_mode"] == "none"


def test_quality_benchmark_runs_with_stubbed_model(tmp_path, monkeypatch) -> None:
    screenshots_dir = tmp_path / "screenshots"
    screenshots_dir.mkdir()
    image_path = screenshots_dir / "case.png"
    PIL.Image.new("RGB", (160, 100), color=(255, 255, 255)).save(image_path)

    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        (
            '{"case_id":"case1","instruction":"Click confirm",'
            '"screenshot_path":"screenshots/case.png","image_size":{"width":1000,"height":1000},'
            '"action_sets":[{"actions":[{"type":"click","target_bbox":{"x_min":100,"y_min":100,"x_max":140,"y_max":140},"button":"left"}]}],'
            '"metadata":{}}\n'
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("GENAI_API_KEY", "dummy-key")
    monkeypatch.setattr(benchmark_mod.genai, "Client", lambda api_key: object())

    def fake_run_single(client, model, image, prompt, action_schema):
        return [
            {
                "action": "click",
                "x": 120,
                "y": 120,
                "end_x": 0,
                "end_y": 0,
                "button": "left",
                "key": "",
                "text": "",
                "reasoning": "bbox hit",
            }
        ], 0.2

    monkeypatch.setattr(benchmark_mod, "run_single", fake_run_single)

    results = benchmark_mod.benchmark_quality_dataset(
        dataset_path=str(dataset_path),
        model="stub-model",
        normalizing_ranges=(250,),
        prompt_variants=("concise_prompt",),
        size_modes=("compressed",),
        background_modes=("none",),
        preprocess_specs=[PreprocessSpec("baseline", "none", "preserve", "none")],
        downscale_long_edge=960,
        background_color=(32, 32, 32),
        background_padding_ratio=0.12,
        experiment_mode="staged",
        ignore_wait=True,
        case_limit=None,
    )

    assert len(results) == 1
    assert results[0]["strict_success_rate"] == 1.0
    assert results[0]["quality_gate_pass"] is True
