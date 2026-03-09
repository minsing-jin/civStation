"""Tests for dataset_loader.py JSONL loading."""

import json
import tempfile
from pathlib import Path

import pytest

from computer_use_test.evaluation.evaluator.action_eval.bbox_eval import (
    DatasetLoadError,
    GTClickAction,
    GTKeyPressAction,
    load_dataset,
    validate_dataset,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_DATASET = FIXTURES_DIR / "sample_bbox_dataset.jsonl"


class TestLoadDataset:
    def test_load_sample_fixture(self):
        cases = load_dataset(SAMPLE_DATASET)
        assert len(cases) == 3

        # First case
        assert cases[0].case_id == "civ6_science_001"
        assert len(cases[0].action_sets) == 2
        assert isinstance(cases[0].action_sets[0].actions[0], GTClickAction)
        assert isinstance(cases[0].action_sets[0].actions[1], GTKeyPressAction)

        # Second case - single key press
        assert cases[1].case_id == "civ6_popup_002"
        assert len(cases[1].action_sets) == 1

        # Third case - drag
        assert cases[2].case_id == "civ6_drag_003"

    def test_file_not_found(self):
        with pytest.raises(DatasetLoadError, match="not found"):
            load_dataset("/nonexistent/path.jsonl")

    def test_wrong_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("{}")
            f.flush()
            with pytest.raises(DatasetLoadError, match="Expected .jsonl"):
                load_dataset(f.name)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            f.write("\n\n")
            f.flush()
            with pytest.raises(DatasetLoadError, match="No valid cases"):
                load_dataset(f.name)

    def test_skip_invalid_lines(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            # Valid line
            valid = {
                "case_id": "test_001",
                "instruction": "Click",
                "screenshot_path": "test.png",
                "image_size": {"width": 1000, "height": 1000},
                "action_sets": [
                    {"actions": [{"type": "press", "keys": ["enter"]}]},
                ],
            }
            f.write(json.dumps(valid) + "\n")
            f.write("not valid json\n")
            f.write(json.dumps({"case_id": "bad"}) + "\n")  # missing fields
            f.flush()

            cases = load_dataset(f.name)
            assert len(cases) == 1
            assert cases[0].case_id == "test_001"

    def test_all_invalid_lines(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            f.write("bad json\n")
            f.write("also bad\n")
            f.flush()
            with pytest.raises(DatasetLoadError, match="No valid cases"):
                load_dataset(f.name)

    def test_blank_lines_skipped(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            valid = {
                "case_id": "test_001",
                "instruction": "Click",
                "screenshot_path": "test.png",
                "image_size": {"width": 1000, "height": 1000},
                "action_sets": [{"actions": [{"type": "press", "keys": ["enter"]}]}],
            }
            f.write("\n")
            f.write(json.dumps(valid) + "\n")
            f.write("\n")
            f.flush()

            cases = load_dataset(f.name)
            assert len(cases) == 1


class TestValidateDataset:
    def test_valid_fixture(self):
        count, errors = validate_dataset(SAMPLE_DATASET)
        assert count == 3
        assert errors == []

    def test_nonexistent_file(self):
        count, errors = validate_dataset("/nonexistent.jsonl")
        assert count == 0
        assert len(errors) == 1

    def test_mixed_valid_invalid(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            valid = {
                "case_id": "test_001",
                "instruction": "Click",
                "screenshot_path": "test.png",
                "image_size": {"width": 1000, "height": 1000},
                "action_sets": [{"actions": [{"type": "press", "keys": ["enter"]}]}],
            }
            f.write(json.dumps(valid) + "\n")
            f.write("bad\n")
            f.flush()

            count, errors = validate_dataset(f.name)
            assert count == 1
            assert len(errors) == 1
