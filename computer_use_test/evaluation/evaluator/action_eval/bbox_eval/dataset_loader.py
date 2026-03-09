"""
Load and validate JSONL datasets for bbox-based evaluation.

Each line in a JSONL file is one JSON object representing a DatasetCase.

Example:
    >>> from computer_use_test.evaluation.evaluator.action_eval.bbox_eval import load_dataset
    >>> cases = load_dataset("path/to/dataset.jsonl")
    >>> print(f"Loaded {len(cases)} cases")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from .schema import DatasetCase

logger = logging.getLogger(__name__)


class DatasetLoadError(Exception):
    """Raised when a dataset cannot be loaded or contains invalid entries."""


def load_dataset(path: str | Path) -> list[DatasetCase]:
    """
    Load a JSONL dataset file and return validated DatasetCase objects.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of validated DatasetCase objects.

    Raises:
        DatasetLoadError: If the file cannot be read or contains no valid cases.
    """
    path = Path(path)
    if not path.exists():
        raise DatasetLoadError(f"Dataset file not found: {path}")
    if not path.suffix == ".jsonl":
        raise DatasetLoadError(f"Expected .jsonl file, got: {path.suffix}")

    cases: list[DatasetCase] = []
    errors: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                case = DatasetCase.model_validate(data)
                cases.append(case)
            except json.JSONDecodeError as e:
                msg = f"Line {line_num}: invalid JSON: {e}"
                errors.append(msg)
                logger.warning(msg)
            except ValidationError as e:
                msg = f"Line {line_num}: validation error: {e}"
                errors.append(msg)
                logger.warning(msg)

    if not cases:
        raise DatasetLoadError(f"No valid cases found in {path}. Errors: {errors}")

    if errors:
        logger.warning(f"Loaded {len(cases)} cases with {len(errors)} skipped lines from {path}")
    else:
        logger.info(f"Loaded {len(cases)} cases from {path}")

    return cases


def validate_dataset(path: str | Path) -> tuple[int, list[str]]:
    """
    Validate a JSONL dataset without loading all cases into memory.

    Returns:
        Tuple of (valid_count, error_messages).
    """
    path = Path(path)
    if not path.exists():
        return 0, [f"File not found: {path}"]

    valid = 0
    errors: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                DatasetCase.model_validate(data)
                valid += 1
            except (json.JSONDecodeError, ValidationError) as e:
                errors.append(f"Line {line_num}: {e}")

    return valid, errors
