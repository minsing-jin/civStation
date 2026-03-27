import json
import logging

import pytest

from civStation.utils.llm_provider.parser import strip_markdown

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "test_name, raw_response, expected_valid",
    [
        ("Valid plain JSON", '{"primitive": "unit_ops_primitive", "reasoning": "test"}', True),
        ("Valid with code fence", '```json\n{"primitive": "unit_ops_primitive", "reasoning": "test"}\n```', True),
        ("Valid with generic fence", '```\n{"primitive": "unit_ops_primitive", "reasoning": "test"}\n```', True),
        ("Valid with whitespace", '  \n{\n  "primitive": "unit_ops_primitive",\n  "reasoning": "test"\n}\n  ', True),
        ("Valid action", '{"action": "click", "x": 500, "y": 300, "reasoning": "click here"}', True),
        ("Code fence no closing", '```json\n{"primitive": "unit_ops_primitive", "reasoning": "test"}', True),
        ("Multiple code fences", '```json\n{"primitive": "unit_ops_primitive", "reasoning": "test"}\n```\n```', True),
        ("Multiline reasoning", '{"primitive": "unit_ops_primitive", "reasoning": "This is\\na test"}', True),
        ("Unterminated string", '{"primitive": "unit_ops_primitive", "reasoning": "test', False),
        ("Missing quote", '{"primitive": unit_ops_primitive, "reasoning": "test"}', False),
        ("Trailing comma", '{"primitive": "unit_ops_primitive", "reasoning": "test",}', False),
        ("Single quotes", "{'primitive': 'unit_ops_primitive', 'reasoning': 'test'}", False),
        (
            "Newline in reasoning (invalid)",
            '{\n  "primitive": "unit_ops_primitive",\n  "reasoning": "Line 1\nLine 2"\n}',
            False,
        ),
    ],
)
def test_strip_markdown_json_parsing(test_name, raw_response, expected_valid):
    stripped = strip_markdown(raw_response)

    if expected_valid:
        try:
            data = json.loads(stripped)
            assert isinstance(data, dict), f"[{test_name}] result was not a dict"
        except json.JSONDecodeError as exc:
            pytest.fail(f"[{test_name}] expected valid JSON but parsing failed: {exc}")
    else:
        with pytest.raises(json.JSONDecodeError):
            json.loads(stripped)
