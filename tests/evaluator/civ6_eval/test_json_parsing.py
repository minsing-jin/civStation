import json
import logging

import pytest

from computer_use_test.utils.llm_provider.base import BaseVLMProvider

# 테스트 시 로그 확인이 필요하다면 설정 (pytest -s 옵션으로 확인 가능)
logger = logging.getLogger(__name__)


def strip_markdown(text: str) -> str:
    """실제 BaseVLMProvider의 메서드를 활용"""
    return BaseVLMProvider._strip_markdown(text)


@pytest.mark.parametrize(
    "test_name, raw_response, expected_valid",
    [
        # --- Valid cases ---
        ("Valid plain JSON", '{"primitive": "unit_ops_primitive", "reasoning": "test"}', True),
        ("Valid with code fence", '```json\n{"primitive": "unit_ops_primitive", "reasoning": "test"}\n```', True),
        ("Valid with generic fence", '```\n{"primitive": "unit_ops_primitive", "reasoning": "test"}\n```', True),
        ("Valid with whitespace", '  \n{\n  "primitive": "unit_ops_primitive",\n  "reasoning": "test"\n}\n  ', True),
        ("Valid action", '{"action": "click", "x": 500, "y": 300, "reasoning": "click here"}', True),
        ("Code fence no closing", '```json\n{"primitive": "unit_ops_primitive", "reasoning": "test"}', True),
        ("Multiple code fences", '```json\n{"primitive": "unit_ops_primitive", "reasoning": "test"}\n```\n```', True),
        ("Multiline reasoning", '{"primitive": "unit_ops_primitive", "reasoning": "This is\\na test"}', True),
        # --- Invalid cases ---
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
def test_json_parsing(test_name, raw_response, expected_valid):
    """JSON 파싱 테스트: 성공해야 하는 케이스와 실패해야 하는 케이스를 검증합니다."""

    stripped = strip_markdown(raw_response)

    if expected_valid:
        # 성공할 것으로 예상되는 경우: 에러 없이 파싱되어야 함
        try:
            data = json.loads(stripped)
            assert isinstance(data, dict), f"[{test_name}] 결과값이 dict가 아닙니다."
        except json.JSONDecodeError as e:
            pytest.fail(f"[{test_name}] 파싱 성공을 예상했으나 실패했습니다: {e}")
    else:
        # 실패할 것으로 예상되는 경우: JSONDecodeError가 발생해야 함
        with pytest.raises(json.JSONDecodeError):
            json.loads(stripped)
