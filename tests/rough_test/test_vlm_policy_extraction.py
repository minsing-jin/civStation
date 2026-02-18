import json
import os

import PIL.Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# 원하는 출력 스키마(JSON Schema)
POLICY_SCHEMA = {
    "type": "object",
    "properties": {
        "military": {"type": "array", "items": {"type": "string"}},
        "economic": {"type": "array", "items": {"type": "string"}},
        "diplomatic": {"type": "array", "items": {"type": "string"}},
        "wildcard": {"type": "array", "items": {"type": "string"}},
        "unknown": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["military", "economic", "diplomatic", "wildcard", "unknown"],
    "additionalProperties": False,
}

FEW_SHOT = r"""
출력은 반드시 아래 JSON 스키마를 따르는 "순수 JSON" 이어야 한다(코드블록/설명/추가 텍스트 금지).

[예시 1]
입력(가상의 카드들): 군사=징병, 경제=도시 계획, 외교=카리스마 리더, 와일드=신비한 유산
출력:
{"military":["징병"],"economic":["도시 계획"],"diplomatic":["카리스마 리더"],"wildcard":["신비한 유산"],"unknown":[]}

[예시 2]
입력(가상의 카드들): 경제=해양 산업, 경제=대상 숙소, 외교=조사
출력:
{"military":[],"economic":["해양 산업","대상 숙소"],"diplomatic":["조사"],"wildcard":[],"unknown":[]}

이제 실제 이미지에서 정책 카드 "제목"만 추출해서 같은 형식으로 출력하라.
""".strip()


def extract_civ6_policies(image_path: str) -> dict:
    client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))

    img = PIL.Image.open(image_path)

    prompt = f"""
{FEW_SHOT}

작업:
- 화면 하단 리스트에 있는 모든 정책 카드의 "제목"만 추출
- 상단 탭/버튼(예: 플레이어 정부/정책 변경) 텍스트는 제외
- 카드 설명문은 제외
- 분류 기준: 군사(빨강/갈색), 경제(노랑/금색), 외교(보라색), 와일드(초록색)
- 분류 불가/애매하면 unknown에 넣기
""".strip()

    resp = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[prompt, img],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=POLICY_SCHEMA,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    # 무조건 JSON으로 파싱(실패 시 예외로 잡아서 재시도 로직을 넣어도 됨)
    return json.loads(resp.text)


# 실행
data = extract_civ6_policies("screenshot.png")
print(data)
