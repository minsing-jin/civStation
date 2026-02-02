# ==============================================================================
from PIL import Image
from io import BytesIO
import base64
import pyautogui

def capture_screen_pil(max_size=2048):
    """화면을 캡처하고 리사이징된 PIL 이미지를 반환"""
    screenshot = pyautogui.screenshot()

    # Retina 디스플레이 등 좌표계 보정을 위해 논리적 해상도 가져오기
    screen_w, screen_h = pyautogui.size()

    if screenshot.mode in ("RGBA", "P"):
        screenshot = screenshot.convert("RGB")

    screenshot.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    return screenshot, screen_w, screen_h


def image_to_base64(pil_image):
    """PIL 이미지를 Base64 문자열로 변환"""
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG", quality=80)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


from abc import ABC, abstractmethod
from dotenv import load_dotenv
import pyautogui
import json
import os

load_dotenv()

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import openai
except ImportError:
    openai = None


# ==============================================================================
# 2. VLM Provider 추상 클래스 (전략 패턴 적용)
# ==============================================================================
class VLMProvider(ABC):
    @abstractmethod
    def analyze(self, pil_image, instruction):
        """이미지와 명령을 받아 JSON 액션 플랜을 반환해야 함"""
        pass

    def _get_common_prompt(self, instruction, normalizing_range=1000):
        """모든 모델에게 공통으로 주입될 시스템 프롬프트"""
        return f"""
        You are a pro gamer AI agent.
        User Goal: '{instruction}'

        Analyze the screenshot and determine the next action.

        CRITICAL INSTRUCTION:
        1. Coordinates must be NORMALIZED (0-{normalizing_range}). (0,0)=Top-Left, ({normalizing_range},{normalizing_range})=Bottom-Right.
        2. Output MUST be a valid JSON object only.

        Action Types:
        - "click": Move mouse and click (requires x, y, button).
        - "press": Press a keyboard key (requires key). Use for shortcuts like 'b', 'm', 'enter', 'esc'.
        - "type": Type a string of text (requires text).

        JSON Format:
        {{
            "action": "click" or "press" or "type",
            "button": "left" or "right" (only for click),
            "key": "string" (e.g., "b", "m", "enter" - only for press),
            "text": "string" (only for type),
            "x": integer (0-{normalizing_range}, only for click),
            "y": integer (0-{normalizing_range}, only for click),
            "reasoning": "brief explanation"
        }}
        """


# ==============================================================================
# 3. 모델별 구현 (Gemini, Claude, GPT)
# ==============================================================================

class GeminiProvider(VLMProvider):
    def __init__(self, api_key):
        if not genai: raise ImportError("google-genai library not installed.")
        self.client = genai.Client(api_key=api_key)
        # 최신 모델 사용 (gemini-2.0-flash or gemini-1.5-pro)
        self.model_name = "gemini-3-flash-preview"

    def analyze(self, pil_image, instruction):
        prompt = self._get_common_prompt(instruction)
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[pil_image, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            return json.loads(response.text.strip())
        except Exception as e:
            print(f"[Gemini Error] {e}")
            return None


class ClaudeProvider(VLMProvider):
    def __init__(self, api_key):
        if not anthropic: raise ImportError("anthropic library not installed.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = "claude-sonnet-4-5-20250929"

    def analyze(self, pil_image, instruction):
        prompt = self._get_common_prompt(instruction)
        base64_img = image_to_base64(pil_image)

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_img}},
                        {"type": "text", "text": prompt}
                    ]
                }]
            )
            # JSON 파싱
            text = response.content[0].text
            if "```json" in text:
                text = text.replace("```json", "").replace("```", "")
            return json.loads(text.strip())
        except Exception as e:
            print(f"[Claude Error] {e}")
            return None


class GPTProvider(VLMProvider):
    def __init__(self, api_key):
        if not openai: raise ImportError("openai library not installed.")
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = "gpt-4o"

    def analyze(self, pil_image, instruction):
        prompt = self._get_common_prompt(instruction)
        base64_img = image_to_base64(pil_image)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=1024
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"[GPT Error] {e}")
            return None


# ==============================================================================
# 4. 실행 엔진 (Action Executor) - 수정됨
# ==============================================================================
def execute_action(action_plan, screen_w, screen_h, normalizing_range=1000):
    if not action_plan:
        return

    # [수정된 부분] action_plan이 리스트로 들어올 경우 첫 번째 요소 선택
    if isinstance(action_plan, list):
        if len(action_plan) == 0:
            print("[!] AI가 빈 리스트를 반환했습니다.")
            return
        # 리스트의 첫 번째 항목을 실제 plan으로 사용
        print(f"[*] 리스트 응답 감지됨. 첫 번째 항목 선택: {len(action_plan)}개 중 1번째")
        action_plan = action_plan[0]

    # 이제 action_plan은 확실히 dict이므로 .get() 사용 가능
    reasoning = action_plan.get("reasoning", "No reasoning")
    action_type = action_plan.get("action")
    print(f"\n[*] AI 결정: {action_type} | 이유: {reasoning}")

    # 1. 마우스 클릭 (좌표 변환 포함)
    if action_type == "click":
        norm_x = max(0, min(normalizing_range, action_plan.get("x", 0)))
        norm_y = max(0, min(normalizing_range, action_plan.get("y", 0)))

        # 0~1000 정규화 좌표를 실제 해상도로 변환
        real_x = int((norm_x / normalizing_range) * screen_w)
        real_y = int((norm_y / normalizing_range) * screen_h)

        button = action_plan.get("button", "left")

        print(f"[*] 이동 및 클릭: ({real_x}, {real_y}) - {button}")
        pyautogui.moveTo(real_x, real_y, duration=0.5)
        pyautogui.click(button=button)

    # 2. 키보드 입력 (단축키 등)
    elif action_type == "press":
        key = action_plan.get("key")
        if key:
            print(f"[*] 키보드 누름: {key}")
            pyautogui.press(key)

    # 3. 텍스트 타이핑
    elif action_type == "type":
        text = action_plan.get("text")
        if text:
            print(f"[*] 텍스트 입력: {text}")
            pyautogui.write(text, interval=0.1)


# ==============================================================================
# 5. 메인 함수
# ==============================================================================
def macro(instruction):
    # --- [설정] 여기서 사용할 모델을 선택하세요 ---
    # provider_type = "gemini"
    provider_type = "gemini"  # or "claude" or "gpt"

    # API 키 로드 (환경변수 또는 직접 입력)
    # api_key = os.getenv("OPENAI_API_KEY")
    # api_key = os.getenv("ANTHROPIC_API_KEY")
    api_key = os.getenv("GENAI_API_KEY")

    # Provider 초기화
    agent = None
    if provider_type == "gemini":
        agent = GeminiProvider(api_key)
    elif provider_type == "claude":
        agent = ClaudeProvider(api_key)
    elif provider_type == "gpt":
        agent = GPTProvider(api_key)

    print(f"[*] 에이전트 시작 ({provider_type})...")

    # 1. 화면 캡처
    pil_image, screen_w, screen_h = capture_screen_pil()
    print(f"[*] 화면 캡처 완료. 논리 해상도: {screen_w}x{screen_h}")

    # 2. AI 분석
    print("[*] 분석 요청 중...")
    action_plan = agent.analyze(pil_image, instruction)

    # 3. 행동 실행
    execute_action(action_plan, screen_w, screen_h)


if __name__ == "__main__":
    """TODO: 해상도 정리, Vlm 문제가 아니었음 - gemini가 제일 잘함. 클로드는 잘 못함
    1. 상위 레벨전략으로 옮겨서 선택해보기
    2. 유닛별로 판단해서 움직이기
    3. 개고트 기술 사람들이 많이 쓰는 엔진
    5턴 넘기기 및 작은 실험 코드 정리 + voice interface
    -> 이런 코딩은 일단 보여주고 코드 정리, 거기서 필요한거를 만들어서 쌓아가는것
    해상도 문제 공유
    사람들이 쓸만한 기술적 도전
    작은거부터 시작해서 쌓아올리는것이 더 코드에서 좋음. 처음부터 모든걸하려니 ㄴㄴ 문제 쪼개고 하기 접근
    다음 목표

    하나에 프롬프트에 박아서 vlm이 모두 판단하게 하는건 잘 못했음
    """

    # 사용자의 명령
    unit_instruction = (
        "너는 문명6 에이전트야. 유닛 정보를 보고 판단해."
        "1. 만약 '개척자'라면 'b'키를 눌러 도시를 건설해."
        "2. 만약 전투 유닛이라면 유닛의 위치를 제외한 화면의 하늘색 타일(이동 가능 영역) 중 좋은 곳을 골라 '우클릭'으로 이동해. "
    )

    """이 사이에 Enter"""

    pop_up_instruction = """
    화면에 팝업이 나타났나고, 예, 아니오 선택이 있다면 'enter'키를 눌러줘.
    화면에 팝업이 나타났는데, 예 아니오 선택이 없다면 'esc'키를 눌러줘.
    오른쪽 맨 아래에 다음턴이라는 글자와 화살표가 나타나면 'enter'키를 눌러줘.
    오른쪽 맨 아래에 연구선택이라는 글자와 파란색 플라스크가 나타나면 'enter'키를 눌러줘.
    오른쪽 맨 아래에 생산품목이라는 글자와 주황색 톱니바퀴가 나타나면 'enter'키를 눌러줘.
    """

    """이 사이에 Enter"""

    policy_instruction = """
    연구 선택이라는 왼쪽에 팝업이 나타났다면 시계 옆 남은 turn 수가 가장 적은 연구로 마우스를 움직여서 클릭해줘
    """

    city_instruction = """
    도시에서 생산할 수있는 생산품목이 오른쪽 팝업으로 나타났다면 생산품목중에서 가장 턴이 적은 생산품목으로 마우스를 움직여서 마우스 좌클릭해줘
    """

    # 개척자 움직임 + 팝업 처리
    macro(instruction=unit_instruction)
    macro(instruction=pop_up_instruction)
    macro(instruction=pop_up_instruction)

    # 전사 이동
    macro(instruction=unit_instruction)

    # 연구 선택
    macro(instruction=pop_up_instruction)
    macro(instruction=policy_instruction)

    # 생산 품목 선택
    macro(instruction=pop_up_instruction)
    macro(instruction=city_instruction)

    macro(instruction=pop_up_instruction)
