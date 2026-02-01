from PIL import Image
from io import BytesIO
import base64
import pyautogui


def capture_screen_pil():
    """화면을 캡처하고 리사이징된 PIL 이미지를 반환"""
    screenshot = pyautogui.screenshot()

    # Retina 디스플레이 등 좌표계 보정을 위해 논리적 해상도 가져오기
    screen_w, screen_h = pyautogui.size()

    if screenshot.mode in ("RGBA", "P"):
        screenshot = screenshot.convert("RGB")

    # AI 전송용 리사이징 (최대 1024px)
    max_size = 1024
    screenshot.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    return screenshot, screen_w, screen_h


def image_to_base64(pil_image):
    """PIL 이미지를 Base64 문자열로 변환"""
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG", quality=80)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
