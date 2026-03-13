from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

try:
    import pyautogui
except Exception:  # pragma: no cover - GUI dependent
    pyautogui = None

try:
    import anthropic
except Exception:  # pragma: no cover - optional dependency
    anthropic = None

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:  # pragma: no cover - optional dependency
    genai = None
    genai_types = None

try:
    import openai
except Exception:  # pragma: no cover - optional dependency
    openai = None

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fully_autonomous_test")

NORMALIZING_RANGE = 1000
DEFAULT_GOAL = "과학 승리에 집중하되, 강한 위협이 있으면 생존과 방어를 우선해."
GAME_WINDOW_KEYWORDS = ("civilization", "civ6", "civ vi")
SCREEN_TYPES = (
    "popup",
    "next_turn",
    "unit_selected",
    "combat",
    "research",
    "civics",
    "city_production",
    "government_choice",
    "policy_cards",
    "governor",
    "diplomatic_envoy",
    "religion",
    "era_dedication",
    "world_congress",
    "deal_or_war",
    "text_input",
    "unknown",
)
BLOCKING_PRIORITIES = ("critical", "mandatory", "normal", "optional")
ALLOWED_ACTIONS = {"click", "double_click", "drag", "scroll", "press", "wait", "type"}
ALLOWED_KEYS = {"enter", "esc", "space", "b", "f", "m"}
SCROLLABLE_SCREENS = {"research", "civics", "city_production", "policy_cards", "religion", "world_congress"}
DEFAULT_MODELS = {
    "gemini": "gemini-3-flash-preview",
    "claude": "claude-sonnet-4-5-20250929",
    "gpt": "gpt-4o",
}

RULEBOOK_PROMPT = """
너는 문명6 실전 화면을 조작하는 컴퓨터-유즈 에이전트다.
아래 규칙을 항상 지켜라.

[공통 안전 규칙]
- 무작위 클릭 금지. 보이는 UI 근거가 없으면 wait / esc / 재분석을 선택.
- 회색, 잠금, 체크 완료, 비활성 항목은 절대 클릭하지 마.
- 필수 관리 화면(연구, 사회제도, 생산, 정부/정책, 총독, 종교, 시대 전략, 세계의회)이 남아 있으면 다음 턴을 누르지 마.
- 화면 변화가 없던 동일 시도는 반복하지 마. precision 재시도 -> 대안 후보 -> esc/wait -> stop 순서로 대응.
- 전쟁/거래 제안처럼 위험한 외교 화면은 기본적으로 보수적으로 닫거나 거절한다.
- type 입력은 이름/텍스트 입력 화면이 명확할 때만 허용한다.

[screen_type 분류]
- popup: 확인/취소/정책변경/연구선택/생산품목/사회제도 선택 같은 팝업
- next_turn: 우하단 다음 턴 또는 즉시 턴 종료 가능 상태
- unit_selected: 선택된 유닛의 행동 판단 화면
- combat: 공격 또는 방어 선택이 핵심인 전투 화면
- research: 기술 선택 또는 기술 트리
- civics: 사회 제도 선택 또는 트리
- city_production: 도시 생산 선택 화면
- government_choice: 새 정부 선택
- policy_cards: 정책 카드 관리 화면
- governor: 총독 임명/진급/배정
- diplomatic_envoy: 도시국가 사절 파견
- religion: 종교관/종교 선택
- era_dedication: 시대 전략 선택
- world_congress: 세계의회 투표
- deal_or_war: 거래/전쟁 선포/고위험 외교
- text_input: 이름 입력 등
- unknown: 위 항목이 아닌 불명 화면

[화면 우선순위]
1. popup / 위험 모달 / 확인 대화상자
2. 연구 / 사회제도 / 생산 / 정부 / 정책 / 총독 / 종교 / 시대전략 / 세계의회
3. 유닛 행동 / 전투
4. next_turn

[상세 규칙]
- popup: 확인/수락 버튼이 있으면 우선 처리. 정보성 팝업이면 esc 우선.
- research/civics: 전략과 부스트를 고려하되, 지금 선택 가능한 항목만 고른다.
- city_production: 전략에 맞는 활성 항목 우선, 없으면 가장 안전한 활성 저턴수 항목.
- government_choice: 비활성 정부 금지. 전략과 슬롯 구성을 보고 선택.
- policy_cards: 현재 슬롯과 카드 목록을 보고 drag로 장착/교체. 모든 정책 배정 버튼은 실제 조정 후에만 누른다.
- governor: 임명/진급/배정 순서 준수.
- religion/era/world_congress: 전체 선택지를 확인해야 하면 먼저 스크롤 탐색.
- unit_selected:
  - 개척자: 좋은 정착지에서 b
  - 건설자: 개선 가능한 자원 타일 우선
  - 전투 유닛: 확실한 이득이 있는 공격 우선, 아니면 좋은 타일로 이동, 체력 낮으면 방어
  - 정찰병: 미탐색 지역 우선
- combat: 확실한 공격 가능 대상이 있으면 공격, 아니면 후퇴/방어.
- deal_or_war: 명확한 전략 지시가 없으면 보수적으로 닫기/거절.
- unknown: 추측 클릭 금지. 재분석, esc, wait 중 안전한 것을 택한다.
""".strip()


@dataclass
class Box:
    left: int
    top: int
    right: int
    bottom: int

    def center(self) -> tuple[int, int]:
        return ((self.left + self.right) // 2, (self.top + self.bottom) // 2)

    def width(self) -> int:
        return max(0, self.right - self.left)

    def height(self) -> int:
        return max(0, self.bottom - self.top)

    def is_valid(self) -> bool:
        return self.right > self.left and self.bottom > self.top

    def clamped(self, normalizing_range: int = NORMALIZING_RANGE) -> Box:
        left = clamp(self.left, 0, normalizing_range)
        top = clamp(self.top, 0, normalizing_range)
        right = clamp(self.right, 0, normalizing_range)
        bottom = clamp(self.bottom, 0, normalizing_range)
        if right <= left:
            right = min(normalizing_range, left + 1)
        if bottom <= top:
            bottom = min(normalizing_range, top + 1)
        return Box(left=left, top=top, right=right, bottom=bottom)


@dataclass
class CandidateTarget:
    target_id: str
    label: str
    kind: str
    enabled: bool = True
    importance: str = "normal"
    confidence: float = 0.5
    hotkey: str = ""
    note: str = ""
    box: Box | None = None


@dataclass
class StrategyState:
    victory_goal: str = "science"
    phase: str = "opening"
    priorities: list[str] = field(default_factory=lambda: ["과학", "확장", "방어"])
    primitive_directives: dict[str, str] = field(default_factory=dict)
    risk_flags: list[str] = field(default_factory=list)
    reasoning: str = ""
    last_refreshed_successes: int = 0

    def to_prompt_string(self) -> str:
        directives = "\n".join(f"- {k}: {v}" for k, v in self.primitive_directives.items()) or "- 없음"
        priorities = " > ".join(self.priorities) if self.priorities else "없음"
        risks = ", ".join(self.risk_flags) if self.risk_flags else "없음"
        return (
            f"승리 목표: {self.victory_goal}\n"
            f"현재 단계: {self.phase}\n"
            f"우선순위: {priorities}\n"
            f"리스크: {risks}\n"
            f"행동 기준:\n{directives}\n"
            f"전략 근거: {self.reasoning or '없음'}"
        )


@dataclass
class ScreenAnalysis:
    screen_type: str = "unknown"
    blocking_priority: str = "normal"
    subgoal: str = "현재 화면을 안전하게 처리"
    expected_transition: str = ""
    reasoning: str = ""
    confidence: float = 0.0
    should_end_turn: bool = False
    entities: list[CandidateTarget] = field(default_factory=list)
    candidate_targets: list[CandidateTarget] = field(default_factory=list)


@dataclass
class CandidateAction:
    action: str = "wait"
    source_target_id: str = ""
    target_id: str = ""
    target_kind: str = ""
    x: int = -1
    y: int = -1
    end_x: int = -1
    end_y: int = -1
    button: str = "left"
    key: str = ""
    text: str = ""
    scroll_amount: int = 0
    duration: float = 0.0
    confidence: float = 0.0
    expected_outcome: str = ""
    reasoning: str = ""

    def summary(self) -> str:
        if self.action == "press":
            return f"press({self.key})"
        if self.action == "drag":
            return f"drag({self.x},{self.y}->{self.end_x},{self.end_y})"
        if self.action == "scroll":
            return f"scroll({self.scroll_amount})"
        if self.action == "wait":
            return f"wait({self.duration:.2f})"
        return f"{self.action}({self.x},{self.y})"


@dataclass
class ReflectionResult:
    outcome: str = "stall"
    failure_mode: str = "no_ui_change"
    next_policy: str = "retry_precise"
    memory_summary: str = ""
    reasoning: str = ""


@dataclass
class AnchorCacheEntry:
    screen_type: str
    label: str
    kind: str
    box: Box
    last_seen_step: int


@dataclass
class ExecutionRecord:
    step_index: int
    screen_type: str
    subgoal: str
    action_summary: str
    outcome: str
    failure_mode: str
    confidence: float
    memory_summary: str


@dataclass
class RunStats:
    total_steps: int = 0
    successful_steps: int = 0
    stall_count: int = 0
    turn_end_count: int = 0
    last_screen_type: str = ""


@dataclass
class AgentConfig:
    command: str
    provider_name: str
    model: str
    goal: str
    image_path: Path | None = None
    max_steps: int = 200
    max_turn_ends: int = 9999
    strategy_refresh_every_successes: int = 12
    judge_samples: int = 3
    action_delay: float = 0.25
    move_duration: float = 0.2
    normalizing_range: int = NORMALIZING_RANGE
    max_long_edge: int = 1600
    save_dir: Path | None = None
    allow_text_input: bool = False
    crop_to_game: bool = True
    dry_run: bool = False
    debug: bool = False
    use_som_overlay: bool = True


@dataclass
class CaptureFrame:
    image: Image.Image
    region_left: int
    region_top: int
    region_width: int
    region_height: int
    screen_width: int
    screen_height: int


@dataclass
class AgentMemory:
    history: list[ExecutionRecord] = field(default_factory=list)
    anchor_cache: dict[str, AnchorCacheEntry] = field(default_factory=dict)
    successful_patterns: dict[str, str] = field(default_factory=dict)
    run_stats: RunStats = field(default_factory=RunStats)
    current_subgoal: str = ""

    def add_record(self, record: ExecutionRecord) -> None:
        self.history.append(record)
        self.history = self.history[-30:]

    def remember_pattern(self, screen_type: str, summary: str) -> None:
        if summary:
            self.successful_patterns[screen_type] = summary[:300]

    def update_anchor(self, screen_type: str, target: CandidateTarget, step_index: int) -> None:
        if not target.label or target.box is None or not target.box.is_valid():
            return
        key = self._anchor_key(screen_type, target.label)
        self.anchor_cache[key] = AnchorCacheEntry(
            screen_type=screen_type,
            label=target.label,
            kind=target.kind,
            box=target.box.clamped(),
            last_seen_step=step_index,
        )

    def get_anchor(self, screen_type: str, label: str) -> Box | None:
        entry = self.anchor_cache.get(self._anchor_key(screen_type, label))
        if entry is None:
            return None
        return entry.box

    def recent_history_text(self, limit: int = 6) -> str:
        if not self.history:
            return "없음"
        lines = []
        for item in self.history[-limit:]:
            lines.append(
                f"- step={item.step_index} screen={item.screen_type} subgoal={item.subgoal} "
                f"action={item.action_summary} outcome={item.outcome}/{item.failure_mode}"
            )
        return "\n".join(lines)

    def recent_failures_text(self, limit: int = 4) -> str:
        failed = [item for item in self.history if item.outcome != "success"]
        if not failed:
            return "없음"
        return "\n".join(
            f"- step={item.step_index} {item.action_summary} -> {item.failure_mode}" for item in failed[-limit:]
        )

    def anchor_hints_text(self, screen_type: str) -> str:
        lines = []
        for entry in self.anchor_cache.values():
            if entry.screen_type != screen_type:
                continue
            cx, cy = entry.box.center()
            lines.append(f"- {entry.label} ({entry.kind}) ≈ ({cx},{cy})")
        return "\n".join(lines) if lines else "없음"

    def successful_pattern_text(self, screen_type: str) -> str:
        return self.successful_patterns.get(screen_type, "없음")

    @staticmethod
    def _anchor_key(screen_type: str, label: str) -> str:
        return f"{screen_type}:{label.strip().lower()}"


class ProviderError(RuntimeError):
    pass


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def _require_pyautogui():
    if pyautogui is None:
        raise RuntimeError("pyautogui is unavailable in this environment")
    return pyautogui


def strip_markdown(text: str) -> str:
    content = text.strip()
    if content.startswith("```json"):
        content = content[7:].lstrip()
    elif content.startswith("```"):
        content = content[3:].lstrip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if match:
        return match.group(1).strip()
    return re.sub(r"```+\s*$", "", content).strip()


def extract_json_payload(text: str) -> Any:
    stripped = strip_markdown(text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", stripped)
        if not match:
            raise
        return json.loads(match.group(1))


def image_to_base64(pil_image: Image.Image) -> str:
    buffer = BytesIO()
    pil_image.convert("RGB").save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def resize_for_model(image: Image.Image, max_long_edge: int) -> Image.Image:
    rgb = image.convert("RGB")
    w, h = rgb.size
    if max(w, h) <= max_long_edge:
        return rgb
    scale = max_long_edge / max(w, h)
    return rgb.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.LANCZOS)


class BaseVLMProvider:
    def __init__(self, provider_name: str, model: str):
        self.provider_name = provider_name
        self.model = model

    def generate_text_json(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 2048) -> Any:
        raise NotImplementedError

    def generate_image_json(
        self,
        image: Image.Image,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 3072,
    ) -> Any:
        raise NotImplementedError


class GeminiProvider(BaseVLMProvider):
    def __init__(self, model: str):
        super().__init__("gemini", model)
        if genai is None or genai_types is None:
            raise ProviderError("google-genai library is not installed")
        api_key = os.getenv("GENAI_API_KEY")
        if not api_key:
            raise ProviderError("GENAI_API_KEY is not set")
        self.client = genai.Client(api_key=api_key)

    def _generate(self, contents: list[Any], *, temperature: float, max_tokens: int) -> Any:
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return extract_json_payload(response.text)

    def generate_text_json(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 2048) -> Any:
        return self._generate([prompt], temperature=temperature, max_tokens=max_tokens)

    def generate_image_json(
        self,
        image: Image.Image,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 3072,
    ) -> Any:
        return self._generate([resize_for_model(image, 1600), prompt], temperature=temperature, max_tokens=max_tokens)


class ClaudeProvider(BaseVLMProvider):
    def __init__(self, model: str):
        super().__init__("claude", model)
        if anthropic is None:
            raise ProviderError("anthropic library is not installed")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate_text_json(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 2048) -> Any:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        return extract_json_payload(text)

    def generate_image_json(
        self,
        image: Image.Image,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 3072,
    ) -> Any:
        base64_img = image_to_base64(resize_for_model(image, 1600))
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_img,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        return extract_json_payload(text)


class OpenAIProvider(BaseVLMProvider):
    def __init__(self, model: str):
        super().__init__("gpt", model)
        if openai is None:
            raise ProviderError("openai library is not installed")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is not set")
        self.client = openai.OpenAI(api_key=api_key)

    def generate_text_json(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 2048) -> Any:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )
        text = response.choices[0].message.content or "{}"
        return extract_json_payload(text)

    def generate_image_json(
        self,
        image: Image.Image,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 3072,
    ) -> Any:
        base64_img = image_to_base64(resize_for_model(image, 1600))
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
                    ],
                }
            ],
        )
        text = response.choices[0].message.content or "{}"
        return extract_json_payload(text)


def create_provider(provider_name: str, model: str | None) -> BaseVLMProvider:
    provider_name = provider_name.lower()
    resolved_model = model or DEFAULT_MODELS.get(provider_name)
    if not resolved_model:
        raise ProviderError(f"Unknown provider '{provider_name}'")
    if provider_name == "gemini":
        return GeminiProvider(resolved_model)
    if provider_name == "claude":
        return ClaudeProvider(resolved_model)
    if provider_name in {"gpt", "openai"}:
        return OpenAIProvider(resolved_model)
    raise ProviderError(f"Unsupported provider '{provider_name}'")


def _detect_game_window() -> tuple[int, int, int, int] | None:
    try:
        import Quartz
    except Exception:
        return None
    windows = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID,
    )
    if not windows:
        return None
    for window in windows:
        name = str(window.get("kCGWindowName") or "").lower()
        owner = str(window.get("kCGWindowOwnerName") or "").lower()
        combined = f"{name} {owner}"
        if not any(keyword in combined for keyword in GAME_WINDOW_KEYWORDS):
            continue
        bounds = window.get("kCGWindowBounds")
        if not bounds:
            continue
        width = int(bounds.get("Width", 0))
        height = int(bounds.get("Height", 0))
        if width < 400 or height < 300:
            continue
        return int(bounds["X"]), int(bounds["Y"]), width, height
    return None


def capture_live_frame(config: AgentConfig) -> CaptureFrame:
    gui = _require_pyautogui()
    screenshot = gui.screenshot().convert("RGB")
    screen_width, screen_height = gui.size()
    region_left = 0
    region_top = 0
    region_width = screen_width
    region_height = screen_height

    if config.crop_to_game:
        bounds = _detect_game_window()
        if bounds:
            left, top, width, height = bounds
            scale = screenshot.size[0] / screen_width
            crop_box = (
                int(left * scale),
                int(top * scale),
                int((left + width) * scale),
                int((top + height) * scale),
            )
            screenshot = screenshot.crop(crop_box)
            region_left = left
            region_top = top
            region_width = width
            region_height = height

    screenshot = resize_for_model(screenshot, config.max_long_edge)
    return CaptureFrame(
        image=screenshot,
        region_left=region_left,
        region_top=region_top,
        region_width=region_width,
        region_height=region_height,
        screen_width=screen_width,
        screen_height=screen_height,
    )


def load_static_frame(config: AgentConfig) -> CaptureFrame:
    if config.image_path is None:
        raise ValueError("static-image requires --image")
    image = Image.open(config.image_path).convert("RGB")
    image = resize_for_model(image, config.max_long_edge)
    return CaptureFrame(
        image=image,
        region_left=0,
        region_top=0,
        region_width=image.size[0],
        region_height=image.size[1],
        screen_width=image.size[0],
        screen_height=image.size[1],
    )


def norm_to_region(value: int, extent: int, normalizing_range: int) -> int:
    return int((clamp(value, 0, normalizing_range) / normalizing_range) * extent)


def region_to_screen_point(
    x: int,
    y: int,
    frame: CaptureFrame,
    normalizing_range: int,
) -> tuple[int, int]:
    real_x = norm_to_region(x, frame.region_width, normalizing_range) + frame.region_left
    real_y = norm_to_region(y, frame.region_height, normalizing_range) + frame.region_top
    return real_x, real_y


def _normalized_mean_abs_diff(img1: Image.Image, img2: Image.Image, size: tuple[int, int]) -> float:
    gray1 = img1.convert("L").resize(size)
    gray2 = img2.convert("L").resize(size)
    pixels1 = gray1.tobytes()
    pixels2 = gray2.tobytes()
    total = sum(abs(a - b) for a, b in zip(pixels1, pixels2, strict=True))
    return total / (len(pixels1) * 255.0)


def _crop_local_region(img: Image.Image, x: int, y: int, normalizing_range: int) -> Image.Image:
    width, height = img.size
    px = norm_to_region(x, width, normalizing_range)
    py = norm_to_region(y, height, normalizing_range)
    half_w = max(32, int(width * 0.12))
    half_h = max(32, int(height * 0.12))
    return img.crop(
        (
            max(0, px - half_w),
            max(0, py - half_h),
            min(width, px + half_w),
            min(height, py + half_h),
        )
    )


def ui_changed(
    before: Image.Image,
    after: Image.Image,
    action: CandidateAction | None,
    normalizing_range: int,
) -> bool:
    global_diff = _normalized_mean_abs_diff(before, after, (96, 96))
    local_diff = 0.0
    if (
        action is not None
        and action.action in {"click", "double_click", "drag", "scroll"}
        and action.x >= 0
        and action.y >= 0
    ):
        crop_before = _crop_local_region(before, action.x, action.y, normalizing_range)
        crop_after = _crop_local_region(after, action.x, action.y, normalizing_range)
        local_diff = _normalized_mean_abs_diff(crop_before, crop_after, (64, 64))
    changed = global_diff >= 0.02 or local_diff >= 0.05
    logger.debug("UI diff global=%.4f local=%.4f changed=%s", global_diff, local_diff, changed)
    return changed


def _dict_to_box(raw: Any) -> Box | None:
    if not isinstance(raw, dict):
        return None
    keys = ("left", "top", "right", "bottom")
    if not all(key in raw for key in keys):
        return None
    return Box(
        left=clamp(raw.get("left", 0), 0, NORMALIZING_RANGE),
        top=clamp(raw.get("top", 0), 0, NORMALIZING_RANGE),
        right=clamp(raw.get("right", 0), 0, NORMALIZING_RANGE),
        bottom=clamp(raw.get("bottom", 0), 0, NORMALIZING_RANGE),
    ).clamped()


def _target_from_raw(raw: Any) -> CandidateTarget | None:
    if not isinstance(raw, dict):
        return None
    target_id = str(raw.get("target_id", "")).strip()
    label = str(raw.get("label", "")).strip()
    kind = str(raw.get("kind", "")).strip() or "unknown"
    if not target_id or not label:
        return None
    box = _dict_to_box(raw.get("box"))
    confidence = raw.get("confidence", 0.5)
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.5
    return CandidateTarget(
        target_id=target_id,
        label=label,
        kind=kind,
        enabled=bool(raw.get("enabled", True)),
        importance=str(raw.get("importance", "normal")).strip() or "normal",
        confidence=max(0.0, min(1.0, confidence)),
        hotkey=str(raw.get("hotkey", "")).strip(),
        note=str(raw.get("note", "")).strip(),
        box=box,
    )


def strategy_from_raw(raw: Any, successes: int) -> StrategyState:
    if not isinstance(raw, dict):
        return StrategyState(
            priorities=["과학", "확장", "방어"],
            primitive_directives={"unit_selected": "정착과 생존을 우선해."},
            reasoning="전략 JSON 파싱 실패 fallback",
            last_refreshed_successes=successes,
        )
    priorities = [str(item).strip() for item in raw.get("priorities", []) if str(item).strip()]
    directives_raw = raw.get("primitive_directives", {})
    directives = {}
    if isinstance(directives_raw, dict):
        directives = {str(k): str(v) for k, v in directives_raw.items() if str(k).strip() and str(v).strip()}
    risks = [str(item).strip() for item in raw.get("risk_flags", []) if str(item).strip()]
    return StrategyState(
        victory_goal=str(raw.get("victory_goal", "science")).strip() or "science",
        phase=str(raw.get("phase", "opening")).strip() or "opening",
        priorities=priorities or ["과학", "확장", "방어"],
        primitive_directives=directives,
        risk_flags=risks,
        reasoning=str(raw.get("reasoning", "")).strip(),
        last_refreshed_successes=successes,
    )


def analysis_from_raw(raw: Any) -> ScreenAnalysis:
    if not isinstance(raw, dict):
        return ScreenAnalysis(reasoning="screen analysis JSON 파싱 실패")
    screen_type = str(raw.get("screen_type", "unknown")).strip()
    if screen_type not in SCREEN_TYPES:
        screen_type = "unknown"
    blocking_priority = str(raw.get("blocking_priority", "normal")).strip()
    if blocking_priority not in BLOCKING_PRIORITIES:
        blocking_priority = "normal"
    entities = [_target_from_raw(item) for item in raw.get("entities", [])]
    targets = [_target_from_raw(item) for item in raw.get("candidate_targets", [])]
    try:
        confidence = float(raw.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    return ScreenAnalysis(
        screen_type=screen_type,
        blocking_priority=blocking_priority,
        subgoal=str(raw.get("subgoal", "현재 화면을 안전하게 처리")).strip() or "현재 화면을 안전하게 처리",
        expected_transition=str(raw.get("expected_transition", "")).strip(),
        reasoning=str(raw.get("reasoning", "")).strip(),
        confidence=max(0.0, min(1.0, confidence)),
        should_end_turn=bool(raw.get("should_end_turn", False)),
        entities=[item for item in entities if item is not None],
        candidate_targets=[item for item in targets if item is not None],
    )


def action_from_raw(raw: Any) -> CandidateAction:
    if not isinstance(raw, dict):
        return CandidateAction(action="wait", duration=1.0, reasoning="action JSON 파싱 실패")
    action = str(raw.get("action", "wait")).strip()
    if action not in ALLOWED_ACTIONS:
        action = "wait"
    confidence = raw.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0
    return CandidateAction(
        action=action,
        source_target_id=str(raw.get("source_target_id", "")).strip(),
        target_id=str(raw.get("target_id", "")).strip(),
        target_kind=str(raw.get("target_kind", "")).strip(),
        x=int(raw.get("x", -1)) if raw.get("x", -1) is not None else -1,
        y=int(raw.get("y", -1)) if raw.get("y", -1) is not None else -1,
        end_x=int(raw.get("end_x", -1)) if raw.get("end_x", -1) is not None else -1,
        end_y=int(raw.get("end_y", -1)) if raw.get("end_y", -1) is not None else -1,
        button=str(raw.get("button", "left")).strip() or "left",
        key=str(raw.get("key", "")).strip(),
        text=str(raw.get("text", "")).strip(),
        scroll_amount=int(raw.get("scroll_amount", 0)) if raw.get("scroll_amount", 0) is not None else 0,
        duration=float(raw.get("duration", 0.0) or 0.0),
        confidence=max(0.0, min(1.0, confidence)),
        expected_outcome=str(raw.get("expected_outcome", "")).strip(),
        reasoning=str(raw.get("reasoning", "")).strip(),
    )


def reflection_from_raw(raw: Any, changed: bool) -> ReflectionResult:
    if not isinstance(raw, dict):
        return ReflectionResult(
            outcome="success" if changed else "stall",
            failure_mode="unknown" if changed else "no_ui_change",
            next_policy="continue" if changed else "retry_precise",
            memory_summary="reflection JSON 파싱 실패 fallback",
        )
    return ReflectionResult(
        outcome=str(raw.get("outcome", "success" if changed else "stall")).strip(),
        failure_mode=str(raw.get("failure_mode", "unknown" if changed else "no_ui_change")).strip(),
        next_policy=str(raw.get("next_policy", "continue" if changed else "retry_precise")).strip(),
        memory_summary=str(raw.get("memory_summary", "")).strip(),
        reasoning=str(raw.get("reasoning", "")).strip(),
    )


def build_strategist_prompt(config: AgentConfig, memory: AgentMemory) -> str:
    return f"""
{RULEBOOK_PROMPT}

너는 문명6 장기 전략 설계자다.
사용자 목표: {config.goal}

최근 실행 기록:
{memory.recent_history_text()}

최근 실패:
{memory.recent_failures_text()}

JSON만 출력:
{{
  "victory_goal": "science|culture|domination|religious|diplomatic|score",
  "phase": "opening|expansion|stabilize|push",
  "priorities": ["핵심 우선순위 3~6개"],
  "primitive_directives": {{
    "popup": "팝업 처리 기준",
    "research": "기술 선택 기준",
    "civics": "사회 제도 기준",
    "city_production": "생산 기준",
    "government_choice": "정부 선택 기준",
    "policy_cards": "정책 카드 기준",
    "unit_selected": "유닛 기준",
    "combat": "전투 기준",
    "governor": "총독 기준",
    "diplomatic_envoy": "사절 기준",
    "religion": "종교 기준",
    "era_dedication": "시대 전략 기준",
    "world_congress": "세계의회 기준",
    "deal_or_war": "거래/전쟁 기준"
  }},
  "risk_flags": ["조심해야 할 것들"],
  "reasoning": "전략 요약"
}}

요구사항:
- 과한 모험보다 안정적 자동화를 우선해.
- 팝업 처리, 필수 관리 화면 해결, 유닛 운영, 턴 종료의 순서를 명확히 둬.
- 외교/전쟁은 명확한 이득이 없으면 보수적으로.
""".strip()


def build_analyst_prompt(strategy: StrategyState, memory: AgentMemory) -> str:
    return f"""
{RULEBOOK_PROMPT}

현재 전략:
{strategy.to_prompt_string()}

최근 실행 기록:
{memory.recent_history_text()}

성공 패턴 메모:
{json.dumps(memory.successful_patterns, ensure_ascii=False)}

너의 임무는 현재 스크린샷을 구조화된 상태로 해석하는 것이다.
후보 target은 지금 화면에서 실제로 행동 가능한 요소만 넣어라.
box는 현재 보이는 화면 기준 0-{NORMALIZING_RANGE} normalized coordinates.

JSON만 출력:
{{
  "screen_type": "{"|".join(SCREEN_TYPES)}",
  "blocking_priority": "critical|mandatory|normal|optional",
  "subgoal": "지금 한 줄 목표",
  "expected_transition": "이 화면이 끝나면 기대되는 상태",
  "should_end_turn": false,
  "confidence": 0.0,
  "reasoning": "판단 근거",
  "entities": [
    {{
      "target_id": "entity_1",
      "label": "텍스트 라벨",
      "kind": "button|list_item|tile|card|slot|tab|popup|panel|unit|icon",
      "enabled": true,
      "importance": "critical|high|normal|low",
      "confidence": 0.0,
      "note": "상태 설명",
      "box": {{"left":0,"top":0,"right":0,"bottom":0}}
    }}
  ],
  "candidate_targets": [
    {{
      "target_id": "target_1",
      "label": "행동 후보 라벨",
      "kind": "button|list_item|tile|card|slot|tab|popup|panel|unit|icon",
      "enabled": true,
      "importance": "critical|high|normal|low",
      "confidence": 0.0,
      "note": "왜 중요한가",
      "hotkey": "",
      "box": {{"left":0,"top":0,"right":0,"bottom":0}}
    }}
  ]
}}

규칙:
- candidate_targets는 최대 12개.
- 비활성/회색 요소는 enabled=false.
- next turn이 보여도 필수 화면이 남아 있으면 should_end_turn=false.
- 화면이 불확실하면 unknown으로 두고 reasoning에 왜 불확실한지 적어라.
""".strip()


def build_actor_prompt(strategy: StrategyState, analysis: ScreenAnalysis, memory: AgentMemory) -> str:
    candidate_payload = [target_to_prompt_dict(item) for item in analysis.candidate_targets]
    return f"""
{RULEBOOK_PROMPT}

현재 전략:
{strategy.to_prompt_string()}

현재 화면 분석:
{json.dumps(screen_analysis_to_prompt_dict(analysis), ensure_ascii=False)}

최근 실행 기록:
{memory.recent_history_text()}

최근 실패:
{memory.recent_failures_text()}

해당 screen_type에서 기억된 anchor:
{memory.anchor_hints_text(analysis.screen_type)}

해당 screen_type 성공 패턴:
{memory.successful_pattern_text(analysis.screen_type)}

가용 candidate targets:
{json.dumps(candidate_payload, ensure_ascii=False)}

다음 1개의 행동만 정해라.
pointing action은 가능하면 target_id 기반으로 선택해라.
drag는 source_target_id와 target_id를 모두 써라.
직접 좌표가 정말 필요할 때만 x/y 또는 end_x/end_y를 채워라.
JSON만 출력:
{{
  "action": "click|double_click|drag|scroll|press|wait|type",
  "source_target_id": "",
  "target_id": "",
  "target_kind": "button|list_item|tile|card|slot|tab|popup|panel|unit|icon",
  "x": -1,
  "y": -1,
  "end_x": -1,
  "end_y": -1,
  "button": "left|right",
  "key": "",
  "text": "",
  "scroll_amount": 0,
  "duration": 0.0,
  "confidence": 0.0,
  "expected_outcome": "행동 후 기대 상태",
  "reasoning": "행동 선택 근거"
}}

제약:
- next turn은 should_end_turn=true일 때만 허용.
- low confidence면 wait보다 더 안전한 대안 target이 있으면 그걸 선택.
- deal_or_war는 보수적으로 닫기/거절/esc 우선.
- text_input이 아니면 type 금지.
""".strip()


def build_judge_prompt(
    strategy: StrategyState,
    analysis: ScreenAnalysis,
    memory: AgentMemory,
    candidates: list[CandidateAction],
) -> str:
    candidate_payload = [candidate_action_to_prompt_dict(item) for item in candidates]
    return f"""
{RULEBOOK_PROMPT}

현재 전략:
{strategy.to_prompt_string()}

현재 화면 분석:
{json.dumps(screen_analysis_to_prompt_dict(analysis), ensure_ascii=False)}

최근 실패:
{memory.recent_failures_text()}

아래 후보 중 가장 안전하고, 가장 실제 진전을 만들 가능성이 높은 action 1개만 골라라.
stall을 반복할 가능성이 높은 후보는 피하라.

후보:
{json.dumps(candidate_payload, ensure_ascii=False)}

출력은 선택한 action JSON 하나만:
{{
  "action": "click|double_click|drag|scroll|press|wait|type",
  "source_target_id": "",
  "target_id": "",
  "target_kind": "",
  "x": -1,
  "y": -1,
  "end_x": -1,
  "end_y": -1,
  "button": "left|right",
  "key": "",
  "text": "",
  "scroll_amount": 0,
  "duration": 0.0,
  "confidence": 0.0,
  "expected_outcome": "",
  "reasoning": ""
}}
""".strip()


def build_grounding_prompt(
    target: CandidateTarget,
    action_kind: str,
    strategy: StrategyState,
    analysis: ScreenAnalysis,
    overlay_targets: list[dict[str, Any]] | None = None,
) -> str:
    overlay_note = ""
    if overlay_targets:
        overlay_note = (
            "아래 번호가 그려진 후보 중 올바른 대상의 번호를 고른 뒤, 그 대상 내부의 정확한 점을 반환해.\n"
            f"후보 설명: {json.dumps(overlay_targets, ensure_ascii=False)}\n"
            '추가로 "chosen_index" 정수를 반환해.\n'
        )
    return f"""
{RULEBOOK_PROMPT}

현재 전략:
{strategy.to_prompt_string()}

현재 screen_type: {analysis.screen_type}
현재 subgoal: {analysis.subgoal}
대상 label: {target.label}
대상 kind: {target.kind}
행동 종류: {action_kind}

{overlay_note}
현재 crop 이미지는 대상 후보 주변만 잘라낸 것이다.
반드시 실제 클릭/드래그하기 좋은 지점을 target 내부에서 골라라.

JSON만 출력:
{{
  "chosen_index": -1,
  "x": 0,
  "y": 0,
  "button": "left|right",
  "reasoning": "왜 이 점인가"
}}

좌표는 crop 이미지 기준 0-{NORMALIZING_RANGE} normalized coordinates.
""".strip()


def build_reflection_prompt(
    strategy: StrategyState,
    analysis: ScreenAnalysis,
    action: CandidateAction,
    changed: bool,
    memory: AgentMemory,
) -> str:
    return f"""
{RULEBOOK_PROMPT}

현재 전략:
{strategy.to_prompt_string()}

실행 전 screen_type: {analysis.screen_type}
실행 전 subgoal: {analysis.subgoal}
예상 전이: {analysis.expected_transition or "없음"}
실행 action: {json.dumps(candidate_action_to_prompt_dict(action), ensure_ascii=False)}
로컬 diff 판단: {"changed" if changed else "not_changed"}

최근 실패:
{memory.recent_failures_text()}

before/after 결합 이미지가 주어진다.
step 결과를 판정해라. JSON만 출력:
{{
  "outcome": "success|partial|stall|regression|unsafe",
  "failure_mode": "none|no_ui_change|wrong_target|blocked_modal|ambiguous_ui|off_window|repeated_loop|unknown",
  "next_policy": "continue|retry_precise|alternative_candidate|escape|scroll_probe|wait|stop",
  "memory_summary": "짧은 학습 요약",
  "reasoning": "판정 근거"
}}
""".strip()


def target_to_prompt_dict(target: CandidateTarget) -> dict[str, Any]:
    payload = asdict(target)
    if target.box is not None:
        payload["box"] = asdict(target.box)
    return payload


def screen_analysis_to_prompt_dict(analysis: ScreenAnalysis) -> dict[str, Any]:
    return {
        "screen_type": analysis.screen_type,
        "blocking_priority": analysis.blocking_priority,
        "subgoal": analysis.subgoal,
        "expected_transition": analysis.expected_transition,
        "should_end_turn": analysis.should_end_turn,
        "confidence": analysis.confidence,
        "reasoning": analysis.reasoning,
        "entities": [target_to_prompt_dict(item) for item in analysis.entities],
        "candidate_targets": [target_to_prompt_dict(item) for item in analysis.candidate_targets],
    }


def candidate_action_to_prompt_dict(action: CandidateAction) -> dict[str, Any]:
    return {
        "action": action.action,
        "source_target_id": action.source_target_id,
        "target_id": action.target_id,
        "target_kind": action.target_kind,
        "x": action.x,
        "y": action.y,
        "end_x": action.end_x,
        "end_y": action.end_y,
        "button": action.button,
        "key": action.key,
        "text": action.text,
        "scroll_amount": action.scroll_amount,
        "duration": action.duration,
        "confidence": action.confidence,
        "expected_outcome": action.expected_outcome,
        "reasoning": action.reasoning,
    }


def combine_before_after(before: Image.Image, after: Image.Image) -> Image.Image:
    width = max(before.size[0], after.size[0])
    height = before.size[1] + after.size[1] + 40
    canvas = Image.new("RGB", (width, height), (20, 20, 20))
    canvas.paste(before.convert("RGB"), (0, 20))
    canvas.paste(after.convert("RGB"), (0, before.size[1] + 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 2), "BEFORE", fill=(255, 255, 0))
    draw.text((10, before.size[1] + 22), "AFTER", fill=(0, 255, 255))
    return canvas


def crop_from_box(frame: CaptureFrame, box: Box, padding_ratio: float = 0.12) -> tuple[Image.Image, Box]:
    image = frame.image
    width, height = image.size
    left = norm_to_region(box.left, width, NORMALIZING_RANGE)
    top = norm_to_region(box.top, height, NORMALIZING_RANGE)
    right = norm_to_region(box.right, width, NORMALIZING_RANGE)
    bottom = norm_to_region(box.bottom, height, NORMALIZING_RANGE)
    pad_x = int((right - left) * padding_ratio) + 20
    pad_y = int((bottom - top) * padding_ratio) + 20
    crop_left = max(0, left - pad_x)
    crop_top = max(0, top - pad_y)
    crop_right = min(width, right + pad_x)
    crop_bottom = min(height, bottom + pad_y)
    crop = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    crop_box = Box(
        left=int(round((crop_left / width) * NORMALIZING_RANGE)),
        top=int(round((crop_top / height) * NORMALIZING_RANGE)),
        right=int(round((crop_right / width) * NORMALIZING_RANGE)),
        bottom=int(round((crop_bottom / height) * NORMALIZING_RANGE)),
    ).clamped()
    return crop, crop_box


def local_norm_to_global_norm(local_x: int, local_y: int, crop_box: Box) -> tuple[int, int]:
    crop_width = max(1, crop_box.width())
    crop_height = max(1, crop_box.height())
    global_x = crop_box.left + int((clamp(local_x, 0, NORMALIZING_RANGE) / NORMALIZING_RANGE) * crop_width)
    global_y = crop_box.top + int((clamp(local_y, 0, NORMALIZING_RANGE) / NORMALIZING_RANGE) * crop_height)
    return clamp(global_x, 0, NORMALIZING_RANGE), clamp(global_y, 0, NORMALIZING_RANGE)


def draw_som_overlay(image: Image.Image, targets: list[CandidateTarget]) -> tuple[Image.Image, list[dict[str, Any]]]:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    metadata = []
    width, height = overlay.size
    for index, target in enumerate(targets, start=1):
        if target.box is None or not target.box.is_valid():
            continue
        left = norm_to_region(target.box.left, width, NORMALIZING_RANGE)
        top = norm_to_region(target.box.top, height, NORMALIZING_RANGE)
        right = norm_to_region(target.box.right, width, NORMALIZING_RANGE)
        bottom = norm_to_region(target.box.bottom, height, NORMALIZING_RANGE)
        color = (255, 80, 80) if target.importance == "critical" else (80, 220, 255)
        draw.rectangle((left, top, right, bottom), outline=color, width=3)
        badge_box = (left, max(0, top - 18), left + 28, top + 10)
        draw.rectangle(badge_box, fill=color)
        draw.text((left + 7, max(0, top - 16)), str(index), fill=(0, 0, 0), font=font)
        metadata.append(
            {
                "index": index,
                "target_id": target.target_id,
                "label": target.label,
                "kind": target.kind,
                "importance": target.importance,
            }
        )
    return overlay, metadata


class AutonomousCivAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.provider = create_provider(config.provider_name, config.model)
        self.memory = AgentMemory()
        self.strategy = StrategyState(
            priorities=["과학", "확장", "방어"],
            primitive_directives={"unit_selected": "생존과 확장 균형"},
            reasoning="초기 fallback 전략",
        )

    def maybe_refresh_strategy(self, frame: CaptureFrame, force: bool = False) -> None:
        successes = self.memory.run_stats.successful_steps
        needs_refresh = force
        if self.memory.run_stats.total_steps == 0:
            needs_refresh = True
        if successes - self.strategy.last_refreshed_successes >= self.config.strategy_refresh_every_successes:
            needs_refresh = True
        if self.memory.run_stats.stall_count >= 2:
            needs_refresh = True
        if not needs_refresh:
            return
        prompt = build_strategist_prompt(self.config, self.memory)
        raw = self.safe_generate_image_json(frame.image, prompt, temperature=0.15, max_tokens=2048)
        self.strategy = strategy_from_raw(raw, successes=successes)
        logger.info(
            "Strategy refreshed: goal=%s phase=%s priorities=%s",
            self.strategy.victory_goal,
            self.strategy.phase,
            " > ".join(self.strategy.priorities),
        )

    def analyze_screen(self, frame: CaptureFrame) -> ScreenAnalysis:
        prompt = build_analyst_prompt(self.strategy, self.memory)
        raw = self.safe_generate_image_json(frame.image, prompt, temperature=0.1, max_tokens=3072)
        analysis = analysis_from_raw(raw)
        for target in analysis.candidate_targets:
            self.memory.update_anchor(analysis.screen_type, target, self.memory.run_stats.total_steps)
        self.memory.current_subgoal = analysis.subgoal
        return analysis

    def sample_action_candidates(
        self, frame: CaptureFrame, analysis: ScreenAnalysis, count: int
    ) -> list[CandidateAction]:
        prompt = build_actor_prompt(self.strategy, analysis, self.memory)
        candidates: list[CandidateAction] = []
        temperatures = [0.15, 0.35, 0.55, 0.75]
        for index in range(max(1, count)):
            raw = self.safe_generate_image_json(
                frame.image,
                prompt,
                temperature=temperatures[min(index, len(temperatures) - 1)],
                max_tokens=2048,
            )
            candidate = action_from_raw(raw)
            if candidate.action not in ALLOWED_ACTIONS:
                continue
            candidates.append(candidate)
        if not candidates:
            candidates.append(CandidateAction(action="wait", duration=1.0, reasoning="action 후보 생성 실패 fallback"))
        return candidates

    def choose_action(self, frame: CaptureFrame, analysis: ScreenAnalysis) -> CandidateAction:
        need_judge = self.memory.run_stats.stall_count > 0
        if analysis.confidence < 0.65:
            need_judge = True
        if (
            analysis.screen_type in {"policy_cards", "world_congress", "city_production"}
            and len(analysis.candidate_targets) > 4
        ):
            need_judge = True
        sample_count = self.config.judge_samples if need_judge else 1
        candidates = self.sample_action_candidates(frame, analysis, sample_count)
        if len(candidates) == 1:
            return candidates[0]
        prompt = build_judge_prompt(self.strategy, analysis, self.memory, candidates)
        raw = self.safe_generate_image_json(frame.image, prompt, temperature=0.1, max_tokens=2048)
        judged = action_from_raw(raw)
        return judged if judged.action in ALLOWED_ACTIONS else candidates[0]

    def resolve_action(self, frame: CaptureFrame, analysis: ScreenAnalysis, action: CandidateAction) -> CandidateAction:
        resolved = action_from_raw(candidate_action_to_prompt_dict(action))
        if resolved.action in {"press", "wait", "type"}:
            return resolved

        if resolved.action == "scroll" and resolved.x < 0 and resolved.y < 0:
            panel_target = self.find_best_scroll_target(analysis)
            if panel_target is not None and panel_target.box is not None:
                x, y = panel_target.box.center()
                resolved.x = x
                resolved.y = y
            return resolved

        if resolved.action in {"click", "double_click", "scroll"}:
            target = self.resolve_target(analysis, resolved.target_id, resolved.target_kind)
            if (
                target is None
                and resolved.target_id == ""
                and self.config.use_som_overlay
                and len(analysis.candidate_targets) > 1
            ):
                target = self.choose_target_with_overlay(frame, analysis)
                if target is not None:
                    resolved.target_id = target.target_id
            if target is not None and target.box is not None:
                x, y = self.resolve_point(frame, analysis, target, resolved.action, resolved.confidence)
                resolved.x = x
                resolved.y = y
                resolved.target_kind = target.kind
            return resolved

        if resolved.action == "drag":
            source = self.resolve_target(analysis, resolved.source_target_id, "card")
            dest = self.resolve_target(analysis, resolved.target_id, resolved.target_kind or "slot")
            if source is not None and source.box is not None:
                resolved.x, resolved.y = self.resolve_point(frame, analysis, source, "drag_source", resolved.confidence)
            if dest is not None and dest.box is not None:
                resolved.end_x, resolved.end_y = self.resolve_point(
                    frame, analysis, dest, "drag_dest", resolved.confidence
                )
            return resolved

        return resolved

    def resolve_target(self, analysis: ScreenAnalysis, target_id: str, fallback_kind: str) -> CandidateTarget | None:
        if target_id:
            for target in analysis.candidate_targets:
                if target.target_id == target_id:
                    return target
        if not target_id and analysis.candidate_targets:
            for target in analysis.candidate_targets:
                if target.kind == fallback_kind:
                    return target
        if target_id:
            for target in analysis.candidate_targets:
                if target.label == target_id:
                    return target
        return None

    def choose_target_with_overlay(self, frame: CaptureFrame, analysis: ScreenAnalysis) -> CandidateTarget | None:
        targets = [target for target in analysis.candidate_targets if target.box is not None and target.enabled]
        if len(targets) < 2:
            return None
        overlay, metadata = draw_som_overlay(frame.image, targets)
        if self.config.save_dir:
            overlay_path = self.config.save_dir / f"step_{self.memory.run_stats.total_steps:03d}_overlay.png"
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            overlay.save(overlay_path)
        prompt = build_grounding_prompt(
            target=targets[0],
            action_kind="choose_target_index",
            strategy=self.strategy,
            analysis=analysis,
            overlay_targets=metadata,
        )
        raw = self.safe_generate_image_json(overlay, prompt, temperature=0.1, max_tokens=1024)
        chosen_index = -1
        if isinstance(raw, dict):
            try:
                chosen_index = int(raw.get("chosen_index", -1))
            except Exception:
                chosen_index = -1
        for item in metadata:
            if item["index"] == chosen_index:
                target_id = item["target_id"]
                return next((target for target in targets if target.target_id == target_id), None)
        return None

    def resolve_point(
        self,
        frame: CaptureFrame,
        analysis: ScreenAnalysis,
        target: CandidateTarget,
        action_kind: str,
        confidence: float,
    ) -> tuple[int, int]:
        if target.box is None:
            anchor_box = self.memory.get_anchor(analysis.screen_type, target.label)
            if anchor_box is not None:
                return anchor_box.center()
            return NORMALIZING_RANGE // 2, NORMALIZING_RANGE // 2
        if confidence >= 0.75 and target.box.width() <= 180 and target.box.height() <= 140:
            return target.box.center()

        crop, crop_box = crop_from_box(frame, target.box)
        overlay_targets = None
        if (
            self.config.use_som_overlay
            and len(analysis.candidate_targets) > 1
            and target.kind in {"button", "list_item", "card", "slot", "tab"}
        ):
            overlay_targets = [
                {
                    "index": 1,
                    "target_id": target.target_id,
                    "label": target.label,
                    "kind": target.kind,
                }
            ]
        prompt = build_grounding_prompt(target, action_kind, self.strategy, analysis, overlay_targets=overlay_targets)
        raw = self.safe_generate_image_json(crop, prompt, temperature=0.1, max_tokens=1024)
        local_x = NORMALIZING_RANGE // 2
        local_y = NORMALIZING_RANGE // 2
        if isinstance(raw, dict):
            local_x = int(raw.get("x", NORMALIZING_RANGE // 2))
            local_y = int(raw.get("y", NORMALIZING_RANGE // 2))
        return local_norm_to_global_norm(local_x, local_y, crop_box)

    def validate_action(self, action: CandidateAction, analysis: ScreenAnalysis) -> tuple[bool, str]:
        if action.action not in ALLOWED_ACTIONS:
            return False, f"unsupported action: {action.action}"
        if action.action in {"click", "double_click", "scroll"}:
            if action.x < 0 or action.y < 0:
                return False, "pointing action has unresolved coordinates"
            if not (0 <= action.x <= NORMALIZING_RANGE and 0 <= action.y <= NORMALIZING_RANGE):
                return False, "pointing action outside normalized range"
        if action.action == "drag":
            coords = [action.x, action.y, action.end_x, action.end_y]
            if any(value < 0 for value in coords):
                return False, "drag has unresolved coordinates"
            if action.x == action.end_x and action.y == action.end_y:
                return False, "drag start and end are identical"
        if action.action == "press":
            if action.key not in ALLOWED_KEYS:
                return False, f"key '{action.key}' is not allowed"
            if action.key == "enter" and analysis.should_end_turn is False and analysis.screen_type == "next_turn":
                return False, "next turn is not allowed while should_end_turn=false"
        if action.action == "type":
            if not self.config.allow_text_input or analysis.screen_type != "text_input":
                return False, "text input is disabled outside explicit text_input screens"
        if action.action == "scroll":
            if action.scroll_amount == 0:
                return False, "scroll amount must be non-zero"
            if analysis.screen_type not in SCROLLABLE_SCREENS:
                return False, f"scroll is not allowed on {analysis.screen_type}"
        return True, ""

    def execute_action(self, action: CandidateAction, frame: CaptureFrame) -> None:
        if self.config.dry_run:
            logger.info("DRY RUN: %s", action.summary())
            return
        gui = _require_pyautogui()
        gui.FAILSAFE = True
        if action.action == "wait":
            time.sleep(max(0.05, action.duration or 0.8))
            return
        if action.action in {"click", "double_click"}:
            real_x, real_y = region_to_screen_point(action.x, action.y, frame, self.config.normalizing_range)
            gui.moveTo(real_x, real_y, duration=max(0.0, self.config.move_duration))
            if action.action == "click":
                gui.click(button=action.button)
            else:
                gui.doubleClick(button=action.button)
            self.move_cursor_to_center(frame)
        elif action.action == "drag":
            start_x, start_y = region_to_screen_point(action.x, action.y, frame, self.config.normalizing_range)
            end_x, end_y = region_to_screen_point(action.end_x, action.end_y, frame, self.config.normalizing_range)
            gui.moveTo(start_x, start_y, duration=max(0.0, self.config.move_duration))
            gui.mouseDown(button=action.button)
            time.sleep(0.08)
            gui.moveTo(end_x, end_y, duration=max(0.0, self.config.move_duration))
            time.sleep(0.05)
            gui.mouseUp(button=action.button)
            self.move_cursor_to_center(frame)
        elif action.action == "scroll":
            real_x, real_y = region_to_screen_point(action.x, action.y, frame, self.config.normalizing_range)
            gui.moveTo(real_x, real_y, duration=max(0.0, self.config.move_duration))
            gui.scroll(action.scroll_amount)
        elif action.action == "press":
            gui.press(action.key)
        elif action.action == "type":
            gui.write(action.text, interval=0.05)
        time.sleep(max(0.0, self.config.action_delay))

    @staticmethod
    def move_cursor_to_center(frame: CaptureFrame) -> None:
        if pyautogui is None:
            return
        center_x = frame.region_left + frame.region_width // 2
        center_y = frame.region_top + frame.region_height // 2
        pyautogui.moveTo(center_x, center_y, duration=0.1)

    def reflect(
        self,
        before_frame: CaptureFrame,
        after_frame: CaptureFrame,
        analysis: ScreenAnalysis,
        action: CandidateAction,
        changed: bool,
    ) -> ReflectionResult:
        if action.action == "wait":
            return ReflectionResult(
                outcome="success" if changed else "stall",
                failure_mode="none" if changed else "no_ui_change",
                next_policy="continue" if changed else "alternative_candidate",
                memory_summary=action.reasoning or action.summary(),
            )
        combined = combine_before_after(before_frame.image, after_frame.image)
        prompt = build_reflection_prompt(self.strategy, analysis, action, changed, self.memory)
        raw = self.safe_generate_image_json(combined, prompt, temperature=0.1, max_tokens=1536)
        return reflection_from_raw(raw, changed)

    def safe_generate_image_json(self, image: Image.Image, prompt: str, *, temperature: float, max_tokens: int) -> Any:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return self.provider.generate_image_json(image, prompt, temperature=temperature, max_tokens=max_tokens)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("image JSON generation failed (attempt %s/3): %s", attempt + 1, exc)
        logger.error("image JSON generation exhausted retries: %s", last_error)
        return {}

    def safe_generate_text_json(self, prompt: str, *, temperature: float, max_tokens: int) -> Any:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return self.provider.generate_text_json(prompt, temperature=temperature, max_tokens=max_tokens)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("text JSON generation failed (attempt %s/3): %s", attempt + 1, exc)
        logger.error("text JSON generation exhausted retries: %s", last_error)
        return {}

    def find_best_scroll_target(self, analysis: ScreenAnalysis) -> CandidateTarget | None:
        for target in analysis.candidate_targets:
            if target.kind in {"panel", "list_item", "card", "tab"} and target.box is not None:
                return target
        return next((item for item in analysis.entities if item.box is not None), None)

    def save_step_artifacts(
        self,
        step_index: int,
        before_frame: CaptureFrame,
        after_frame: CaptureFrame | None = None,
    ) -> None:
        if self.config.save_dir is None:
            return
        self.config.save_dir.mkdir(parents=True, exist_ok=True)
        before_frame.image.save(self.config.save_dir / f"step_{step_index:03d}_before.png")
        if after_frame is not None:
            after_frame.image.save(self.config.save_dir / f"step_{step_index:03d}_after.png")

    def run_step(self, before_frame: CaptureFrame) -> tuple[CandidateAction, ScreenAnalysis, ReflectionResult]:
        self.maybe_refresh_strategy(before_frame)
        analysis = self.analyze_screen(before_frame)
        if (
            self.memory.run_stats.last_screen_type
            and self.memory.run_stats.last_screen_type != analysis.screen_type
            and self.memory.run_stats.successful_steps - self.strategy.last_refreshed_successes >= 4
        ):
            self.maybe_refresh_strategy(before_frame, force=True)
            analysis = self.analyze_screen(before_frame)
        action = self.choose_action(before_frame, analysis)
        action = self.resolve_action(before_frame, analysis, action)
        valid, reason = self.validate_action(action, analysis)
        if not valid:
            logger.warning("Invalid action '%s': %s", action.summary(), reason)
            action = self.safe_recovery_action(analysis, reason)
        logger.info(
            "Step %s screen=%s subgoal=%s action=%s conf=%.2f",
            self.memory.run_stats.total_steps,
            analysis.screen_type,
            analysis.subgoal,
            action.summary(),
            action.confidence,
        )
        self.execute_action(action, before_frame)
        after_frame = (
            capture_live_frame(self.config)
            if self.config.command in {"live", "dry-run", "analyze-once"}
            else before_frame
        )
        self.save_step_artifacts(self.memory.run_stats.total_steps, before_frame, after_frame)
        changed = ui_changed(before_frame.image, after_frame.image, action, self.config.normalizing_range)
        reflection = self.reflect(before_frame, after_frame, analysis, action, changed)
        return action, analysis, reflection

    def safe_recovery_action(self, analysis: ScreenAnalysis, reason: str) -> CandidateAction:
        if analysis.screen_type in {"popup", "deal_or_war", "unknown"}:
            return CandidateAction(action="press", key="esc", confidence=0.4, reasoning=f"safe recovery: {reason}")
        if analysis.screen_type in SCROLLABLE_SCREENS:
            target = self.find_best_scroll_target(analysis)
            if target is not None and target.box is not None:
                x, y = target.box.center()
                return CandidateAction(
                    action="scroll",
                    x=x,
                    y=y,
                    scroll_amount=-400,
                    confidence=0.4,
                    reasoning=f"scroll probe recovery: {reason}",
                )
        return CandidateAction(action="wait", duration=1.0, confidence=0.2, reasoning=f"wait recovery: {reason}")

    def update_after_step(
        self,
        action: CandidateAction,
        analysis: ScreenAnalysis,
        reflection: ReflectionResult,
    ) -> None:
        self.memory.run_stats.total_steps += 1
        success = reflection.outcome in {"success", "partial"} and reflection.failure_mode not in {
            "off_window",
            "repeated_loop",
        }
        if success:
            self.memory.run_stats.successful_steps += 1
            self.memory.run_stats.stall_count = 0
            self.memory.remember_pattern(analysis.screen_type, reflection.memory_summary or action.reasoning)
            if action.target_id:
                target = self.resolve_target(analysis, action.target_id, action.target_kind)
                if target is not None:
                    self.memory.update_anchor(analysis.screen_type, target, self.memory.run_stats.total_steps)
        else:
            self.memory.run_stats.stall_count += 1
        if analysis.screen_type == "next_turn" and action.action in {"press", "click"} and action.key in {"enter", ""}:
            if success:
                self.memory.run_stats.turn_end_count += 1
        self.memory.run_stats.last_screen_type = analysis.screen_type
        self.memory.add_record(
            ExecutionRecord(
                step_index=self.memory.run_stats.total_steps,
                screen_type=analysis.screen_type,
                subgoal=analysis.subgoal,
                action_summary=action.summary(),
                outcome=reflection.outcome,
                failure_mode=reflection.failure_mode,
                confidence=action.confidence,
                memory_summary=reflection.memory_summary or action.reasoning,
            )
        )

    def run_live(self) -> None:
        for step_index in range(1, self.config.max_steps + 1):
            if self.memory.run_stats.turn_end_count >= self.config.max_turn_ends:
                logger.info("Reached max_turn_ends=%s", self.config.max_turn_ends)
                return
            before_frame = capture_live_frame(self.config)
            action, analysis, reflection = self.run_step(before_frame)
            self.update_after_step(action, analysis, reflection)
            logger.info(
                "Result step=%s outcome=%s failure=%s next_policy=%s",
                step_index,
                reflection.outcome,
                reflection.failure_mode,
                reflection.next_policy,
            )
            if self.memory.run_stats.stall_count >= 4 or reflection.next_policy == "stop":
                logger.warning("Stopping due to repeated stall or explicit stop policy")
                return

    def analyze_once(self, frame: CaptureFrame) -> None:
        self.maybe_refresh_strategy(frame, force=True)
        analysis = self.analyze_screen(frame)
        action = self.choose_action(frame, analysis)
        action = self.resolve_action(frame, analysis, action)
        payload = {
            "strategy": asdict(self.strategy),
            "analysis": screen_analysis_to_prompt_dict(analysis),
            "action": candidate_action_to_prompt_dict(action),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SOTA-inspired one-file Civilization VI autonomous agent")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "claude", "gpt"], help="VLM provider")
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--goal", default=DEFAULT_GOAL, help="High-level autonomous goal")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum live steps")
    parser.add_argument("--max-turn-ends", type=int, default=9999, help="Maximum confirmed turn-end actions")
    parser.add_argument(
        "--strategy-refresh-turns",
        type=int,
        default=12,
        help="Refresh strategy after this many successful steps",
    )
    parser.add_argument("--judge-samples", type=int, default=3, help="Candidate samples for judge mode")
    parser.add_argument("--action-delay", type=float, default=0.25, help="Delay after each action")
    parser.add_argument("--move-duration", type=float, default=0.2, help="Mouse move duration")
    parser.add_argument("--max-long-edge", type=int, default=1600, help="Maximum long edge for model input")
    parser.add_argument("--save-dir", default="", help="Optional directory for debug screenshots")
    parser.add_argument("--allow-type", action="store_true", help="Allow type action on text_input screens")
    parser.add_argument("--no-game-crop", action="store_true", help="Disable game window cropping")
    parser.add_argument("--no-som", action="store_true", help="Disable Set-of-Mark overlay fallback")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("live", help="Run the live autonomous loop")
    subparsers.add_parser("dry-run", help="Run the live loop without executing GUI actions")
    subparsers.add_parser("analyze-once", help="Capture once and print strategy/analysis/action")
    static_parser = subparsers.add_parser("static-image", help="Run strategy/analysis/action on a saved screenshot")
    static_parser.add_argument("--image", required=True, help="Path to image file")
    return parser


def config_from_args(args: argparse.Namespace) -> AgentConfig:
    save_dir = Path(args.save_dir).expanduser() if args.save_dir else None
    image_path = Path(args.image).expanduser() if getattr(args, "image", None) else None
    model = args.model or DEFAULT_MODELS[args.provider]
    return AgentConfig(
        command=args.command,
        provider_name=args.provider,
        model=model,
        goal=args.goal,
        image_path=image_path,
        max_steps=args.max_steps,
        max_turn_ends=args.max_turn_ends,
        strategy_refresh_every_successes=args.strategy_refresh_turns,
        judge_samples=max(1, args.judge_samples),
        action_delay=max(0.0, args.action_delay),
        move_duration=max(0.0, args.move_duration),
        max_long_edge=max(400, args.max_long_edge),
        save_dir=save_dir,
        allow_text_input=args.allow_type,
        crop_to_game=not args.no_game_crop,
        dry_run=args.command == "dry-run",
        debug=args.debug,
        use_som_overlay=not args.no_som,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    if config.debug:
        logger.setLevel(logging.DEBUG)
    try:
        agent = AutonomousCivAgent(config)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to initialize agent: %s", exc)
        return 1

    try:
        if config.command in {"live", "dry-run"}:
            if pyautogui is None:
                raise RuntimeError("pyautogui is unavailable; live mode cannot run")
            agent.run_live()
        elif config.command == "analyze-once":
            frame = capture_live_frame(config)
            agent.analyze_once(frame)
        elif config.command == "static-image":
            frame = load_static_frame(config)
            agent.analyze_once(frame)
        else:
            parser.error(f"Unknown command: {config.command}")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.error("Execution failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
