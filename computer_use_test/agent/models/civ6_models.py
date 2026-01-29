from enum import Enum
from typing import List, Union, Optional
from pydantic import BaseModel, Field

# 0. 행동 타입 정의 (PyAutoGUI 대응)
class ActionType(str, Enum):
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    PRESS = "press"  # 키보드 입력
    DRAG = "drag"
    WAIT = "wait"

# 기본 Action 모델
class BaseAction(BaseModel):
    action_type: ActionType
    description: Optional[str] = None # 디버깅용 설명

class ClickAction(BaseAction):
    action_type: ActionType = ActionType.CLICK
    x: int
    y: int
    button: str = "left"

class KeyPressAction(BaseAction):
    action_type: ActionType = ActionType.PRESS
    keys: List[str] # 예: ['ctrl', 'c']
    interval: float = 0.1

class DragAction(BaseAction):
    action_type: ActionType = ActionType.DRAG
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    duration: float = 0.5

# 통합 Action 타입 (Polymorphic)
Action = Union[ClickAction, KeyPressAction, DragAction]

# 에이전트가 반환할 최종 Plan (Action들의 시퀀스)
class AgentPlan(BaseModel):
    primitive_name: str
    reasoning: str
    actions: List[Action]
