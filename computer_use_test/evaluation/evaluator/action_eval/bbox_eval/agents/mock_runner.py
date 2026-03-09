"""Mock agent runner for testing — returns a fixed click at (500, 300) + esc press."""

from __future__ import annotations

from computer_use_test.agent.models.schema import ClickAction, KeyPressAction

from ..schema import AgentResponse, DatasetCase
from .base import BaseAgentRunner


class MockAgentRunner(BaseAgentRunner):
    """Mock agent runner that returns a fixed response for testing."""

    def run_case(self, case: DatasetCase) -> AgentResponse:
        return AgentResponse(
            actions=[
                ClickAction(type="click", x=500, y=300, button="left"),
                KeyPressAction(type="press", keys=["esc"]),
            ],
            meta={"provider": "mock"},
        )
