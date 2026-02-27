"""Tests for agent_runner.py and builtin_agent_adapter.py."""

import sys
import textwrap

import pytest

from computer_use_test.agent.models.schema import ClickAction, KeyPressAction
from computer_use_test.evaluation.evaluator.action_eval.bbox_eval import (
    AgentRunnerError,
    BBox,
    BuiltinAgentRunner,
    DatasetCase,
    GTActionSet,
    GTClickAction,
    ImageSize,
    MockAgentRunner,
    SubprocessAgentRunner,
)


@pytest.fixture
def sample_case():
    return DatasetCase(
        case_id="test_001",
        instruction="Click the button",
        screenshot_path="test.png",
        image_size=ImageSize(width=1000, height=1000),
        action_sets=[GTActionSet(actions=[GTClickAction(target_bbox=BBox(x_min=80, y_min=180, x_max=120, y_max=220))])],
    )


class TestMockAgentRunner:
    def test_returns_actions(self, sample_case):
        runner = MockAgentRunner()
        response = runner.run_case(sample_case)
        assert len(response.actions) == 2
        assert isinstance(response.actions[0], ClickAction)
        assert isinstance(response.actions[1], KeyPressAction)


class TestSubprocessAgentRunner:
    def test_successful_agent(self, sample_case, tmp_path):
        """Test subprocess agent that echoes a valid response."""
        agent_script = tmp_path / "agent.py"
        agent_script.write_text(
            textwrap.dedent("""\
            import sys, json
            data = json.loads(sys.stdin.read())
            response = {
                "actions": [
                    {"type": "click", "x": 100, "y": 200, "button": "left"},
                    {"type": "press", "keys": ["enter"]}
                ],
                "meta": {"agent": "test"}
            }
            print(json.dumps(response))
            """)
        )

        runner = SubprocessAgentRunner(cmd=f"{sys.executable} {agent_script}", timeout=10.0)
        response = runner.run_case(sample_case)
        assert len(response.actions) == 2
        assert isinstance(response.actions[0], ClickAction)
        assert response.actions[0].x == 100

    def test_agent_timeout(self, sample_case, tmp_path):
        """Test that timeout is raised for slow agents."""
        agent_script = tmp_path / "slow_agent.py"
        agent_script.write_text(
            textwrap.dedent("""\
            import time, sys
            sys.stdin.read()
            time.sleep(60)
            """)
        )

        runner = SubprocessAgentRunner(cmd=f"{sys.executable} {agent_script}", timeout=1.0)
        with pytest.raises(AgentRunnerError, match="timed out"):
            runner.run_case(sample_case)

    def test_agent_invalid_json(self, sample_case, tmp_path):
        """Test handling of invalid JSON output."""
        agent_script = tmp_path / "bad_agent.py"
        agent_script.write_text(
            textwrap.dedent("""\
            import sys
            sys.stdin.read()
            print("not valid json")
            """)
        )

        runner = SubprocessAgentRunner(cmd=f"{sys.executable} {agent_script}", timeout=10.0)
        with pytest.raises(AgentRunnerError, match="Invalid JSON"):
            runner.run_case(sample_case)

    def test_agent_nonzero_exit(self, sample_case, tmp_path):
        """Test handling of non-zero exit code."""
        agent_script = tmp_path / "fail_agent.py"
        agent_script.write_text(
            textwrap.dedent("""\
            import sys
            sys.stdin.read()
            sys.exit(1)
            """)
        )

        runner = SubprocessAgentRunner(cmd=f"{sys.executable} {agent_script}", timeout=10.0)
        with pytest.raises(AgentRunnerError, match="exited with code"):
            runner.run_case(sample_case)

    def test_agent_command_not_found(self, sample_case):
        runner = SubprocessAgentRunner(cmd="nonexistent_command_xyz")
        with pytest.raises(AgentRunnerError, match="not found"):
            runner.run_case(sample_case)


class TestBuiltinAgentRunner:
    def test_with_mock_provider(self, sample_case):
        """Test BuiltinAgentRunner with MockVLMProvider."""
        from computer_use_test.utils.llm_provider.base import MockVLMProvider

        provider = MockVLMProvider()
        runner = BuiltinAgentRunner(provider=provider)
        response = runner.run_case(sample_case)
        assert len(response.actions) == 2
        assert isinstance(response.actions[0], ClickAction)
        assert response.meta["provider"] == "mock"
