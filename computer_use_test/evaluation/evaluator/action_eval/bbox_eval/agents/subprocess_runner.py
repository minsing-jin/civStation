"""
Subprocess agent runner — communicates with an external agent via stdin/stdout.

Protocol:
    - Sends JSON to stdin:   {"instruction": ..., "screenshot_path": ..., "image_size": {...}}
    - Reads JSON from stdout: {"actions": [...], "meta": {...}}

Example:
    >>> runner = SubprocessAgentRunner(cmd="python my_agent.py", timeout=10.0)
    >>> response = runner.run_case(case)
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess

from ..schema import AgentResponse, DatasetCase
from .base import AgentRunnerError, BaseAgentRunner

logger = logging.getLogger(__name__)


class SubprocessAgentRunner(BaseAgentRunner):
    """Runs an external agent as a subprocess."""

    def __init__(self, cmd: str, timeout: float = 10.0):
        self.cmd = cmd
        self.timeout = timeout

    def run_case(self, case: DatasetCase) -> AgentResponse:
        request = json.dumps(
            {
                "instruction": case.instruction,
                "screenshot_path": case.screenshot_path,
                "image_size": {"width": case.image_size.width, "height": case.image_size.height},
            }
        )

        try:
            args = shlex.split(self.cmd)
            result = subprocess.run(
                args,
                input=request,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise AgentRunnerError(f"Agent timed out after {self.timeout}s for case {case.case_id}") from e
        except FileNotFoundError as e:
            raise AgentRunnerError(f"Agent command not found: {self.cmd}") from e

        if result.returncode != 0:
            stderr = result.stderr.strip()[:500] if result.stderr else ""
            raise AgentRunnerError(f"Agent exited with code {result.returncode} for case {case.case_id}: {stderr}")

        stdout = result.stdout.strip()
        if not stdout:
            raise AgentRunnerError(f"Agent returned empty stdout for case {case.case_id}")

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise AgentRunnerError(f"Invalid JSON from agent for case {case.case_id}: {e}") from e

        try:
            return AgentResponse.model_validate(data)
        except Exception as e:
            raise AgentRunnerError(f"Failed to parse agent response for case {case.case_id}: {e}") from e
