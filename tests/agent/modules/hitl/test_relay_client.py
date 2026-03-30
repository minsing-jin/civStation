"""Tests for RelayClient message routing logic."""

from __future__ import annotations

from unittest.mock import MagicMock

from civStation.agent.modules.hitl.command_queue import CommandQueue, DirectiveType
from civStation.agent.modules.hitl.relay.relay_client import RelayClient


def _make_client(agent_gate=None, command_queue=None) -> RelayClient:
    return RelayClient(
        url="wss://relay.example.com/ws",
        token="test-token",
        agent_gate=agent_gate,
        command_queue=command_queue,
    )


# ---------------------------------------------------------------------------
# Control routing
# ---------------------------------------------------------------------------


class TestControlRouting:
    def test_start_routed_to_agent_gate(self):
        gate = MagicMock()
        gate.start.return_value = True
        client = _make_client(agent_gate=gate)
        client._handle_control("start")
        gate.start.assert_called_once()

    def test_stop_routed_to_agent_gate(self):
        gate = MagicMock()
        gate.stop.return_value = True
        client = _make_client(agent_gate=gate)
        client._handle_control("stop")
        gate.stop.assert_called_once()

    def test_pause_routed_to_agent_gate(self):
        gate = MagicMock()
        gate.pause.return_value = True
        client = _make_client(agent_gate=gate)
        client._handle_control("pause")
        gate.pause.assert_called_once()

    def test_resume_routed_to_agent_gate(self):
        gate = MagicMock()
        gate.resume.return_value = True
        client = _make_client(agent_gate=gate)
        client._handle_control("resume")
        gate.resume.assert_called_once()

    def test_unknown_action_ignored(self):
        gate = MagicMock()
        client = _make_client(agent_gate=gate)
        # Should not raise; no gate methods called
        client._handle_control("unknown_action")
        gate.start.assert_not_called()
        gate.stop.assert_not_called()

    def test_no_agent_gate_does_not_raise(self):
        client = _make_client(agent_gate=None)
        client._handle_control("start")  # should be a no-op, not raise

    def test_action_is_case_insensitive(self):
        gate = MagicMock()
        gate.pause.return_value = True
        client = _make_client(agent_gate=gate)
        client._handle_control("PAUSE")
        gate.pause.assert_called_once()


# ---------------------------------------------------------------------------
# Command routing
# ---------------------------------------------------------------------------


class TestCommandRouting:
    def test_command_pushed_as_change_strategy(self):
        queue = CommandQueue()
        client = _make_client(command_queue=queue)
        client._handle_command("과학 승리에 집중해")
        directives = queue.drain()
        assert len(directives) == 1
        assert directives[0].directive_type == DirectiveType.CHANGE_STRATEGY
        assert directives[0].payload == "과학 승리에 집중해"
        assert directives[0].source == "relay"

    def test_empty_command_ignored(self):
        queue = CommandQueue()
        client = _make_client(command_queue=queue)
        client._handle_command("")
        assert queue.size == 0

    def test_no_queue_does_not_raise(self):
        client = _make_client(command_queue=None)
        client._handle_command("some command")  # should be no-op, not raise


# ---------------------------------------------------------------------------
# Status broadcast format
# ---------------------------------------------------------------------------


class TestStatusBroadcast:
    def test_send_status_enqueues_correct_type(self):
        """send_status wraps payload in {"type":"status","data":...}."""
        import asyncio

        client = _make_client()
        # Set up a real event loop and outbound queue to test the format
        loop = asyncio.new_event_loop()
        client._loop = loop
        client._outbound = asyncio.Queue()

        data = {"agent_state": "running", "game_turn": 5}
        client.send_status(data)

        # Drain the asyncio queue synchronously
        item = loop.run_until_complete(client._outbound.get())
        loop.close()

        assert item["type"] == "status"
        assert item["data"] == data
