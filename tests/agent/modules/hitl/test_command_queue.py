"""Unit tests for CommandQueue — push, drain, peek, thread safety."""

import threading

from computer_use_test.agent.modules.hitl.command_queue import CommandQueue, Directive, DirectiveType


def _make_directive(dtype: DirectiveType = DirectiveType.CHANGE_STRATEGY, payload: str = "test") -> Directive:
    return Directive(directive_type=dtype, payload=payload, source="test")


class TestCommandQueueBasic:
    def test_push_and_drain(self):
        q = CommandQueue()
        d1 = _make_directive(payload="a")
        d2 = _make_directive(payload="b")
        q.push(d1)
        q.push(d2)
        assert q.size == 2
        assert q.has_pending()

        drained = q.drain()
        assert len(drained) == 2
        assert drained[0].payload == "a"
        assert drained[1].payload == "b"
        assert q.size == 0
        assert not q.has_pending()

    def test_peek_does_not_remove(self):
        q = CommandQueue()
        q.push(_make_directive(payload="peek_test"))
        peeked = q.peek()
        assert len(peeked) == 1
        assert q.size == 1  # Still there

    def test_clear(self):
        q = CommandQueue()
        q.push(_make_directive())
        q.push(_make_directive())
        q.clear()
        assert q.size == 0
        assert not q.has_pending()

    def test_maxlen_evicts_oldest(self):
        q = CommandQueue(maxlen=3)
        for i in range(5):
            q.push(_make_directive(payload=str(i)))
        assert q.size == 3
        drained = q.drain()
        assert [d.payload for d in drained] == ["2", "3", "4"]

    def test_drain_empty_returns_empty_list(self):
        q = CommandQueue()
        assert q.drain() == []


class TestCommandQueueThreadSafety:
    def test_concurrent_push_drain(self):
        q = CommandQueue(maxlen=200)  # Large enough to hold all items
        total_pushed = 100
        collected = []

        def pusher():
            for i in range(total_pushed):
                q.push(_make_directive(payload=str(i)))

        def drainer():
            while len(collected) < total_pushed:
                items = q.drain()
                collected.extend(items)

        t_push = threading.Thread(target=pusher)
        t_drain = threading.Thread(target=drainer)
        t_push.start()
        t_drain.start()
        t_push.join()
        # Give drainer time to finish collecting
        t_drain.join(timeout=5)

        # Drain any remaining
        collected.extend(q.drain())
        assert len(collected) == total_pushed

    def test_wait_unblocks_on_push(self):
        q = CommandQueue()
        result = []

        def waiter():
            q.wait(timeout=5)
            result.append("unblocked")

        t = threading.Thread(target=waiter)
        t.start()
        q.push(_make_directive())
        t.join(timeout=2)
        assert result == ["unblocked"]
