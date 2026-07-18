"""Race-shape tests for the busy-steer atomicity fix.

A busy-mode steer is folded into a running turn by stashing text into
``agent._pending_steer`` (event-loop / busy-handler thread) which the turn
finalizer later drains (executor / worker thread).  Those two are
cross-thread and were NOT atomic: a ``steer()`` that lost the race with the
turn's terminal drain got stranded on an already-finalized agent and was
lost forever (production "shape 1"), or — combined with a resubmit of the
same text — double-delivered (production "shape 2").

The fix seals the steer slot atomically with the terminal drain
(``_seal_and_drain_pending_steer``): a steer is EITHER captured by that
drain (delivered as the leftover/next turn) OR rejected by a now-sealed
``steer()`` (re-queued by the gateway as exactly one follow-up turn).  It is
never stranded, and never both folded AND resubmitted.

These tests exercise the seal boundary directly and under real thread
contention — no asyncio, no live provider — so they are deterministic.
"""
from __future__ import annotations

import threading

import pytest

from run_agent import AIAgent


def _bare_agent() -> AIAgent:
    """AIAgent with just the steer state installed (matches the
    object.__new__ stub pattern used across the suite)."""
    agent = object.__new__(AIAgent)
    agent._pending_steer = None
    agent._pending_steer_lock = threading.Lock()
    agent._pending_steer_sealed = False
    return agent


class TestSealBoundary:
    """Shape 1: a steer that lands after the turn's terminal drain must be
    rejected, never stranded."""

    def test_seal_and_drain_captures_pre_seal_steer(self):
        # Steer stashed BEFORE the terminal drain is returned and delivered.
        agent = _bare_agent()
        assert agent.steer("check auth.log too") is True
        captured = agent._seal_and_drain_pending_steer()
        assert captured == "check auth.log too"
        assert agent._pending_steer is None
        # The slot is now sealed.
        assert agent._pending_steer_sealed is True

    def test_steer_rejected_after_seal(self):
        # Steer arriving AFTER the terminal drain (the losing side of the
        # race) is rejected so the gateway re-queues it instead of stranding
        # it on a finalized agent.
        agent = _bare_agent()
        agent._seal_and_drain_pending_steer()  # turn ends, slot sealed
        assert agent.steer("landed too late") is False
        assert agent._pending_steer is None  # nothing stashed

    def test_seal_is_terminal_until_reset(self):
        # Multiple late steers all rejected until the next turn resets the
        # seal (run_conversation prologue).
        agent = _bare_agent()
        agent._seal_and_drain_pending_steer()
        assert agent.steer("a") is False
        assert agent.steer("b") is False
        assert agent._pending_steer is None
        # Next turn begins → prologue clears the seal.
        agent._pending_steer_sealed = False
        assert agent.steer("fresh turn") is True
        assert agent._pending_steer == "fresh turn"

    def test_empty_slot_seal_returns_none_still_seals(self):
        agent = _bare_agent()
        assert agent._seal_and_drain_pending_steer() is None
        assert agent._pending_steer_sealed is True
        assert agent.steer("rejected") is False


class TestMidLoopDrainDoesNotSeal:
    """Shape 3 (between calls) must be UNCHANGED: the non-sealing mid-loop
    drains keep accepting steers on the still-running turn."""

    def test_plain_drain_does_not_seal(self):
        agent = _bare_agent()
        agent.steer("first")
        assert agent._drain_pending_steer() == "first"
        # NOT sealed — the turn is still going, more steers are welcome.
        assert agent._pending_steer_sealed is False
        assert agent.steer("second") is True
        assert agent._pending_steer == "second"

    def test_fold_then_more_steers_accepted(self):
        # A steer folded mid-loop, then another arrives before turn end —
        # both accepted (shape 3 semantics preserved).
        agent = _bare_agent()
        agent.steer("note one")
        messages = [{"role": "tool", "content": "out", "tool_call_id": "1"}]
        agent._apply_pending_steer_to_tool_results(messages, num_tool_msgs=1)
        assert agent._pending_steer is None
        assert agent._pending_steer_sealed is False
        assert agent.steer("note two") is True


class TestSealRaceThreaded:
    """The atomicity property under real contention: for every trial, the
    steer text is accounted for EXACTLY once — either the terminal drain
    returned it, or steer() rejected it. It is never accepted-but-orphaned
    (the shape-1 loss) and never both drained AND accepted (the shape-2
    double)."""

    def test_concurrent_steer_and_terminal_drain_no_loss_no_double(self):
        for _ in range(2000):
            agent = _bare_agent()
            drained_holder: list = []
            accepted_holder: list = []
            start = threading.Barrier(2)

            def _steerer():
                start.wait()
                accepted_holder.append(agent.steer("payload"))

            def _finalizer():
                start.wait()
                drained_holder.append(agent._seal_and_drain_pending_steer())

            t1 = threading.Thread(target=_steerer)
            t2 = threading.Thread(target=_finalizer)
            t1.start(); t2.start()
            t1.join(); t2.join()

            accepted = accepted_holder[0]
            drained = drained_holder[0]

            # Exactly-once accounting: accepted XOR still-strandable.
            if accepted:
                # steer() won the race → the terminal drain MUST have
                # captured it (delivered as the leftover/next turn).
                assert drained == "payload", (
                    "accepted steer was not captured by the terminal drain — "
                    "it would be stranded (shape 1)"
                )
            else:
                # steer() was rejected (seal already set) → the gateway will
                # re-queue it; the drain must NOT have also produced it, or it
                # would be delivered twice (shape 2).
                assert drained is None, (
                    "rejected steer was ALSO drained — double delivery (shape 2)"
                )
            # In all cases the slot is left clean and sealed.
            assert agent._pending_steer is None
            assert agent._pending_steer_sealed is True


class TestStubFallbackNoLockPath:
    """The no-lock stub path (object.__new__ without a real lock) must honor
    the same seal contract so stub-based callers behave identically."""

    def test_nolock_seal_rejects_subsequent_steer(self):
        agent = object.__new__(AIAgent)
        agent._pending_steer = None
        # No _pending_steer_lock and no _pending_steer_sealed installed.
        assert agent.steer("before") is True
        assert agent._seal_and_drain_pending_steer() == "before"
        assert agent._pending_steer_sealed is True
        assert agent.steer("after") is False


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
