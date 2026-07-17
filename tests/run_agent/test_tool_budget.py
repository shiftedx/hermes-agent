"""Per-turn tool-call budget (``agent.max_tools_per_turn``).

Once a turn has dispatched its allotment of tool calls, every remaining
completion call in that turn is made tool-free — the ``tools`` parameter is
omitted entirely — so a small local model that loops many successful tool calls
is forced to produce a final answer instead of another tool call.

These are behavior contracts about the budget's relationship to the request
builder and the intent-ack enforcement gate, not snapshots of any value.
"""

import sys
import types
from types import SimpleNamespace

from unittest.mock import patch


sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

from run_agent import AIAgent


# ── Helpers ──────────────────────────────────────────────────────────────────

def _tool_defs(*names):
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


class _FakeOpenAI:
    def __init__(self, **kw):
        self.api_key = kw.get("api_key", "test")
        self.base_url = kw.get("base_url", "http://test")

    def close(self):
        pass


def _make_agent(monkeypatch, *, max_tools_per_turn=0, base_url="https://openrouter.ai/api/v1"):
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **kw: _tool_defs("web_search", "terminal"))
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("run_agent.OpenAI", _FakeOpenAI)
    return AIAgent(
        api_key="test",
        base_url=base_url,
        api_mode="chat_completions",
        max_iterations=8,
        max_tools_per_turn=max_tools_per_turn,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def _tool_call(name="web_search"):
    return SimpleNamespace(
        id=f"call_{name}",
        type="function",
        function=SimpleNamespace(name=name, arguments="{}"),
    )


def _assistant_with_tools(*names):
    return SimpleNamespace(tool_calls=[_tool_call(n) for n in names], content=None)


def _tools_offered(agent):
    """Whether ``_build_api_kwargs`` would offer tools right now."""
    kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
    return bool(kwargs.get("tools"))


# ── _tool_budget_reached predicate ───────────────────────────────────────────

def test_budget_off_by_default_never_reached(monkeypatch):
    agent = _make_agent(monkeypatch)  # max_tools_per_turn defaults to 0
    assert agent.max_tools_per_turn == 0
    agent._tools_dispatched_this_turn = 999
    assert agent._tool_budget_reached() is False


def test_budget_trips_once_tally_meets_limit(monkeypatch):
    agent = _make_agent(monkeypatch, max_tools_per_turn=2)
    agent._tools_dispatched_this_turn = 0
    assert agent._tool_budget_reached() is False
    agent._tools_dispatched_this_turn = 1
    assert agent._tool_budget_reached() is False
    agent._tools_dispatched_this_turn = 2
    assert agent._tool_budget_reached() is True
    agent._tools_dispatched_this_turn = 3
    assert agent._tool_budget_reached() is True


# ── build_api_kwargs: offer vs withhold ──────────────────────────────────────

def test_default_offers_tools_on_every_call(monkeypatch):
    """Budget off (0): tools are always offered — byte-identical to before."""
    agent = _make_agent(monkeypatch)
    for tally in (0, 1, 5, 50):
        agent._tools_dispatched_this_turn = tally
        assert _tools_offered(agent) is True


def test_tools_withheld_once_budget_reached(monkeypatch):
    agent = _make_agent(monkeypatch, max_tools_per_turn=2)
    agent._tools_dispatched_this_turn = 1
    assert _tools_offered(agent) is True
    agent._tools_dispatched_this_turn = 2
    assert _tools_offered(agent) is False


def test_withheld_call_omits_tools_key_entirely(monkeypatch):
    """Preferred shape: omit ``tools`` rather than send an empty list or
    ``tool_choice: "none"`` — most compatible with local OpenAI-compat servers."""
    agent = _make_agent(monkeypatch, max_tools_per_turn=1)
    agent._tools_dispatched_this_turn = 1
    kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs


def test_budget_two_offers_then_withholds_after_second_tool_call(monkeypatch):
    """N=2: the first two completion calls offer tools; the call after the
    second executed tool call offers none. Counting flows through the real
    dispatch choke point ``_execute_tool_calls``."""
    agent = _make_agent(monkeypatch, max_tools_per_turn=2)
    agent._tools_dispatched_this_turn = 0

    # Neutralize the actual executor — we only care about the tally the
    # dispatch entrypoint keeps.
    monkeypatch.setattr("run_agent._should_parallelize_tool_batch", lambda _tc: False)
    monkeypatch.setattr(agent, "_execute_tool_calls_sequential", lambda *a, **k: None)

    # Completion call #1 — under budget, tools offered. Model calls one tool.
    assert _tools_offered(agent) is True
    agent._execute_tool_calls(_assistant_with_tools("web_search"), [], "task")
    assert agent._tools_dispatched_this_turn == 1

    # Completion call #2 — still under budget, tools offered. Model calls one tool.
    assert _tools_offered(agent) is True
    agent._execute_tool_calls(_assistant_with_tools("terminal"), [], "task")
    assert agent._tools_dispatched_this_turn == 2

    # Completion call #3 — budget reached, tools withheld → model must answer.
    assert _tools_offered(agent) is False


def test_parallel_batch_counts_every_call_in_the_batch(monkeypatch):
    """A single batched completion that dispatches 3 tool calls tips a budget
    of 2 — the next completion call is tool-free."""
    agent = _make_agent(monkeypatch, max_tools_per_turn=2)
    agent._tools_dispatched_this_turn = 0
    monkeypatch.setattr("run_agent._should_parallelize_tool_batch", lambda _tc: False)
    monkeypatch.setattr(agent, "_execute_tool_calls_sequential", lambda *a, **k: None)

    assert _tools_offered(agent) is True
    agent._execute_tool_calls(_assistant_with_tools("web_search", "terminal", "web_search"), [], "task")
    assert agent._tools_dispatched_this_turn == 3
    assert _tools_offered(agent) is False


# ── config plumbing ──────────────────────────────────────────────────────────

def test_config_value_reaches_the_agent(monkeypatch):
    """``agent.max_tools_per_turn`` in config.yaml reaches the AIAgent instance,
    the same way ``agent.max_turns`` reaches ``max_iterations``."""
    from hermes_cli.config import load_config as _real_load

    base = _real_load()
    base["agent"]["max_tools_per_turn"] = 3
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: base)

    agent = _make_agent(monkeypatch)  # no explicit constructor arg
    assert agent.max_tools_per_turn == 3


def test_constructor_arg_reaches_the_agent(monkeypatch):
    agent = _make_agent(monkeypatch, max_tools_per_turn=5)
    assert agent.max_tools_per_turn == 5


def test_negative_config_value_falls_back_to_off(monkeypatch):
    from hermes_cli.config import load_config as _real_load

    base = _real_load()
    base["agent"]["max_tools_per_turn"] = -4
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: base)

    agent = _make_agent(monkeypatch)
    assert agent.max_tools_per_turn == 0
    agent._tools_dispatched_this_turn = 10
    assert agent._tool_budget_reached() is False


def test_non_int_config_value_falls_back_to_off(monkeypatch):
    from hermes_cli.config import load_config as _real_load

    base = _real_load()
    base["agent"]["max_tools_per_turn"] = "lots"
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: base)

    agent = _make_agent(monkeypatch)
    assert agent.max_tools_per_turn == 0


def test_constructor_arg_wins_over_config(monkeypatch):
    from hermes_cli.config import load_config as _real_load

    base = _real_load()
    base["agent"]["max_tools_per_turn"] = 2
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: base)

    agent = _make_agent(monkeypatch, max_tools_per_turn=7)
    assert agent.max_tools_per_turn == 7


# ── counter reset / subagent isolation ───────────────────────────────────────

def test_counter_starts_fresh_on_a_new_agent_instance(monkeypatch):
    """A subagent / background-review fork is a fresh AIAgent, so it never
    inherits a parent's exhausted budget counter."""
    parent = _make_agent(monkeypatch, max_tools_per_turn=2)
    parent._tools_dispatched_this_turn = 2
    assert parent._tool_budget_reached() is True

    child = _make_agent(monkeypatch, max_tools_per_turn=2)
    assert child._tools_dispatched_this_turn == 0
    assert child._tool_budget_reached() is False


# ── composition with tool_use_enforcement / intent_ack_continuation ──────────

def _intent_ack_gate(agent, *, ack_mode, codex_ack_continuations, user_message,
                     assistant_content, messages):
    """Faithful mirror of the intent-ack enforcement gate in
    ``agent/conversation_loop.py``. Enforcement fires only when this is True."""
    return (
        ack_mode != "off"
        and not agent._tool_budget_reached()
        and bool(agent.valid_tool_names)
        and codex_ack_continuations < 2
        and agent._looks_like_codex_intermediate_ack(
            user_message=user_message,
            assistant_content=assistant_content,
            messages=messages,
            require_workspace=(ack_mode == "codex_only"),
        )
    )


def test_enforcement_fires_while_tools_are_still_offered(monkeypatch):
    """Control: with the budget NOT tripped, an intent ack still nudges the
    model to actually call a tool (enforcement behaves exactly as today)."""
    agent = _make_agent(monkeypatch, max_tools_per_turn=2)
    agent._intent_ack_continuation = True  # enforcement active for all api_modes
    agent._tools_dispatched_this_turn = 0  # budget not reached

    user = "look into the repo files and report back"
    ack = "I'll start by inspecting the repository files."
    messages = [{"role": "user", "content": user}]

    # Sanity: the detector recognizes this as an intermediate ack.
    assert agent._looks_like_codex_intermediate_ack(
        user_message=user, assistant_content=ack, messages=messages, require_workspace=False
    ) is True
    # Gate fires → enforcement would nudge for a tool call.
    assert _intent_ack_gate(
        agent, ack_mode="all", codex_ack_continuations=0,
        user_message=user, assistant_content=ack, messages=messages,
    ) is True


def test_budget_tripped_suppresses_enforcement_nudge(monkeypatch):
    """The core composition: once the budget has withheld tools, the text answer
    is accepted as final — enforcement does NOT retry/nudge for a tool call,
    even when the detector would otherwise recognize an intermediate ack."""
    agent = _make_agent(monkeypatch, max_tools_per_turn=2)
    agent._intent_ack_continuation = True  # enforcement active
    agent._tools_dispatched_this_turn = 2  # budget reached → tools withheld

    user = "look into the repo files and report back"
    ack = "I'll start by inspecting the repository files."
    # No tool results in history, so ONLY the budget veto can suppress the gate
    # (isolates the new gate from the detector's own has-tool-result guard).
    messages = [{"role": "user", "content": user}]

    # Detector alone would still recognize the ack …
    assert agent._looks_like_codex_intermediate_ack(
        user_message=user, assistant_content=ack, messages=messages, require_workspace=False
    ) is True
    # … but the budget veto flips the composed gate to False: no nudge.
    assert _intent_ack_gate(
        agent, ack_mode="all", codex_ack_continuations=0,
        user_message=user, assistant_content=ack, messages=messages,
    ) is False
