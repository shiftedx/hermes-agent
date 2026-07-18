"""End-to-end: the per-turn tool budget drives the real conversation loop.

Exercised through ``AIAgent.run_conversation`` against an in-process mock
provider so we assert what actually goes on the wire: tools are offered on the
first N completion calls and omitted on every call after the Nth executed tool
call, forcing a final answer. The real ``_execute_tool_calls`` keeps the tally;
only the tool *dispatch* is stubbed so no real tool runs.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class _MockHandler(BaseHTTPRequestHandler):
    captured_requests: list = []
    response_queue: list = []

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length).decode())
        type(self).captured_requests.append(req)
        is_stream = req.get("stream") is True
        if type(self).response_queue:
            resp = type(self).response_queue.pop(0)
        else:
            resp = _text_resp("DONE")
        msg = resp["choices"][0]["message"]
        if is_stream:
            content = msg.get("content") or ""
            tcs = msg.get("tool_calls")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [{"id": "m", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}]
            if content:
                chunks.append({"id": "m", "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]})
            if tcs:
                for ti, tc in enumerate(tcs):
                    chunks.append({"id": "m", "choices": [{"index": 0, "delta": {"tool_calls": [{
                        "index": ti, "id": tc["id"], "type": "function",
                        "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}]}, "finish_reason": None}]})
            chunks.append({"id": "m", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tcs else "stop"}]})
            for c in chunks:
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            body = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *a, **kw):
        pass


def _tc_resp(name: str, call_id: str, args: str = "{}") -> dict:
    return {
        "id": "m",
        "choices": [{"index": 0, "message": {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name, "arguments": args}}]},
            "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


def _text_resp(text: str) -> dict:
    return {
        "id": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


@pytest.fixture()
def agent_env():
    _MockHandler.captured_requests = []
    _MockHandler.response_queue = []
    srv = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    test_home = tempfile.mkdtemp(prefix="hermes_tool_budget_e2e_")
    os.makedirs(os.path.join(test_home, ".hermes"))
    prev_home = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = os.path.join(test_home, ".hermes")

    for mod in list(sys.modules):
        if mod == "run_agent" or mod.startswith("agent.") or mod.startswith("tools.") or mod.startswith("hermes_"):
            del sys.modules[mod]
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key", base_url=f"http://127.0.0.1:{port}/v1",
        provider="openai-compat", model="test-model",
        max_iterations=10, enabled_toolsets=[],
        quiet_mode=True, skip_context_files=True, skip_memory=True,
        save_trajectories=False, platform="cli",
    )
    # Give the agent a real (small) toolset so build_api_kwargs has tools to
    # offer on the wire; the budget's job is to withhold this list.
    _names = ["session_search", "terminal", "read_file"]
    agent.tools = [
        {"type": "function", "function": {
            "name": n, "description": f"{n} tool",
            "parameters": {"type": "object", "properties": {}}}}
        for n in _names
    ]
    agent.valid_tool_names = set(_names)

    # Stub only the dispatch: append a tool result per call, no real execution.
    # The real _execute_tool_calls still runs (and keeps the budget tally).
    def _fake_dispatch(assistant_message, messages, effective_task_id, api_call_count=0):
        for tc in (assistant_message.tool_calls or []):
            messages.append({
                "role": "tool",
                "tool_call_id": getattr(tc, "id", "call_x"),
                "name": tc.function.name,
                "content": "ok",
            })

    agent._execute_tool_calls_sequential = _fake_dispatch
    agent._execute_tool_calls_concurrent = _fake_dispatch

    try:
        yield agent, _MockHandler
    finally:
        srv.shutdown()
        shutil.rmtree(test_home, ignore_errors=True)
        if prev_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = prev_home


def _has_tools(req: dict) -> bool:
    return bool(req.get("tools"))


def _has_wrapup_note(req: dict) -> bool:
    """Whether the request carries the one-time tool-budget wrap-up system note."""
    from agent.chat_completion_helpers import _TOOL_BUDGET_WRAPUP_NOTE

    return any(
        isinstance(m, dict)
        and m.get("role") == "system"
        and m.get("content") == _TOOL_BUDGET_WRAPUP_NOTE
        for m in (req.get("messages") or [])
    )


def _chat_requests(handler) -> list:
    """Only the chat-completion calls (skip model-probe / warmup POSTs)."""
    return [r for r in handler.captured_requests if "messages" in r]


def test_e2e_tools_withheld_on_wire_after_budget(agent_env):
    """Budget N=2: the first two completion calls carry ``tools``; the third —
    after the second executed tool call — omits ``tools`` entirely, so the model
    must answer. Exactly three requests hit the wire (no extra loop)."""
    agent, handler = agent_env
    agent.max_tools_per_turn = 2
    handler.response_queue = [
        _tc_resp("session_search", "c1", '{"q":"one"}'),
        _tc_resp("terminal", "c2", '{"command":"echo two"}'),
        _text_resp("Here is the final answer."),
    ]

    result = agent.run_conversation("do the multi-step thing", conversation_history=[], task_id="t")

    reqs = _chat_requests(handler)
    assert len(reqs) == 3
    assert _has_tools(reqs[0]) is True
    assert _has_tools(reqs[1]) is True
    assert _has_tools(reqs[2]) is False          # budget reached → tools withheld
    assert "tool_choice" not in reqs[2]          # omitted, not tool_choice:"none"
    assert result["final_response"] == "Here is the final answer."


def test_e2e_default_off_offers_tools_on_every_call(agent_env):
    """Budget off (default 0): every completion call carries ``tools`` — the
    loop is byte-identical to today (only the model's own ``stop`` ends it)."""
    agent, handler = agent_env
    assert agent.max_tools_per_turn == 0
    handler.response_queue = [
        _tc_resp("session_search", "c1", '{"q":"one"}'),
        _tc_resp("terminal", "c2", '{"command":"echo two"}'),
        _tc_resp("read_file", "c3", '{"path":"/x"}'),
        _text_resp("All done."),
    ]

    agent.run_conversation("keep going", conversation_history=[], task_id="t")

    reqs = _chat_requests(handler)
    assert len(reqs) == 4
    assert all(_has_tools(r) for r in reqs)      # tools offered on every call


def test_e2e_wrapup_note_on_wire_when_tools_withheld(agent_env):
    """Budget N=2: the tool-free third request carries BOTH omitted tools AND the
    one-time wrap-up system note — the fix that keeps an enforcement-prompted
    model from returning empty once tools disappear. The note rides exactly one
    request, and a plain-text answer ends the turn cleanly."""
    agent, handler = agent_env
    agent.max_tools_per_turn = 2
    handler.response_queue = [
        _tc_resp("session_search", "c1", '{"q":"one"}'),
        _tc_resp("terminal", "c2", '{"command":"echo two"}'),
        _text_resp("Here is the final answer, in plain text."),
    ]

    result = agent.run_conversation("do the multi-step thing",
                                    conversation_history=[], task_id="t")

    reqs = _chat_requests(handler)
    assert len(reqs) == 3
    # Under budget: tools offered, no wrap-up note.
    assert _has_tools(reqs[0]) is True and _has_wrapup_note(reqs[0]) is False
    assert _has_tools(reqs[1]) is True and _has_wrapup_note(reqs[1]) is False
    # Budget reached: tools withheld AND the wrap-up note present.
    assert _has_tools(reqs[2]) is False
    assert _has_wrapup_note(reqs[2]) is True
    # Exactly one request across the turn carries the note.
    assert sum(_has_wrapup_note(r) for r in reqs) == 1
    assert result["final_response"] == "Here is the final answer, in plain text."


def test_e2e_no_wrapup_note_when_budget_off(agent_env):
    """Budget off (default 0): no request ever carries the wrap-up note."""
    agent, handler = agent_env
    assert agent.max_tools_per_turn == 0
    handler.response_queue = [
        _tc_resp("session_search", "c1", '{"q":"one"}'),
        _text_resp("All done."),
    ]

    agent.run_conversation("keep going", conversation_history=[], task_id="t")

    reqs = _chat_requests(handler)
    assert all(not _has_wrapup_note(r) for r in reqs)


def test_e2e_budget_tripped_ack_text_is_accepted_as_final(agent_env):
    """Composition with enforcement: with tool_use_enforcement / intent-ack
    active and the budget tripped, an ack-like final text on the tool-free call
    is accepted as the final answer — the harness does NOT nudge/loop for
    another tool call. Only N+1 requests hit the wire."""
    agent, handler = agent_env
    agent.max_tools_per_turn = 1
    agent._intent_ack_continuation = True  # enforcement active for all api_modes
    handler.response_queue = [
        _tc_resp("read_file", "c1", '{"path":"/repo"}'),
        # An "I'll go do X" ack that WOULD trip intent-ack continuation if tools
        # were still offered — but the budget has withheld them.
        _text_resp("I'll start by inspecting the repository files next."),
    ]

    result = agent.run_conversation("look into the repo files and report back",
                                    conversation_history=[], task_id="t")

    reqs = _chat_requests(handler)
    assert len(reqs) == 2                        # no third (nudge) request
    assert _has_tools(reqs[0]) is True
    assert _has_tools(reqs[1]) is False
    assert result["final_response"] == "I'll start by inspecting the repository files next."
