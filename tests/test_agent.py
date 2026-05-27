"""Tests for Agent.step, tool dispatch, and memory helpers."""
from __future__ import annotations

from typing import Any

import pytest

from fleet.agents.agent import Agent
from fleet.agents.memory import ScratchpadMemory, VectorMemory
from fleet.agents.planner import Planner
from fleet.core.messages import AgentMessage, ToolCall
from fleet.core.state import GraphState
from fleet.tools.base import tool


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _MockLLM:
    """Duck-typed LLM that yields pre-canned AgentMessage responses."""

    def __init__(self, responses: list[AgentMessage]) -> None:
        self._responses = list(responses)
        self._idx = 0
        self.calls: list[tuple[list[AgentMessage], Any]] = []

    async def complete(
        self,
        messages: list[AgentMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AgentMessage:
        self.calls.append((list(messages), tools))
        resp = self._responses[self._idx]
        self._idx = min(self._idx + 1, len(self._responses) - 1)
        return resp

    @property
    def usage(self) -> None:
        return None


def _text(content: str) -> AgentMessage:
    return AgentMessage(role="assistant", content=content)


def _tool_call_msg(name: str, args: dict, call_id: str = "tc_abc123") -> AgentMessage:
    return AgentMessage(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id=call_id, name=name, arguments=args)],
    )


# ---------------------------------------------------------------------------
# Agent.step — happy paths
# ---------------------------------------------------------------------------

async def test_step_text_only_response():
    """LLM responds with plain text; agent appends it and stops."""
    llm = _MockLLM([_text("Hello!")])
    agent = Agent(llm, tools=[])
    state = GraphState(goal="say hi", messages=[AgentMessage(role="user", content="Hi")])

    result = await agent.step(state)

    assert result.messages[-1].role == "assistant"
    assert result.messages[-1].content == "Hello!"
    assert len(llm.calls) == 1


async def test_step_tool_call_then_text():
    """LLM → tool call → LLM → text; two LLM calls, one tool executed."""
    # Register a dummy tool for this test
    @tool
    async def _echo_agent_test(text: str) -> str:
        """Echo text."""
        return f"echoed: {text}"

    llm = _MockLLM([
        _tool_call_msg("_echo_agent_test", {"text": "hi"}, "tc_001"),
        _text("Done!"),
    ])
    agent = Agent(llm, tools=["_echo_agent_test"])
    state = GraphState(goal="test", messages=[AgentMessage(role="user", content="go")])

    result = await agent.step(state)

    assert len(llm.calls) == 2
    # Sequence: user → assistant(tool_call) → tool_result → assistant(text)
    roles = [m.role for m in result.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    # Tool result contains echoed text
    assert "echoed: hi" in result.messages[2].tool_results[0].content
    assert result.messages[-1].content == "Done!"


async def test_step_tool_result_is_error_on_exception():
    """If a tool raises, the tool-result message carries is_error=True."""
    @tool
    async def _boom_test() -> str:
        """Explodes."""
        raise RuntimeError("kaboom")

    llm = _MockLLM([
        _tool_call_msg("_boom_test", {}, "tc_boom"),
        _text("handled"),
    ])
    agent = Agent(llm, tools=["_boom_test"])
    state = GraphState(goal="test", messages=[AgentMessage(role="user", content="go")])

    result = await agent.step(state)

    tool_msg = next(m for m in result.messages if m.role == "tool")
    assert tool_msg.tool_results[0].is_error is True
    assert "kaboom" in tool_msg.tool_results[0].content


async def test_step_unknown_tool_returns_error_message():
    """A tool call for an unregistered tool gets an error result (no crash)."""
    llm = _MockLLM([
        _tool_call_msg("__nonexistent_tool__", {}, "tc_miss"),
        _text("ok"),
    ])
    agent = Agent(llm, tools=[])
    state = GraphState(goal="test", messages=[AgentMessage(role="user", content="x")])

    result = await agent.step(state)

    tool_msg = next(m for m in result.messages if m.role == "tool")
    assert tool_msg.tool_results[0].is_error is True
    assert "not registered" in tool_msg.tool_results[0].content


async def test_step_max_iters_stops_loop():
    """Agent stops after max_iters even if LLM keeps returning tool calls."""
    @tool
    async def _noop_test() -> str:
        """Does nothing."""
        return "ok"

    # Always returns a tool call
    always_tool = _tool_call_msg("_noop_test", {}, "tc_x")
    llm = _MockLLM([always_tool] * 20)
    agent = Agent(llm, tools=["_noop_test"], max_iters=3)
    state = GraphState(goal="loop", messages=[AgentMessage(role="user", content="go")])

    await agent.step(state)

    # 3 LLM calls maximum
    assert len(llm.calls) == 3


async def test_step_multiple_tool_calls_in_one_turn():
    """All tool_calls in a single assistant message are dispatched."""
    @tool
    async def _add_test(a: int, b: int) -> str:
        """Add two numbers."""
        return str(a + b)

    llm = _MockLLM([
        AgentMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(id="tc_1", name="_add_test", arguments={"a": 1, "b": 2}),
                ToolCall(id="tc_2", name="_add_test", arguments={"a": 3, "b": 4}),
            ],
        ),
        _text("all done"),
    ])
    agent = Agent(llm, tools=["_add_test"])
    state = GraphState(goal="add", messages=[AgentMessage(role="user", content="go")])

    result = await agent.step(state)

    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert len(tool_msgs) == 2
    assert "3" in tool_msgs[0].tool_results[0].content
    assert "7" in tool_msgs[1].tool_results[0].content


async def test_step_accumulates_state_messages():
    """Each call to step appends onto state.messages."""
    llm = _MockLLM([_text("reply")])
    agent = Agent(llm)
    state = GraphState(goal="chat", messages=[AgentMessage(role="user", content="hello")])

    result = await agent.step(state)
    assert len(result.messages) == 2  # user + assistant


async def test_step_seeds_goal_as_user_message():
    """When messages is empty and goal is set, goal is prepended as a user message."""
    llm = _MockLLM([_text("answer")])
    agent = Agent(llm)
    state = GraphState(goal="What is 2+2?")

    result = await agent.step(state)

    assert result.messages[0].role == "user"
    assert result.messages[0].content == "What is 2+2?"
    assert result.messages[1].role == "assistant"


async def test_step_no_seed_when_messages_present():
    """Goal seeding is skipped when messages already exist."""
    llm = _MockLLM([_text("answer")])
    agent = Agent(llm)
    existing = AgentMessage(role="user", content="existing message")
    state = GraphState(goal="ignored goal", messages=[existing])

    result = await agent.step(state)

    assert result.messages[0].content == "existing message"
    assert len(result.messages) == 2


# ---------------------------------------------------------------------------
# ScratchpadMemory
# ---------------------------------------------------------------------------

def test_scratchpad_get_missing_returns_default():
    state = GraphState(goal="t")
    mem = ScratchpadMemory(state, "node_a")
    assert mem.get("key", "fallback") == "fallback"


def test_scratchpad_set_and_get():
    state = GraphState(goal="t")
    mem = ScratchpadMemory(state, "node_a")
    new_state = mem.set("x", 42)
    mem2 = ScratchpadMemory(new_state, "node_a")
    assert mem2.get("x") == 42


def test_scratchpad_as_dict():
    state = GraphState(goal="t", scratchpad={"node_a": {"k": "v"}})
    mem = ScratchpadMemory(state, "node_a")
    assert mem.as_dict() == {"k": "v"}


def test_scratchpad_set_returns_new_state():
    state = GraphState(goal="t")
    mem = ScratchpadMemory(state, "n")
    new_state = mem.set("a", 1)
    assert new_state is not state
    assert state.scratchpad.get("n") is None  # original unchanged


# ---------------------------------------------------------------------------
# VectorMemory (stub)
# ---------------------------------------------------------------------------

async def test_vector_memory_add_raises():
    vm = VectorMemory()
    with pytest.raises(NotImplementedError):
        await vm.add("some text")


async def test_vector_memory_search_raises():
    vm = VectorMemory()
    with pytest.raises(NotImplementedError):
        await vm.search("query")


# ---------------------------------------------------------------------------
# Planner (no LLM — placeholder path)
# ---------------------------------------------------------------------------

async def test_planner_no_llm_writes_scratchpad():
    planner = Planner(llm=None)
    state = GraphState(goal="build a rocket")
    result = await planner.plan(state)

    assert "planner" in result.scratchpad
    assert "build a rocket" in result.scratchpad["planner"]["subgoals"]
    # An assistant message was appended
    assert any(m.role == "assistant" for m in result.messages)


async def test_planner_with_mock_llm():
    llm = _MockLLM([_text("1. Step one\n2. Step two")])
    planner = Planner(llm=llm)
    state = GraphState(goal="write tests")

    result = await planner.plan(state)

    assert result.scratchpad["planner"]["subgoals"] == "1. Step one\n2. Step two"
    assert len(llm.calls) == 1
