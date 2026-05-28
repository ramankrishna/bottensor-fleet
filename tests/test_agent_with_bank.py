"""Tests for Agent + ReasoningBank integration."""

from __future__ import annotations

from fleet.agents.agent import Agent
from fleet.core.messages import AgentMessage
from fleet.core.state import GraphState
from fleet.memory import ReasoningBank
from fleet.memory.stores.inmemory import InMemoryStore


class _StaticEmbedder:
    DIM = 8

    @property
    def dim(self) -> int:
        return self.DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.DIM
            for i, ch in enumerate(text.lower()):
                vec[(ord(ch) + i) % self.DIM] += 1.0
            n = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / n for v in vec])
        return out


class _RecordingLLM:
    """LLM that records the messages it sees and returns a plain text reply."""

    def __init__(self, reply: str = "ok") -> None:
        self.calls: list[list[AgentMessage]] = []
        self._reply = reply

    async def complete(self, messages, tools=None):
        self.calls.append(list(messages))
        return AgentMessage(role="assistant", content=self._reply)

    @property
    def usage(self):
        return None


def _make_bank() -> ReasoningBank:
    return ReasoningBank(
        store=InMemoryStore(),
        embedder=_StaticEmbedder(),
        async_writeback=False,
    )


async def test_no_memories_no_injection() -> None:
    bank = _make_bank()
    llm = _RecordingLLM()
    agent = Agent(llm=llm, memory_bank=bank, memory_k=3)

    state = GraphState(goal="solve the puzzle")
    await agent.step(state)

    # No system message injected when bank is empty.
    roles = [m.role for m in llm.calls[0]]
    assert "system" not in roles


async def test_memories_appear_in_prompt_on_second_run() -> None:
    bank = _make_bank()
    # Seed the bank with a memory relevant to the goal.
    await bank.add_manual(
        title="Always paginate API responses",
        description="When listing items from external APIs",
        content="Use page=N and stop at a hard cap to avoid infinite loops.",
    )

    llm = _RecordingLLM()
    agent = Agent(llm=llm, memory_bank=bank, memory_k=3)

    state = GraphState(goal="paginate the users list")
    await agent.step(state)

    system_msgs = [m for m in llm.calls[0] if m.role == "system"]
    assert len(system_msgs) == 1
    body = system_msgs[0].content or ""
    assert "Always paginate API responses" in body
    assert "Relevant past learnings" in body


async def test_memory_k_zero_skips_injection() -> None:
    bank = _make_bank()
    await bank.add_manual(title="X", description="d", content="c")

    llm = _RecordingLLM()
    agent = Agent(llm=llm, memory_bank=bank, memory_k=0)
    await agent.step(GraphState(goal="anything"))

    roles = [m.role for m in llm.calls[0]]
    assert "system" not in roles


async def test_two_runs_one_populates_one_uses() -> None:
    """First run: bank starts empty, no injection. Manually ingest. Second run:
    memory is present and appears in the prompt."""
    bank = _make_bank()
    llm = _RecordingLLM()
    agent = Agent(llm=llm, memory_bank=bank, memory_k=5)

    # First run — empty bank, no system message.
    await agent.step(GraphState(goal="how to handle 429s"))
    assert all(m.role != "system" for m in llm.calls[0])

    # Manually populate the bank as if a writeback had landed.
    await bank.add_manual(
        title="Back off on 429",
        description="When an API returns 429",
        content="Sleep with exponential backoff before retrying.",
    )

    # Second run — bank now has the memory.
    await agent.step(GraphState(goal="how to handle 429s"))
    system_msgs = [m for m in llm.calls[1] if m.role == "system"]
    assert len(system_msgs) == 1
    assert "Back off on 429" in (system_msgs[0].content or "")

    # use_count incremented on the stored item.
    items = await bank.list()
    assert items[0].use_count == 1
