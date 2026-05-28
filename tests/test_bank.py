"""Tests for fleet.memory.bank.ReasoningBank — the public API."""

from __future__ import annotations

import json
from typing import Any

import pytest

from fleet.core.messages import AgentMessage
from fleet.memory import ReasoningBank
from fleet.memory.stores.inmemory import InMemoryStore


class _StaticEmbedder:
    """Deterministic embedder driven by simple word features."""

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
            # length-normalize for stable cosine
            n = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / n for v in vec])
        return out


class _ScriptedLLM:
    def __init__(self, *responses: Any) -> None:
        self._responses = list(responses)
        self._idx = 0
        self.calls: list[list[AgentMessage]] = []

    async def complete(self, messages, tools=None):
        self.calls.append(list(messages))
        if not self._responses:
            raise AssertionError("no scripted response left")
        item = self._responses[self._idx]
        self._idx = min(self._idx + 1, len(self._responses) - 1)
        text = item if isinstance(item, str) else json.dumps(item)
        return AgentMessage(role="assistant", content=text)

    @property
    def usage(self):
        return None


def _bank(*, judge=None, induction=None, async_writeback: bool = False) -> ReasoningBank:
    return ReasoningBank(
        store=InMemoryStore(),
        embedder=_StaticEmbedder(),
        judge_llm=judge,
        induction_llm=induction,
        async_writeback=async_writeback,
    )


# ---------------------------------------------------------------------------
# add_manual + retrieve
# ---------------------------------------------------------------------------

async def test_add_manual_then_retrieve() -> None:
    bank = _bank()
    item = await bank.add_manual(
        title="Always paginate",
        description="When listing API resources",
        content="Use ?page=N with a hard cap on iterations.",
    )
    assert item.id in {it.id for it in await bank.list()}
    results = await bank.retrieve("how to paginate API listings", k=1)
    assert len(results) == 1
    assert results[0].id == item.id
    assert results[0].use_count == 1


async def test_list_and_delete() -> None:
    bank = _bank()
    a = await bank.add_manual(title="A", description="d", content="c")
    b = await bank.add_manual(title="B", description="d", content="c")
    ids = {it.id for it in await bank.list()}
    assert {a.id, b.id} <= ids
    await bank.delete(a.id)
    ids = {it.id for it in await bank.list()}
    assert a.id not in ids and b.id in ids


# ---------------------------------------------------------------------------
# ingest_trajectory — full round trip with mocked LLMs
# ---------------------------------------------------------------------------

async def test_ingest_trajectory_with_explicit_outcome() -> None:
    induction = _ScriptedLLM(
        [
            {"title": "Lesson 1", "description": "When X", "content": "Do Y"},
        ]
    )
    bank = _bank(induction=induction)
    trajectory = [AgentMessage(role="user", content="task"), AgentMessage(role="assistant", content="done")]
    items = await bank.ingest_trajectory(trajectory, "task", outcome="success")
    assert len(items) == 1
    assert items[0].source == "success"
    assert items[0].embedding is not None
    # Item lives in the store too.
    assert items[0].id in {it.id for it in await bank.list()}


async def test_ingest_trajectory_uses_judge_when_outcome_none() -> None:
    judge = _ScriptedLLM({"outcome": "failure", "rationale": "no answer"})
    induction = _ScriptedLLM(
        [{"title": "Bad path", "description": "Avoid X", "content": "Because Y"}]
    )
    bank = _bank(judge=judge, induction=induction)
    items = await bank.ingest_trajectory(
        [AgentMessage(role="user", content="?")], "?", outcome=None
    )
    assert items[0].source == "failure"


async def test_ingest_trajectory_without_judge_when_outcome_none_raises() -> None:
    bank = _bank(induction=_ScriptedLLM([]))
    with pytest.raises(ValueError):
        await bank.ingest_trajectory(
            [AgentMessage(role="user", content="?")], "?", outcome=None
        )


async def test_ingest_trajectory_empty_returns_empty() -> None:
    bank = _bank(induction=_ScriptedLLM([]))
    assert await bank.ingest_trajectory([], "task", outcome="success") == []


# ---------------------------------------------------------------------------
# merge integration — second ingest with identical content collapses
# ---------------------------------------------------------------------------

async def test_ingest_merges_near_duplicate() -> None:
    induction = _ScriptedLLM(
        [{"title": "Same lesson", "description": "Same desc", "content": "Same body"}],
        [{"title": "Same lesson", "description": "Same desc", "content": "Same body"}],
        # merge prompt response — induction LLM is reused for merge calls
        {"action": "merge", "merged_title": "Merged", "merged_description": "m", "merged_content": "b"},
    )
    bank = _bank(induction=induction)
    first = await bank.ingest_trajectory(
        [AgentMessage(role="user", content="t")], "t", outcome="success"
    )
    second = await bank.ingest_trajectory(
        [AgentMessage(role="user", content="t")], "t", outcome="success"
    )
    assert len(first) == 1 and len(second) == 1
    # Still only one item in the store — the second got merged.
    assert len(await bank.list()) == 1
