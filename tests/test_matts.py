"""Tests for fleet.memory.matts — parallel rollouts + contrastive distillation."""

from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from fleet.cli import app
from fleet.core.messages import AgentMessage
from fleet.core.state import GraphState
from fleet.memory import ReasoningBank, matts_run
from fleet.memory.matts import _resolve_bank
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


class _ScriptedLLM:
    """LLM whose ``complete`` returns a queue of pre-written responses."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[list[AgentMessage]] = []

    async def complete(self, messages, tools=None):
        self.calls.append(list(messages))
        if not self._replies:
            raise AssertionError("ScriptedLLM ran out of replies")
        return AgentMessage(role="assistant", content=self._replies.pop(0))

    @property
    def usage(self):
        return None


class _StubRunnable:
    """Async-runnable stub that mimics a CompiledGraph.

    Each invocation of ``run`` appends a unique assistant message so trajectories
    differ between rollouts. Optionally exposes ``memory_bank`` to test
    auto-resolution.
    """

    def __init__(self, memory_bank: Any = None) -> None:
        self.memory_bank = memory_bank
        self.calls = 0

    async def run(self, state: GraphState) -> GraphState:
        idx = self.calls
        self.calls += 1
        msg = AgentMessage(
            role="assistant", content=f"rollout-{idx}: {state.goal}"
        )
        return state.model_copy(update={"messages": [*state.messages, msg]})


def _bank(induction_llm: Any = None) -> ReasoningBank:
    return ReasoningBank(
        store=InMemoryStore(),
        embedder=_StaticEmbedder(),
        induction_llm=induction_llm,
        async_writeback=False,
    )


# ---------------------------------------------------------------------------
# matts_run — happy path
# ---------------------------------------------------------------------------


async def test_matts_run_executes_k_rollouts_and_calls_contrast() -> None:
    contrast = _ScriptedLLM(
        [
            json.dumps(
                [
                    {
                        "title": "Validate tool output shape early",
                        "description": "When chaining tools whose output feeds the next call",
                        "content": "Winning trajectories checked schema before passing data on; "
                        "losing ones crashed on unexpected shapes.",
                    }
                ]
            )
        ]
    )
    bank = _bank(induction_llm=contrast)
    runnable = _StubRunnable()

    states, distilled = await matts_run(runnable, "do a thing", bank=bank, k=3)

    assert runnable.calls == 3
    assert len(states) == 3
    # Each rollout left its own message in its own state.
    contents = sorted(s.messages[-1].content for s in states)
    assert contents == ["rollout-0: do a thing", "rollout-1: do a thing", "rollout-2: do a thing"]

    # Contrast LLM was invoked once and saw all three trajectories.
    assert len(contrast.calls) == 1
    prompt = contrast.calls[0][0].content or ""
    assert "Trajectory 1" in prompt and "Trajectory 2" in prompt and "Trajectory 3" in prompt
    assert "do a thing" in prompt

    # Distilled item stored with source=matts_contrast.
    assert len(distilled) == 1
    assert distilled[0].source == "matts_contrast"
    assert distilled[0].title == "Validate tool output shape early"
    stored = await bank.list()
    assert any(it.id == distilled[0].id for it in stored)


async def test_matts_run_uses_explicit_contrast_llm_over_bank_induction() -> None:
    bank_llm = _ScriptedLLM(["[]"])  # would error if used
    override = _ScriptedLLM([json.dumps([])])
    bank = _bank(induction_llm=bank_llm)
    runnable = _StubRunnable()

    states, distilled = await matts_run(
        runnable, "task", bank=bank, k=2, contrast_llm=override
    )

    assert len(states) == 2
    assert distilled == []
    assert len(override.calls) == 1
    assert bank_llm.calls == []  # override took precedence


async def test_matts_run_resolves_bank_from_runnable_attribute() -> None:
    contrast = _ScriptedLLM([json.dumps([])])
    bank = _bank(induction_llm=contrast)
    runnable = _StubRunnable(memory_bank=bank)

    states, distilled = await matts_run(runnable, "task", k=2)

    assert len(states) == 2
    assert distilled == []


# ---------------------------------------------------------------------------
# matts_run — error paths
# ---------------------------------------------------------------------------


async def test_matts_run_k_zero_raises() -> None:
    bank = _bank(induction_llm=_ScriptedLLM([]))
    with pytest.raises(ValueError, match="k must be >= 1"):
        await matts_run(_StubRunnable(), "task", bank=bank, k=0)


async def test_matts_run_no_bank_raises() -> None:
    runnable = _StubRunnable()  # no memory_bank attribute set to a real bank
    with pytest.raises(ValueError, match="no memory_bank configured"):
        await matts_run(runnable, "task", k=2)


async def test_matts_run_no_contrast_llm_raises() -> None:
    bank = _bank(induction_llm=None)
    with pytest.raises(ValueError, match="no contrast LLM"):
        await matts_run(_StubRunnable(), "task", bank=bank, k=2)


async def test_matts_run_rejects_non_array_response() -> None:
    bad = _ScriptedLLM([json.dumps({"not": "an array"})])
    bank = _bank(induction_llm=bad)
    with pytest.raises(ValueError, match="must be a JSON array"):
        await matts_run(_StubRunnable(), "task", bank=bank, k=2)


# ---------------------------------------------------------------------------
# _resolve_bank
# ---------------------------------------------------------------------------


def test_resolve_bank_walks_graph_nodes() -> None:
    bank = _bank(induction_llm=_ScriptedLLM([]))

    class _Owner:
        def __init__(self) -> None:
            self.memory_bank = bank

        async def step(self, state: GraphState) -> GraphState:
            return state

    class _Node:
        def __init__(self, fn: Any) -> None:
            self.fn = fn

    class _Graph:
        def __init__(self, nodes: dict) -> None:
            self.nodes = nodes

    owner = _Owner()
    graph = _Graph({"a": _Node(owner.step)})
    assert _resolve_bank(graph) is bank


def test_resolve_bank_returns_none_when_missing() -> None:
    class _Bare:
        pass

    assert _resolve_bank(_Bare()) is None


# ---------------------------------------------------------------------------
# CLI --matts flag
# ---------------------------------------------------------------------------


runner = CliRunner()


_MATTS_GRAPH_SRC = '''
from fleet import Graph
from fleet.core.state import GraphState


async def step(state: GraphState) -> GraphState:
    return state


graph = (
    Graph("matts-test")
    .add_node("step", step)
    .set_entry("step")
    .set_exit("step")
    .compile()
)
'''


def test_cli_matts_without_bank_errors(tmp_path) -> None:
    gf = tmp_path / "g.py"
    gf.write_text(_MATTS_GRAPH_SRC)

    result = runner.invoke(app, ["run", str(gf), "--goal", "hi", "--matts", "2"])
    assert result.exit_code == 1, result.output
    assert "memory_bank" in result.output


def test_cli_matts_rejects_zero(tmp_path) -> None:
    gf = tmp_path / "g.py"
    gf.write_text(_MATTS_GRAPH_SRC)

    result = runner.invoke(app, ["run", str(gf), "--goal", "hi", "--matts", "0"])
    assert result.exit_code == 1, result.output
    assert "k >= 1" in result.output
